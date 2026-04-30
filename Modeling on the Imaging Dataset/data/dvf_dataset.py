"""
dvf_dataset — Longitudinal DVF Dataset and normalization utilities for
the ADNI Advanced Survival Transformer pipeline.

Phase 1 patch changes vs. the pre-fix version:
- LongitudinalDVFDataset.__init__ now does ID-aware label matching via
  inner_join_with_folder_ids instead of positional slicing.
- NormalizationStats.compute RNG is instantiated ONCE outside the loop.
- Zero-visit subjects are filtered out with a warning.
- All other methods (__len__, __getitem__, collate_fn, etc.) are unchanged.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from Transformer.data.label_io import (
        DURATION_FNAME,
        EVENT_FNAME,
        SUBJECT_IDS_FNAME,
        inner_join_with_folder_ids,
        load_survival_labels,
    )
except ImportError:
    from label_io import (  # type: ignore
        DURATION_FNAME,
        EVENT_FNAME,
        SUBJECT_IDS_FNAME,
        inner_join_with_folder_ids,
        load_survival_labels,
    )


logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Per-channel outlier-clip bounds and z-score statistics for DVFs."""

    p_low: np.ndarray
    p_high: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def compute(
        cls,
        dvf_paths: Sequence[Path],
        subject_indices: Optional[Sequence[int]] = None,
    ) -> "NormalizationStats":
        """Compute per-channel normalisation statistics from DVF files.

        Uses reservoir sampling + Welford's online algorithm.
        """
        paths = (
            [dvf_paths[i] for i in subject_indices]
            if subject_indices is not None
            else list(dvf_paths)
        )
        if len(paths) == 0:
            raise ValueError("dvf_paths is empty after filtering.")

        n_samples_per_file = 200_000
        reservoir: List[np.ndarray] = []

        rng = np.random.RandomState(42)

        for p in paths:
            dvf = np.load(str(p), mmap_mode="r")  # [3, 128, 128, 128]
            n_voxels = dvf.shape[1] * dvf.shape[2] * dvf.shape[3]
            idx = rng.choice(
                n_voxels,
                size=min(n_samples_per_file, n_voxels),
                replace=False,
            )
            flat = dvf.reshape(3, -1)[:, idx]
            reservoir.append(np.array(flat))

        pooled = np.concatenate(reservoir, axis=1)
        p_low = np.percentile(pooled, 0.5, axis=1).astype(np.float32)
        p_high = np.percentile(pooled, 99.5, axis=1).astype(np.float32)

        # Pass 2: Welford on the clipped distribution.
        count = np.zeros(3, dtype=np.float64)
        welford_mean = np.zeros(3, dtype=np.float64)
        welford_m2 = np.zeros(3, dtype=np.float64)

        for p in paths:
            dvf = np.load(str(p), mmap_mode="r")
            for c in range(3):
                channel = np.clip(
                    dvf[c].ravel().astype(np.float64),
                    p_low[c],
                    p_high[c],
                )
                for val in _chunk_iter(channel, chunk_size=65536):
                    batch_count = val.shape[0]
                    batch_mean = val.mean()
                    batch_var = val.var()
                    delta = batch_mean - welford_mean[c]
                    total = count[c] + batch_count
                    new_mean = welford_mean[c] + delta * batch_count / total
                    m2_delta = (
                        batch_var * batch_count
                        + delta ** 2 * count[c] * batch_count / total
                    )
                    welford_m2[c] += m2_delta
                    welford_mean[c] = new_mean
                    count[c] = total

        final_mean = welford_mean.astype(np.float32)
        final_std = np.sqrt(welford_m2 / count).astype(np.float32)
        final_std = np.maximum(final_std, 1e-7)

        return cls(p_low=p_low, p_high=p_high, mean=final_mean, std=final_std)

    def apply(self, dvf: np.ndarray) -> np.ndarray:
        """Apply outlier clipping and z-score normalisation per channel."""
        if dvf.shape[0] != 3:
            raise ValueError(f"Expected 3 channels, got {dvf.shape[0]}")
        out = dvf.astype(np.float32, copy=True)
        for c in range(3):
            out[c] = np.clip(out[c], self.p_low[c], self.p_high[c])
            out[c] = (out[c] - self.mean[c]) / (self.std[c] + 1e-7)
        return out

    def save(self, path: Path) -> None:
        """Pickle-serialize normalization stats to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        """Deserialize normalization stats from a pickle file."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected NormalizationStats, got {type(obj).__name__}")
        return obj


def _chunk_iter(arr: np.ndarray, chunk_size: int = 65536):
    """Yield successive chunks of a 1-D array."""
    for start in range(0, len(arr), chunk_size):
        yield arr[start : start + chunk_size]


class LongitudinalDVFDataset(Dataset):
    """PyTorch Dataset yielding per-subject longitudinal DVF sequences.
        list by PTID, not by row index.
    """

    def __init__(
        self,
        subject_ids: List[str],
        dvf_dir: Path,
        config,
        norm_stats: NormalizationStats,
        survival_labels_dir: Path,
        tabular_path: Optional[Path] = None,
        *,
        label_subject_ids: Optional[Sequence[str]] = None,
        allow_positional: bool = False,
    ) -> None:
        """Initialize the dataset with ID-aware label matching.

        Args:
            subject_ids: Ordered list of PTIDs to iterate over.
            dvf_dir: Root directory with per-subject DVF folders.
            config: ModelConfig instance.
            norm_stats: Pre-computed NormalizationStats.
            survival_labels_dir: Directory containing mci_y_*.npy files.
            tabular_path: Optional [N_subjects, F] .npy of tabular features.
            label_subject_ids: Explicit row-order of the label arrays.
                If None, mci_subject_ids.npy in survival_labels_dir is used.
            allow_positional: LEGACY ESCAPE HATCH. If True and no ID index
                is available, fall back to positional matching.
        """
        super().__init__()

        self.dvf_dir = Path(dvf_dir)
        self.config = config
        self.norm_stats = norm_stats
        self.v_max = config.v_max

        # Load label arrays
        labels_dir = Path(survival_labels_dir)
        if label_subject_ids is None:
            loaded_ids, durations_all, events_all = load_survival_labels(
                labels_dir, require_ids=False,
            )
            label_ids: List[str] = loaded_ids
        else:
            label_ids = list(label_subject_ids)
            durations_all = np.load(str(labels_dir / DURATION_FNAME))
            events_all = np.load(str(labels_dir / EVENT_FNAME))
            if len(label_ids) != len(durations_all):
                raise ValueError(
                    f"label_subject_ids length ({len(label_ids)}) does not "
                    f"match durations length ({len(durations_all)})"
                )

        # Choose matching strategy
        if label_ids:
            # ID-based inner join — the correct path.
            kept_ids, durations, events = inner_join_with_folder_ids(
                label_subject_ids=label_ids,
                folder_subject_ids=subject_ids,
                durations=durations_all,
                events=events_all,
            )
            dropped = set(subject_ids) - set(kept_ids)
            if dropped:
                logger.warning(
                    "Dropping %d subject(s) with DVF folder but no label "
                    "row: %s%s",
                    len(dropped),
                    list(dropped)[:5],
                    "..." if len(dropped) > 5 else "",
                )
            if not kept_ids:
                raise ValueError(
                    "Inner join of DVF folders and label IDs produced zero "
                    "subjects. Check that your folder names match the PTIDs "
                    "in mci_subject_ids.npy."
                )
            resolved_ids = kept_ids
        elif allow_positional:
            logger.warning(
                "LEGACY positional label matching is active. If your "
                "subject_ids list is in any order other than the one that "
                "produced the label arrays, your labels are silently "
                "shuffled. Pass label_subject_ids or save an ID index."
            )
            if len(durations_all) != len(subject_ids):
                raise ValueError(
                    f"Positional mode requires equal lengths: "
                    f"durations={len(durations_all)}, subject_ids={len(subject_ids)}"
                )
            resolved_ids = list(subject_ids)
            durations = durations_all
            events = events_all
        else:
            raise FileNotFoundError(
                f"No mci_subject_ids.npy found in {labels_dir} and "
                f"label_subject_ids was not provided. Refusing to do "
                f"positional matching (silent label shuffling risk). "
                f"Pass allow_positional=True if you truly want the old "
                f"behavior, or run "
                f"`save_survival_labels(..., subject_ids=...)` upstream."
            )

        # Filter zero-visit subjects
        subject_dvf_paths: List[List[Path]] = []
        subject_visit_times: List[np.ndarray] = []
        keep_rows: List[int] = []
        for row_idx, sid in enumerate(resolved_ids):
            subj_dir = self.dvf_dir / str(sid)
            npy_files = (
                sorted(subj_dir.glob("*.npy"))
                if subj_dir.is_dir()
                else []
            )
            if not npy_files:
                continue
            times = []
            for f in npy_files:
                try:
                    times.append(float(f.stem))
                except ValueError:
                    times.append(float(len(times)))
            subject_dvf_paths.append(npy_files)
            subject_visit_times.append(np.array(times, dtype=np.float32))
            keep_rows.append(row_idx)

        dropped_no_visit = len(resolved_ids) - len(keep_rows)
        if dropped_no_visit:
            logger.warning(
                "Dropping %d subject(s) with a labeled row but no .npy "
                "visit files under %s",
                dropped_no_visit, self.dvf_dir,
            )

        if not keep_rows:
            raise ValueError(
                "Every subject was dropped for having zero visit files. "
                "Check that `dvf_dir` really points at the per-subject folders."
            )

        # Re-index everything to the filtered subset.
        kept_idx = np.asarray(keep_rows, dtype=np.int64)
        self.subject_ids: List[str] = [resolved_ids[i] for i in keep_rows]
        self.durations: np.ndarray = durations[kept_idx]
        self.events: np.ndarray = events[kept_idx]
        self.subject_dvf_paths = subject_dvf_paths
        self.subject_visit_times = subject_visit_times

        # Tabular features
        self.tabular: Optional[np.ndarray]
        if tabular_path is not None and Path(tabular_path).exists():
            tab_full = np.load(str(tabular_path))
            if len(tab_full) == len(durations_all):
                full_row_by_id = {sid: i for i, sid in enumerate(label_ids)}
                pick = [full_row_by_id[sid] for sid in self.subject_ids]
                self.tabular = tab_full[np.asarray(pick, dtype=np.int64)]
            elif len(tab_full) == len(self.subject_ids):
                self.tabular = tab_full
            else:
                logger.warning(
                    "Tabular array has %d rows but subject list has %d — "
                    "ignoring tabular features.",
                    len(tab_full), len(self.subject_ids),
                )
                self.tabular = None
        else:
            self.tabular = None

        logger.info(
            "LongitudinalDVFDataset ready: %d subjects, %d total visits, "
            "v_max=%d",
            len(self.subject_ids),
            sum(len(v) for v in self.subject_dvf_paths),
            self.v_max,
        )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )
        dvf_paths = self.subject_dvf_paths[idx]
        visit_times = self.subject_visit_times[idx]
        n_visits = len(dvf_paths)

        dvf_list = []
        for p in dvf_paths:
            raw = np.load(str(p), mmap_mode="r")
            normed = self.norm_stats.apply(raw)
            dvf_list.append(normed)

        if n_visits > 0:
            dvf_stack = np.stack(dvf_list, axis=0)
        else:
            dvf_stack = np.zeros((0, 3, 128, 128, 128), dtype=np.float32)

        pad_visits = self.v_max - n_visits
        if pad_visits > 0:
            pad_shape = (pad_visits, 3, 128, 128, 128)
            dvf_padded = np.concatenate(
                [dvf_stack, np.zeros(pad_shape, dtype=np.float32)],
                axis=0,
            )
        else:
            dvf_padded = dvf_stack[: self.v_max]
            n_visits = self.v_max

        dvf_sequence = torch.from_numpy(dvf_padded)

        vt = np.zeros(self.v_max, dtype=np.float32)
        actual_n = min(len(visit_times), self.v_max)
        vt[:actual_n] = visit_times[:actual_n]
        visit_times_tensor = torch.from_numpy(vt)

        td = np.zeros(self.v_max, dtype=np.float32)
        for i in range(1, actual_n):
            td[i] = vt[i] - vt[i - 1]
        time_deltas = torch.from_numpy(td)

        mask = np.zeros(self.v_max, dtype=np.int64)
        mask[:actual_n] = 1
        missing_mask = torch.from_numpy(mask)

        if self.tabular is not None:
            tab_row = self.tabular[idx]
            tabular_tensor = torch.from_numpy(
                np.tile(tab_row, (self.v_max, 1)).astype(np.float32)
            )
        else:
            tabular_tensor = torch.zeros(self.v_max, 1, dtype=torch.float32)

        duration = torch.tensor(self.durations[idx], dtype=torch.float32)
        event = torch.tensor(self.events[idx], dtype=torch.int64)

        return {
            "dvf_sequence": dvf_sequence,
            "visit_times": visit_times_tensor,
            "time_deltas": time_deltas,
            "missing_mask": missing_mask,
            "tabular": tabular_tensor,
            "duration": duration,
            "event": event,
            "subject_id": self.subject_ids[idx],
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        """Collate a list of per-subject dicts into a batched dict."""
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty batch.")
        v_dims = [item["dvf_sequence"].shape[0] for item in batch]
        v_max_batch = max(v_dims)

        dvf_list, vt_list, td_list, mask_list = [], [], [], []
        tab_list, dur_list, evt_list, sid_list = [], [], [], []

        for item in batch:
            v_cur = item["dvf_sequence"].shape[0]
            pad_v = v_max_batch - v_cur
            if pad_v > 0:
                dvf = torch.cat(
                    [item["dvf_sequence"],
                     torch.zeros(pad_v, 3, 128, 128, 128, dtype=torch.float32)],
                    dim=0,
                )
                vt = torch.cat([item["visit_times"],
                                torch.zeros(pad_v, dtype=torch.float32)])
                td = torch.cat([item["time_deltas"],
                                torch.zeros(pad_v, dtype=torch.float32)])
                mm = torch.cat([item["missing_mask"],
                                torch.zeros(pad_v, dtype=torch.int64)])
                tab = torch.cat(
                    [item["tabular"],
                     torch.zeros(pad_v, item["tabular"].shape[1], dtype=torch.float32)],
                    dim=0,
                )
            else:
                dvf = item["dvf_sequence"]
                vt = item["visit_times"]
                td = item["time_deltas"]
                mm = item["missing_mask"]
                tab = item["tabular"]

            dvf_list.append(dvf)
            vt_list.append(vt)
            td_list.append(td)
            mask_list.append(mm)
            tab_list.append(tab)
            dur_list.append(item["duration"])
            evt_list.append(item["event"])
            sid_list.append(item["subject_id"])

        return {
            "dvf_sequence": torch.stack(dvf_list, dim=0),
            "visit_times": torch.stack(vt_list, dim=0),
            "time_deltas": torch.stack(td_list, dim=0),
            "missing_mask": torch.stack(mask_list, dim=0),
            "tabular": torch.stack(tab_list, dim=0),
            "duration": torch.stack(dur_list, dim=0),
            "event": torch.stack(evt_list, dim=0),
            "subject_id": sid_list,
        }


if __name__ == "__main__":
    import sys
    import traceback

    _this_dir = Path(__file__).resolve().parent
    _transformer_dir = _this_dir.parent
    _repo_root = _transformer_dir.parent
    if str(_transformer_dir) not in sys.path:
        sys.path.insert(0, str(_transformer_dir))
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from config.model_config import ModelConfig
    try:
        from Transformer.data.label_io import save_survival_labels
    except ImportError:
        from label_io import save_survival_labels  # type: ignore

    passed = 0
    failed = 0
    tmp_dir = None

    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="dvf_smoke_"))
        print(f"[setup] Temporary directory: {tmp_dir}")

        n_subjects = 4
        visits_per_subject = [2, 3, 4, 0]  # last subject has zero visits
        subject_ids = [f"subj_{i:03d}" for i in range(n_subjects)]
        dvf_dir = tmp_dir / "dvf"
        labels_dir = tmp_dir / "labels"
        labels_dir.mkdir(parents=True)

        rng = np.random.RandomState(42)
        all_dvf_paths: List[Path] = []

        for sid, n_vis in zip(subject_ids, visits_per_subject):
            subj_dir = dvf_dir / sid
            subj_dir.mkdir(parents=True)
            for v in range(n_vis):
                arr = rng.randn(3, 128, 128, 128).astype(np.float32)
                fpath = subj_dir / f"{v * 12:03d}.npy"
                np.save(str(fpath), arr)
                all_dvf_paths.append(fpath)

        # Create labels in SCRAMBLED order (different from folder order)
        label_ids_scrambled = ["subj_002", "subj_000", "subj_001", "subj_003"]
        durations_scrambled = np.array([10.0, 5.0, 7.0, 3.0], dtype=np.float32)
        events_scrambled = np.array([1, 0, 1, 0], dtype=np.int64)
        save_survival_labels(
            labels_dir, label_ids_scrambled,
            durations_scrambled, events_scrambled,
        )

        config = ModelConfig(v_max=5, dvf_dir=dvf_dir, survival_labels_dir=labels_dir)
        config.validate()

        # TEST 1: ID-aware inner join returns correct labels
        print("\nTEST 1: ID-aware inner join returns correct labels despite order mismatch")
        ds = LongitudinalDVFDataset(
            subject_ids=subject_ids[:3],  # exclude zero-visit subject
            dvf_dir=dvf_dir, config=config,
            norm_stats=NormalizationStats.compute(all_dvf_paths),
            survival_labels_dir=labels_dir,
        )
        # subj_000 should have duration=5.0, subj_001=7.0, subj_002=10.0
        id_to_dur = {ds.subject_ids[i]: ds.durations[i] for i in range(len(ds))}
        assert abs(id_to_dur["subj_000"] - 5.0) < 1e-5, f"subj_000 dur={id_to_dur['subj_000']}"
        assert abs(id_to_dur["subj_001"] - 7.0) < 1e-5, f"subj_001 dur={id_to_dur['subj_001']}"
        assert abs(id_to_dur["subj_002"] - 10.0) < 1e-5, f"subj_002 dur={id_to_dur['subj_002']}"
        print("[PASS] ID-aware inner join returns correct labels despite order mismatch")
        passed += 1

        # TEST 2: Refuses positional matching when IDs missing
        print("\nTEST 2: Refuses positional matching when mci_subject_ids.npy is absent")
        noid_dir = tmp_dir / "noid_labels"
        noid_dir.mkdir()
        np.save(str(noid_dir / "mci_y_duration.npy"), durations_scrambled[:3])
        np.save(str(noid_dir / "mci_y_event.npy"), events_scrambled[:3])
        try:
            LongitudinalDVFDataset(
                subject_ids=subject_ids[:3], dvf_dir=dvf_dir, config=config,
                norm_stats=NormalizationStats.compute(all_dvf_paths),
                survival_labels_dir=noid_dir,
            )
            print("[FAIL] Should have raised FileNotFoundError")
            failed += 1
        except FileNotFoundError:
            print("[PASS] Refuses positional matching when mci_subject_ids.npy is absent")
            passed += 1

        # TEST 3: allow_positional=True opt-in works
        print("\nTEST 3: allow_positional=True opt-in still works for legacy data")
        ds_pos = LongitudinalDVFDataset(
            subject_ids=subject_ids[:3], dvf_dir=dvf_dir, config=config,
            norm_stats=NormalizationStats.compute(all_dvf_paths),
            survival_labels_dir=noid_dir, allow_positional=True,
        )
        assert len(ds_pos) == 3
        print("[PASS] allow_positional=True opt-in still works for legacy data")
        passed += 1

        # TEST 4: Zero-visit subject is dropped
        print("\nTEST 4: Zero-visit subject is dropped from the iteration list")
        ds_all = LongitudinalDVFDataset(
            subject_ids=subject_ids,  # includes subj_003 with 0 visits
            dvf_dir=dvf_dir, config=config,
            norm_stats=NormalizationStats.compute(all_dvf_paths),
            survival_labels_dir=labels_dir,
        )
        assert "subj_003" not in ds_all.subject_ids, "subj_003 should be dropped"
        assert len(ds_all) == 3
        print("[PASS] Zero-visit subject is dropped from the iteration list")
        passed += 1

    except Exception:
        failed += 1
        traceback.print_exc()

    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print(f"\n[cleanup] Removed {tmp_dir}")

    total = passed + failed
    if failed == 0:
        print(f"\nAll {passed} tests passed.")
    else:
        print(f"\nFAIL — {failed}/{total} tests failed.")
    sys.exit(0 if failed == 0 else 1)
