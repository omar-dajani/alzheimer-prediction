"""
dvf_dataset — Longitudinal DVF Dataset and normalization utilities for
the ADNI Advanced Survival Transformer pipeline.

Pipeline position:
    Phase 0 of the ADNI Advanced Survival Pipeline. This module is the
    entry point for all data flow into the Transformer stack. It runs
    before any model component is instantiated.

Inputs:
    - DVF .npy arrays: [3, 128, 128, 128] float32, one per visit per subject
    - Survival labels: mci_y_duration.npy, mci_y_event.npy from Baseline/outputs/
    - (Optional) ComBat-harmonized tabular features from Baseline pipeline # Talk about option

Outputs:
    - LongitudinalDVFDataset: yields per-subject dicts with keys:
        dvf_sequence, visit_times, time_deltas, missing_mask, tabular,
        duration, event, subject_id
    - NormalizationStats: serializable stats for inference-time normalization

Dependencies:
    - Baseline/outputs/ must exist and contain mci_y_duration.npy, mci_y_event.npy
    - No other Transformer modules are required at this phase
"""

from __future__ import annotations

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


@dataclass
class NormalizationStats:
    """Per-channel outlier-clip bounds and z-score statistics for DVFs.

    This dataclass stores the four quantities needed to normalise any
    DVF tensor identically to the training distribution. A dataclass
    is used because it is pickle-serializable, provides attribute access,
    and integrates cleanly with checkpoint saving.

    Tensor shapes consumed / produced:
        - All stored arrays have shape [3] — one entry per DVF
          displacement channel (delta x, delta y, delta z).
        - apply() consumes [3, D, H, W] and returns the same
          shape after normalization.
    """

    p_low: np.ndarray    # [3] — 0.5th percentile per channel (clip lower bound)
    p_high: np.ndarray   # [3] — 99.5th percentile per channel (clip upper bound)
    mean: np.ndarray     # [3] — per-channel mean (after clipping)
    std: np.ndarray      # [3] — per-channel std  (after clipping)

    @classmethod
    def compute(
        cls,
        dvf_paths: Sequence[Path],
        subject_indices: Optional[Sequence[int]] = None,
    ) -> "NormalizationStats":
        """Compute per-channel normalisation statistics from DVF files.

        Uses Welford's online algorithm to accumulate running mean and
        variance in a single streaming pass over files. Welford's is
        chosen over the naive two-pass approach because it avoids
        loading all training DVFs into RAM simultaneously — a single
        5-visit subject occupies a lot of space, making in-memory accumulation
        infeasible for even a modest cohort.

        Args:
            dvf_paths: Ordered list of .npy file paths, each
                containing a [3, 128, 128, 128] float32 DVF.
            subject_indices: Optional indices into dvf_paths to
                restrict computation to a training subset.  If None
                all paths are used.

        Returns:
            A NormalizationStats instance fitted on the selected
            files.

        Raises:
            ValueError: If dvf_paths is empty (after index
                filtering) or any file cannot be memory-mapped.
        """
        paths = (
            [dvf_paths[i] for i in subject_indices]
            if subject_indices is not None
            else list(dvf_paths)
        )
        if len(paths) == 0:
            raise ValueError("dvf_paths is empty after filtering.")

        n_samples_per_file = 200_000 # ≈ 10 % of 128^3 voxels per channel
        reservoir: List[np.ndarray] = [] # list of [3, n_samples] arrays

        for p in paths:
            # Memory-mapped read — no full array copy into RAM
            dvf = np.load(str(p), mmap_mode="r") # [3, 128, 128, 128]
            n_voxels = dvf.shape[1] * dvf.shape[2] * dvf.shape[3]
            rng = np.random.RandomState(42)
            idx = rng.choice(n_voxels, size=min(n_samples_per_file, n_voxels), replace=False)
            flat = dvf.reshape(3, -1)[:, idx] # [3, n_samples]
            reservoir.append(np.array(flat)) # materialise from mmap

        pooled = np.concatenate(reservoir, axis=1) # [3, total_samples]

        # Per-channel percentile bounds (channels have systematically
        # different displacement distributions — delta x lateral ventricular
        # expansion vs delta z anterior-posterior hippocampal atrophy)
        p_low = np.percentile(pooled, 0.5, axis=1).astype(np.float32)   # [3]
        p_high = np.percentile(pooled, 99.5, axis=1).astype(np.float32)  # [3]

        # Pass 2: Welford's online algorithm for mean/variance after
        # clipping. We iterate over files again because the clipped
        # distribution differs from the raw one, so we cannot reuse
        # the reservoir.

        count = np.zeros(3, dtype=np.float64)
        welford_mean = np.zeros(3, dtype=np.float64)
        welford_m2 = np.zeros(3, dtype=np.float64)

        for p in paths:
            dvf = np.load(str(p), mmap_mode="r")  # [3, 128, 128, 128]
            for c in range(3):
                # Clip channel c at its percentile bounds
                channel = np.clip(
                    dvf[c].ravel().astype(np.float64),
                    p_low[c],
                    p_high[c],
                )  # [128*128*128]

                # Welford's update — numerically stable single-pass
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

        final_mean = welford_mean.astype(np.float32)  # [3]
        final_std = np.sqrt(welford_m2 / count).astype(np.float32)  # [3]

        # Guard against degenerate zero-std channels
        final_std = np.maximum(final_std, 1e-7)

        return cls(
            p_low=p_low,
            p_high=p_high,
            mean=final_mean,
            std=final_std,
        )

    # apply — clip then z-score per channel
    def apply(self, dvf: np.ndarray) -> np.ndarray:
        """Apply outlier clipping and z-score normalisation per channel.

        The three DVF channels (delta x, delta y, delta z) have systematically
        different displacement distributions: lateral ventricular
        expansion primarily affects delta x, while anterior-posterior
        hippocampal atrophy dominates delta z. Normalising jointly would
        bias the z-score toward the channel with the largest dynamic
        range. Per-channel treatment ensures each displacement axis
        contributes equally to downstream feature learning.

        Args:
            dvf: Raw DVF array of shape [3, D, H, W] (float32 or
                float64).

        Returns:
            Normalised DVF of shape [3, D, H, W], dtype float32.

        Raises:
            ValueError: If dvf does not have 3 channels in dim 0.
        """
        if dvf.shape[0] != 3:
            raise ValueError(
                f"Expected 3 channels in dim 0, got {dvf.shape[0]}"
            )

        out = dvf.astype(np.float32, copy=True)  # [3, D, H, W]

        for c in range(3):
            # Step 1: clip at [0.5th, 99.5th] percentile bounds
            # out[c] = clamp(out[c], p_low[c], p_high[c])
            out[c] = np.clip(out[c], self.p_low[c], self.p_high[c])

            # Step 2: z-score
            out[c] = (out[c] - self.mean[c]) / (self.std[c] + 1e-7)

        return out  # [3, D, H, W], float32

    # Serialisation helpers
    def save(self, path: Path) -> None:
        """Pickle-serialize normalization stats to disk.

        Args:
            path: Target file path (e.g., norm_stats.pkl).

        Raises:
            OSError: If the parent directory does not exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path) -> "NormalizationStats":
        """Deserialize normalization stats from a pickle file.

        Args:
            path: Path to a previously saved .pkl file.

        Returns:
            A NormalizationStats instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected NormalizationStats, got {type(obj).__name__}"
            )
        return obj


def _chunk_iter(arr: np.ndarray, chunk_size: int = 65536):
    """Yield successive chunks of a 1-D array.

    Helper for streaming Welford updates without materialising large
    intermediate arrays.
    """
    for start in range(0, len(arr), chunk_size):
        yield arr[start : start + chunk_size]


# LongitudinalDVFDataset — PyTorch Dataset for per-subject DVF sequences
class LongitudinalDVFDataset(Dataset):
    """PyTorch Dataset yielding per-subject longitudinal DVF sequences.

    Each item is a dictionary containing the DVF volume sequence,
    visit timing metadata, a binary missing-visit mask, optional ComBat-
    harmonized tabular features, and survival labels sourced exclusively
    from Baseline/outputs/mci_y_duration.npy and
    mci_y_event.npy. and will be changed based on the final dataset so its meant to be modular

    This design (one dict per subject) is preferred over a flattened
    visit-level dataset because the survival label is inherently a
    subject-level quantity: every visit from the same subject shares
    the same (duration, event) pair. Flattening would either duplicate
    labels or require an awkward grouping step at collation time.

    Tensor shapes consumed / produced:
        - Input DVF files: [3, 128, 128, 128] float32 per visit.
        - Output dvf_sequence: [v_max, 3, 128, 128, 128] —
          zero-padded to config.v_max along the visit axis.
        - Output visit_times:  [v_max] float32.
        - Output time_deltas:  [v_max] float32.
        - Output missing_mask: [v_max] int64 (1=present, 0=pad).
        - Output tabular:      [v_max, n_features] float32.
        - Output duration:     scalar float32.
        - Output event:        scalar int64.
        - Output subject_id:   Python str.
    """

    def __init__(
        self,
        subject_ids: List[str],
        dvf_dir: Path,
        config,
        norm_stats: NormalizationStats,
        survival_labels_dir: Path,
        tabular_path: Optional[Path] = None,
    ) -> None:
        """Initialise the dataset.

        Args:
            subject_ids: Ordered list of subject identifiers.  Each
                subject must have a sub-directory under dvf_dir
                named by its ID, containing one .npy DVF per visit.
            dvf_dir: Root directory containing per-subject DVF folders.
            config: A ModelConfig instance providing v_max and
                other pipeline hyperparameters.
            norm_stats: Pre-computed NormalizationStats fitted on
                the training split.
            survival_labels_dir: Directory containing
                mci_y_duration.npy and mci_y_event.npy.
                These files are the *sole* source of survival labels;
                the Baseline pipeline already applied reversion-removal
                and right-censoring, so they must not be recomputed.
            tabular_path: Optional path to a .npy file of shape
                [N_subjects, n_features] providing per-subject
                ComBat-harmonized features.  If None, a zero tensor
                of shape [v_max, 1] is returned for each sample so
                that batch collation remains uniform.

        Raises:
            FileNotFoundError: If survival label files are missing.
            ValueError: If the number of survival labels does not match
                the number of subject_ids.
        """
        super().__init__()

        self.subject_ids = list(subject_ids)
        self.dvf_dir = Path(dvf_dir)
        self.config = config
        self.norm_stats = norm_stats
        self.v_max = config.v_max

        # Load survival labels — EXCLUSIVELY from Baseline/outputs/
        duration_path = Path(survival_labels_dir) / "mci_y_duration.npy"
        event_path = Path(survival_labels_dir) / "mci_y_event.npy"

        if not duration_path.exists():
            raise FileNotFoundError(
                f"Survival duration labels not found: {duration_path}"
            )
        if not event_path.exists():
            raise FileNotFoundError(
                f"Survival event labels not found: {event_path}"
            )

        self.durations = np.load(str(duration_path)) # [N_subjects]
        self.events = np.load(str(event_path)) # [N_subjects]

        if len(self.durations) != len(self.subject_ids):
            raise ValueError(
                f"Duration label count ({len(self.durations)}) does not "
                f"match subject_ids count ({len(self.subject_ids)})"
            )
        if len(self.events) != len(self.subject_ids):
            raise ValueError(
                f"Event label count ({len(self.events)}) does not "
                f"match subject_ids count ({len(self.subject_ids)})"
            )

        # Modularity for tabular features
        if tabular_path is not None and Path(tabular_path).exists():
            self.tabular = np.load(str(tabular_path))  # [N_subjects, F]
        else:
            self.tabular = None

        # Pre-index per-subject DVF file lists
        self.subject_dvf_paths: List[List[Path]] = []
        self.subject_visit_times: List[np.ndarray] = []

        for sid in self.subject_ids:
            subj_dir = self.dvf_dir / str(sid)
            if subj_dir.is_dir():
                npy_files = sorted(subj_dir.glob("*.npy"))
            else:
                npy_files = []

            self.subject_dvf_paths.append(npy_files)
            # Fabrizio or Omar should edit based on the datasets structure/naming conventions

            times = []
            for f in npy_files:
                stem = f.stem
                try:
                    times.append(float(stem))
                except ValueError:
                    times.append(float(len(times)))
            self.subject_visit_times.append(np.array(times, dtype=np.float32))

    def __len__(self) -> int:
        """Return the number of subjects in the dataset.

        Returns:
            Number of subjects (int).
        """
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Load and return the full per-subject dictionary.

        Pad the visit dimension to config.v_max with zeros and
        update missing_mask accordingly.

        Args:
            idx: Integer index into the dataset.

        Returns:
            Dictionary with keys: dvf_sequence, visit_times,
            time_deltas, missing_mask, tabular, duration, event, subject_id.

        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        dvf_paths = self.subject_dvf_paths[idx]
        visit_times = self.subject_visit_times[idx]
        n_visits = len(dvf_paths)

        # Load DVFs with memory mapping, normalise, pad to v_max
        dvf_list = []
        for p in dvf_paths:
            raw = np.load(str(p), mmap_mode="r") # [3, 128, 128, 128]
            normed = self.norm_stats.apply(raw) # [3, 128, 128, 128]
            dvf_list.append(normed)

        if n_visits > 0:
            dvf_stack = np.stack(dvf_list, axis=0) # [n_visits, 3, 128, 128, 128]
        else:
            # Degenerate case: no visits available for this subject
            dvf_stack = np.zeros(
                (0, 3, 128, 128, 128), dtype=np.float32
            )  # [0, 3, 128, 128, 128]

        # Pad visit dimension to v_max
        pad_visits = self.v_max - n_visits
        if pad_visits > 0:
            pad_shape = (pad_visits, 3, 128, 128, 128)
            dvf_padded = np.concatenate(
                [dvf_stack, np.zeros(pad_shape, dtype=np.float32)],
                axis=0,
            )  # [v_max, 3, 128, 128, 128]
        else:
            # Truncate if more visits than v_max
            dvf_padded = dvf_stack[: self.v_max]  # [v_max, 3, 128, 128, 128]
            n_visits = self.v_max

        dvf_sequence = torch.from_numpy(
            dvf_padded
        )  # [v_max, 3, 128, 128, 128]

        # Visit times and inter-visit deltas
        vt = np.zeros(self.v_max, dtype=np.float32)  # [v_max]
        actual_n = min(len(visit_times), self.v_max)
        vt[:actual_n] = visit_times[:actual_n]
        visit_times_tensor = torch.from_numpy(vt)  # [v_max]

        # time_deltas[0] = 0 (baseline), time_deltas[i] = t[i] - t[i-1]
        td = np.zeros(self.v_max, dtype=np.float32)  # [v_max]
        for i in range(1, actual_n):
            td[i] = vt[i] - vt[i - 1]
        time_deltas = torch.from_numpy(td)  # [v_max]

        # Missing mask: 1 = present, 0 = padded/missing
        mask = np.zeros(self.v_max, dtype=np.int64)  # [v_max]
        mask[:actual_n] = 1
        missing_mask = torch.from_numpy(mask)  # [v_max]

        # Tabular features (optional)
        if self.tabular is not None:
            tab_row = self.tabular[idx]  # [F]
            # Broadcast to [v_max, F] — same tabular at every visit
            tabular_tensor = torch.from_numpy(
                np.tile(tab_row, (self.v_max, 1)).astype(np.float32)
            )  # [v_max, F]
        else:
            # Return a zero tensor so collation remains uniform
            tabular_tensor = torch.zeros(
                self.v_max, 1, dtype=torch.float32
            )  # [v_max, 1]

        # Survival labels
        duration = torch.tensor(
            self.durations[idx], dtype=torch.float32
        )  # scalar
        event = torch.tensor(
            self.events[idx], dtype=torch.int64
        )  # scalar

        return {
            "dvf_sequence": dvf_sequence, # [v_max, 3, 128, 128, 128]
            "visit_times": visit_times_tensor, # [v_max]
            "time_deltas": time_deltas, # [v_max]
            "missing_mask": missing_mask, # [v_max]
            "tabular": tabular_tensor, # [v_max, F] or [v_max, 1]
            "duration": duration, # scalar
            "event": event, # scalar
            "subject_id": self.subject_ids[idx],
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
        """Collate a list of per-subject dicts into a batched dict.

        Pads the visit dimension to the maximum V present in the batch
        (not necessarily v_max from config) and updates
        missing_mask to reflect any additional padding introduced
        by collation.

        This is a @staticmethod so it can be passed directly to
        DataLoader(collate_fn=LongitudinalDVFDataset.collate_fn).

        Args:
            batch: List of dicts as returned by __getitem__.

        Returns:
            Single dict with the same keys, where tensor values are
            stacked along a new batch dimension (dim 0). Non-tensor
            values (subject_id) are collected into a list.

        Raises:
            ValueError: If batch is empty.
        """
        if len(batch) == 0:
            raise ValueError("Cannot collate an empty batch.")

        # Determine the max visit count across the batch.
        # Each sample is already padded to its own v_max, but
        # different configs or subsets could yield different sizes.
        v_dims = [item["dvf_sequence"].shape[0] for item in batch]
        v_max_batch = max(v_dims)

        dvf_list = []
        vt_list = []
        td_list = []
        mask_list = []
        tab_list = []
        dur_list = []
        evt_list = []
        sid_list = []

        for item in batch:
            v_cur = item["dvf_sequence"].shape[0]
            pad_v = v_max_batch - v_cur

            # DVF: pad visit dim if necessary
            if pad_v > 0:
                dvf_pad = torch.zeros(
                    pad_v, 3, 128, 128, 128, dtype=torch.float32
                )  # [pad_v, 3, 128, 128, 128]
                dvf = torch.cat(
                    [item["dvf_sequence"], dvf_pad], dim=0
                )  # [v_max_batch, 3, 128, 128, 128]
            else:
                dvf = item["dvf_sequence"] # [v_max_batch, 3, 128, 128, 128]

            # visit_times: pad with zeros
            if pad_v > 0:
                vt = torch.cat([
                    item["visit_times"],
                    torch.zeros(pad_v, dtype=torch.float32),
                ])  # [v_max_batch]
            else:
                vt = item["visit_times"] # [v_max_batch]

            # time_deltas: pad with zeros
            if pad_v > 0:
                td = torch.cat([
                    item["time_deltas"],
                    torch.zeros(pad_v, dtype=torch.float32),
                ])  # [v_max_batch]
            else:
                td = item["time_deltas"] # [v_max_batch]

            # missing_mask: extra pads are 0 (missing)
            if pad_v > 0:
                mm = torch.cat([
                    item["missing_mask"],
                    torch.zeros(pad_v, dtype=torch.int64),
                ])  # [v_max_batch]
            else:
                mm = item["missing_mask"] # [v_max_batch]

            # tabular: pad visit dim
            tab = item["tabular"]
            if pad_v > 0:
                tab_pad = torch.zeros(
                    pad_v, tab.shape[1], dtype=torch.float32
                ) # [pad_v, F]
                tab = torch.cat([tab, tab_pad], dim=0) # [v_max_batch, F]

            dvf_list.append(dvf)
            vt_list.append(vt)
            td_list.append(td)
            mask_list.append(mm)
            tab_list.append(tab)
            dur_list.append(item["duration"])
            evt_list.append(item["event"])
            sid_list.append(item["subject_id"])

        return {
            "dvf_sequence": torch.stack(dvf_list, dim=0), # [B, v_max_batch, 3, 128, 128, 128]
            "visit_times": torch.stack(vt_list, dim=0), # [B, v_max_batch]
            "time_deltas": torch.stack(td_list, dim=0), # [B, v_max_batch]
            "missing_mask": torch.stack(mask_list, dim=0), # [B, v_max_batch]
            "tabular": torch.stack(tab_list, dim=0), # [B, v_max_batch, F]
            "duration": torch.stack(dur_list, dim=0), # [B]
            "event": torch.stack(evt_list, dim=0), # [B]
            "subject_id": sid_list, # list[str], length B
        }


# Smoke test

if __name__ == "__main__":
    import sys
    import traceback

    # Append parent directories so we can import ModelConfig when run
    # directly (outside of a package install).
    _this_dir = Path(__file__).resolve().parent
    _transformer_dir = _this_dir.parent
    _repo_root = _transformer_dir.parent
    if str(_transformer_dir) not in sys.path:
        sys.path.insert(0, str(_transformer_dir))
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))

    from config.model_config import ModelConfig 

    passed = 0
    failed = 0
    tmp_dir = None

    try:
        # 1. Create temp directory with dummy subjects
        tmp_dir = Path(tempfile.mkdtemp(prefix="dvf_smoke_"))
        print(f"[setup] Temporary directory: {tmp_dir}")

        n_subjects = 3
        visits_per_subject = [2, 3, 4]
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
                # Each visit file: [3, 128, 128, 128] float32
                arr = rng.randn(3, 128, 128, 128).astype(np.float32)
                fpath = subj_dir / f"{v * 12:03d}.npy"
                np.save(str(fpath), arr)
                all_dvf_paths.append(fpath)

        # 2. Create dummy survival labels
        durations = rng.uniform(1, 10, size=n_subjects).astype(np.float64)
        events = rng.randint(0, 2, size=n_subjects).astype(np.int64)
        np.save(str(labels_dir / "mci_y_duration.npy"), durations)
        np.save(str(labels_dir / "mci_y_event.npy"), events)
        print("[setup] Dummy data created.\n")

        # 3. ModelConfig.validate()
        print("TEST 1: ModelConfig instantiation + validate()")
        config = ModelConfig(
            v_max=5,
            dvf_dir=dvf_dir,
            survival_labels_dir=labels_dir,
        )
        config.validate()
        print(" ModelConfig.validate() passed.")
        passed += 1

        # 4. NormalizationStats.compute()
        print("\nTEST 2: NormalizationStats.compute()")
        norm_stats = NormalizationStats.compute(all_dvf_paths)
        print(f"p_low = {norm_stats.p_low}")
        print(f"p_high = {norm_stats.p_high}")
        print(f"mean = {norm_stats.mean}")
        print(f"std = {norm_stats.std}")
        assert norm_stats.p_low.shape == (3,), f"p_low shape: {norm_stats.p_low.shape}"
        assert norm_stats.std.shape == (3,), f"std shape: {norm_stats.std.shape}"

        # Save / load round-trip
        stats_path = tmp_dir / "norm_stats.pkl"
        norm_stats.save(stats_path)
        loaded = NormalizationStats.load(stats_path)
        assert np.allclose(loaded.mean, norm_stats.mean), "Save/load mismatch"
        print("Save/load round-trip passed.")
        passed += 1

        # 5. LongitudinalDVFDataset.__getitem__(0)
        print("\nTEST 3: LongitudinalDVFDataset.__getitem__(0)")
        ds = LongitudinalDVFDataset(
            subject_ids=subject_ids,
            dvf_dir=dvf_dir,
            config=config,
            norm_stats=norm_stats,
            survival_labels_dir=labels_dir,
            tabular_path=None,
        )
        assert len(ds) == n_subjects, f"len(ds) = {len(ds)}"

        sample = ds[0]
        print("Keys and shapes:")
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f"{k:20s} -> {tuple(v.shape)}")
            else:
                print(f"{k:20s} -> {v}")
        passed += 1

        # 6. Shape assertion
        print("\nTEST 4: dvf_sequence shape assertion")
        expected = (config.v_max, 3, 128, 128, 128)
        actual = tuple(sample["dvf_sequence"].shape)
        assert actual == expected, (
            f"Expected {expected}, got {actual}"
        )
        print(f"dvf_sequence.shape = {actual}")
        passed += 1

        # 7. Collate a 2-item batch
        print("\nTEST 5: collate_fn with 2-item batch")
        batch = [ds[0], ds[1]]
        collated = LongitudinalDVFDataset.collate_fn(batch)
        print("Collated shapes:")
        for k, v in collated.items():
            if isinstance(v, torch.Tensor):
                print(f"{k:20s} -> {tuple(v.shape)}")
            else:
                print(f"{k:20s} -> {v}")

        assert collated["dvf_sequence"].shape[0] == 2, "Batch dim != 2"
        assert collated["dvf_sequence"].shape[1] == config.v_max, (
            f"Visit dim = {collated['dvf_sequence'].shape[1]}, "
            f"expected {config.v_max}"
        )
        passed += 1

    except Exception:
        failed += 1
        traceback.print_exc()

    finally:
        # 9. Cleanup
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
            print(f"\n[cleanup] Removed {tmp_dir}")

    # Summary
    total = passed + failed
    if failed == 0:
        print(f"PASS — all {passed}/{total} tests passed.")
    else:
        print(f"FAIL — {failed}/{total} tests failed.")
    sys.exit(0 if failed == 0 else 1)
