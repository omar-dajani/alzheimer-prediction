from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
from pathlib import Path

# Set MPS env vars BEFORE torch is imported — otherwise they're ignored.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Transformer.config.model_config import ModelConfig, DEVICE
from Transformer.data.dvf_dataset import (
    LongitudinalDVFDataset,
    NormalizationStats,
)
from Transformer.data.label_io import save_survival_labels
from Transformer.losses.ipcw_loss import CensoringSurvivalEstimator
from Transformer.metrics.concordance import format_comparison_table
from Transformer.models.longformer_sequence import LongformerSequence
from Transformer.models.mamba_sequence import Mamba3DSequence
from Transformer.models.pipeline import ADNISurvivalPipeline
from Transformer.training.trainer import LPFTTrainer
from Transformer.utils.chi_interpolation import ConstantHazardInterpolator


SEQUENCE_MODELS = {
    "longformer": LongformerSequence,
    "mamba": Mamba3DSequence,
    "parallel": None,  # Gated fusion of both
}


def parse_args():
    p = argparse.ArgumentParser("ADNI Advanced Survival Pipeline")
    p.add_argument("--sequence_model", choices=list(SEQUENCE_MODELS.keys()),
                   default="parallel")
    p.add_argument("--pretrained_brainiac", default=None)
    p.add_argument("--dvf_dir", default="data/dvf_structured")
    p.add_argument("--master_csv",
                   default="data/master_data_improved_04052026_v3.csv",
                   help="Master CSV with PTID, VISCODE, DX, DX_bl, Month columns")
    p.add_argument("--split_out_dir", default="Transformer/split_labels",
                   help="Where the train-split labels (+subject_ids) land")
    p.add_argument("--val_out_dir", default="Transformer/val_labels")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--stage1_epochs", type=int, default=5)
    p.add_argument("--stage2_epochs", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default=None)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--clean", action="store_true",
                   help="Wipe checkpoints, labels, outputs before training")
    return p.parse_args()


def build_labels_from_csv(
    master_csv_path: Path,
    dvf_subject_ids: set[str],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Build (subject_ids, durations_months, events) from the master CSV.

    Only keeps subjects that (a) have DX_bl in {LMCI, EMCI}, (b) have
    event_time > 0, and (c) also appear as a DVF folder.

    Returns the triple in label-row order so it can be saved directly
    via save_survival_labels.
    """
    df = pd.read_csv(str(master_csv_path), low_memory=False)

    df_bl = df[df["VISCODE"] == "bl"].drop_duplicates("PTID").copy()
    mci_bl = df_bl[df_bl["DX_bl"].isin(["LMCI", "EMCI"])].copy()

    ids: list[str] = []
    durations: list[float] = []
    events: list[int] = []

    for _, bl_row in mci_bl.iterrows():
        ptid = bl_row["PTID"]
        if ptid not in dvf_subject_ids:
            continue  # no imaging — can't use them
        subj = df[df["PTID"] == ptid].sort_values("Month")
        follow_dementia = subj[
            (subj["VISCODE"] != "bl") & (subj["DX"] == "Dementia")
        ]
        if len(follow_dementia) > 0:
            t = float(follow_dementia["Month"].min())
            e = 1
        else:
            t = float(subj["Month"].max())
            e = 0
        if t <= 0:
            continue
        ids.append(str(ptid))
        durations.append(t)
        events.append(e)

    return (
        ids,
        np.asarray(durations, dtype=np.float32),
        np.asarray(events, dtype=np.int64),
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Optional cleanup for reproducible fresh runs
    if args.clean:
        import shutil
        for d in [
            Path(args.split_out_dir),
            Path(args.val_out_dir),
            Path("Transformer/checkpoints"),
            Path("Transformer/outputs"),
            Path("Transformer/figures"),
        ]:
            if d.exists() and any(d.iterdir()):
                shutil.rmtree(d)
                d.mkdir(parents=True)
                logger.info("Cleaned %s", d)
        logger.info("All output directories wiped. Starting fresh.")

    config = ModelConfig(
        d_model=args.d_model,
        max_epochs_stage1=args.stage1_epochs,
        max_epochs_stage2=args.stage2_epochs,
        dvf_dir=Path(args.dvf_dir),
        survival_labels_dir=Path(args.split_out_dir),  # train
    )
    config.validate()
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    logger.info("Device=%s | torch=%s | system=%s",
                config.device, torch.__version__, platform.platform())

    # Discover DVF subjects
    dvf_dir = Path(args.dvf_dir)
    dvf_subjects = {d.name for d in dvf_dir.iterdir() if d.is_dir()}
    if not dvf_subjects:
        logger.error("No subject folders under %s", dvf_dir)
        sys.exit(1)
    logger.info("DVF folders found: %d", len(dvf_subjects))

    # Build ID-aware labels
    subject_ids, durations, events = build_labels_from_csv(
        Path(args.master_csv), dvf_subjects,
    )
    logger.info(
        "Labeled MCI subjects with DVF: %d (events=%d, censored=%d)",
        len(subject_ids),
        int((events == 1).sum()),
        int((events == 0).sum()),
    )

    # Stratified train/val split
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(subject_ids))
    train_idx, val_idx = train_test_split(
        idx, test_size=args.test_size,
        random_state=config.random_seed,
        stratify=events,
    )
    train_ids = [subject_ids[i] for i in train_idx]
    val_ids = [subject_ids[i] for i in val_idx]
    train_dur, val_dur = durations[train_idx], durations[val_idx]
    train_evt, val_evt = events[train_idx], events[val_idx]

    # Persist both splits WITH their ID indexes.
    save_survival_labels(Path(args.split_out_dir), train_ids, train_dur, train_evt)
    save_survival_labels(Path(args.val_out_dir),   val_ids,   val_dur,   val_evt)

    # Normalization stats from training subjects only
    train_dvf_paths = []
    for sid in train_ids:
        train_dvf_paths.extend(sorted((dvf_dir / sid).glob("*.npy")))
    if not train_dvf_paths:
        logger.error("Zero .npy files found for training subjects.")
        sys.exit(1)
    norm_stats = NormalizationStats.compute(train_dvf_paths)

    # Datasets
    train_ds = LongitudinalDVFDataset(
        subject_ids=train_ids, dvf_dir=dvf_dir, config=config,
        norm_stats=norm_stats,
        survival_labels_dir=Path(args.split_out_dir),
    )
    val_ds = LongitudinalDVFDataset(
        subject_ids=val_ids, dvf_dir=dvf_dir, config=config,
        norm_stats=norm_stats,
        survival_labels_dir=Path(args.val_out_dir),
    )

    # DataLoaders — Apple-Silicon-friendly defaults
    is_mac = platform.system() == "Darwin"
    is_cuda = (config.device.type == "cuda")
    nw = args.num_workers
    mp_ctx = "spawn" if (is_mac and nw > 0) else None

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.batch_size_physical,
        shuffle=True,
        num_workers=nw,
        persistent_workers=(nw > 0),
        multiprocessing_context=mp_ctx,
        pin_memory=is_cuda,
        collate_fn=train_ds.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.batch_size_physical,
        shuffle=False,
        num_workers=nw,
        persistent_workers=(nw > 0),
        multiprocessing_context=mp_ctx,
        pin_memory=is_cuda,
        collate_fn=val_ds.collate_fn,
    )

    # IPCW and CHI
    # Use dataset arrays directly — no need to iterate the whole dataset.
    censoring_est = CensoringSurvivalEstimator()
    censoring_est.fit(train_ds.durations, train_ds.events)
    chi = ConstantHazardInterpolator.from_config(config)

    # Model + trainer
    seq_cls = SEQUENCE_MODELS[args.sequence_model]
    model = ADNISurvivalPipeline(
        config=config,
        sequence_model_cls=seq_cls,
        pretrained_brainiac_path=args.pretrained_brainiac,
    )

    trainer = LPFTTrainer(
        model=model, config=config,
        train_loader=train_loader, val_loader=val_loader,
        censoring_estimator=censoring_est, chi_interpolator=chi,
        train_durations=train_ds.durations,
        train_events=train_ds.events,
        norm_stats=norm_stats,
        device=args.device,
    )

    trainer.run_stage1()
    best = trainer.run_stage2()
    table, _ = format_comparison_table(best)
    print("\n" + table)
    logger.info(
        "Done. Best C_td=%.4f @ epoch %d",
        trainer.best_ctd, trainer.best_epoch,
    )

if __name__ == "__main__":
    main()
