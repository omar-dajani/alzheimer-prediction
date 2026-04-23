import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Transformer.config.model_config import ModelConfig
from Transformer.data.dvf_dataset import (
    LongitudinalDVFDataset,
    NormalizationStats,
)
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
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="ADNI Advanced Survival Pipeline — Train"
    )
    parser.add_argument(
        "--sequence_model",
        choices=list(SEQUENCE_MODELS.keys()),
        default="mamba",
        help="Sequence model backbone (default: mamba)",
    )
    parser.add_argument(
        "--pretrained_brainiac",
        type=str,
        default=None,
        help="Path to pretrained BrainIAC weights",
    )
    parser.add_argument(
        "--dvf_dir",
        type=str,
        default="data/dvf",
        help="Directory containing per-subject DVF subdirectories",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="Baseline/outputs",
        help="Directory containing mci_y_duration.npy and mci_y_event.npy",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Model embedding dimension (default: 512)",
    )
    parser.add_argument(
        "--stage1_epochs",
        type=int,
        default=5,
        help="Number of Stage 1 (linear probe) epochs",
    )
    parser.add_argument(
        "--stage2_epochs",
        type=int,
        default=100,
        help="Maximum Stage 2 (fine-tune) epochs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/mps, default: auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Config
    config = ModelConfig(
        d_model=args.d_model,
        max_epochs_stage1=args.stage1_epochs,
        max_epochs_stage2=args.stage2_epochs,
        dvf_dir=Path(args.dvf_dir),
        survival_labels_dir=Path(args.labels_dir),
    )
    config.validate()

    # Seed
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    logger = logging.getLogger(__name__)
    logger.info("Config: d_model=%d, sequence=%s", config.d_model, args.sequence_model)

    # Data — discover subjects from DVF directory
    dvf_dir = Path(args.dvf_dir)
    subject_ids = sorted([
        d.name for d in dvf_dir.iterdir() if d.is_dir()
    ])
    if not subject_ids:
        logger.error("No subject directories found in %s", dvf_dir)
        sys.exit(1)

    logger.info("Found %d subjects in %s", len(subject_ids), dvf_dir)

    # Split into train/val (80/20)
    n_train = int(0.8 * len(subject_ids))
    train_ids = subject_ids[:n_train]
    val_ids = subject_ids[n_train:]

    # Compute normalization stats from training set
    train_paths = []
    for sid in train_ids:
        train_paths.extend(sorted((dvf_dir / sid).glob("*.npy")))
    norm_stats = NormalizationStats.compute(train_paths)

    # Datasets
    labels_dir = Path(args.labels_dir)
    train_dataset = LongitudinalDVFDataset(
        subject_ids=train_ids,
        dvf_dir=dvf_dir,
        config=config,
        norm_stats=norm_stats,
        survival_labels_dir=labels_dir,
    )
    val_dataset = LongitudinalDVFDataset(
        subject_ids=val_ids,
        dvf_dir=dvf_dir,
        config=config,
        norm_stats=norm_stats,
        survival_labels_dir=labels_dir,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size_physical,
        shuffle=True,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size_physical,
        shuffle=False,
        num_workers=0,
        collate_fn=val_dataset.collate_fn,
    )

    # Training labels for IPCW and metrics
    all_durations = np.concatenate([
        np.array([train_dataset[i]["duration"].item() for i in range(len(train_dataset))])
    ])
    all_events = np.concatenate([
        np.array([train_dataset[i]["event"].item() for i in range(len(train_dataset))])
    ])

    # Censoring estimator
    censoring_est = CensoringSurvivalEstimator()
    censoring_est.fit(all_durations, all_events)

    # CHI interpolator
    chi = ConstantHazardInterpolator.from_config(config)

    # Pipeline
    seq_cls = SEQUENCE_MODELS[args.sequence_model]
    model = ADNISurvivalPipeline(
        config=config,
        sequence_model_cls=seq_cls,
        pretrained_brainiac_path=args.pretrained_brainiac,
    )

    # Trainer
    trainer = LPFTTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        censoring_estimator=censoring_est,
        chi_interpolator=chi,
        train_durations=all_durations,
        train_events=all_events,
        norm_stats=norm_stats,
        device=args.device,
    )

    # Stage 1: Linear Probe
    stage1_metrics = trainer.run_stage1()

    # Stage 2: Fine-Tune
    best_metrics = trainer.run_stage2()

    # Comparison table
    table, csv_rows = format_comparison_table(best_metrics)
    print("\n" + table)

    logger.info("Training complete. Best C_td=%.4f at epoch %d",
                trainer.best_ctd, trainer.best_epoch)


if __name__ == "__main__":
    main()
