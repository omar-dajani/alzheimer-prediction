"""
trainer — Two-stage LP-FT trainer with AMP, gradient accumulation, and
early stopping on Antolini C_td.

Phase 9b: implements the training protocol from advanced_model_blueprint.md.

Stage 1 (linear probe, 5 epochs):
    Freeze backbone + sequence model. Train PMA + survival head at LR=1e-3.

Stage 2 (fine-tune, up to 100 epochs):
    Unfreeze sequence model with differential LRs.
    Early stopping on validation C_td with patience=15.
"""

import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Transformer.config.model_config import ModelConfig
from Transformer.losses.ipcw_loss import (
    CensoringSurvivalEstimator,
    ipcw_survival_loss,
)
from Transformer.metrics.concordance import compute_all_metrics
from Transformer.models.survival_head import TraCeRSurvivalHead
from Transformer.utils.chi_interpolation import ConstantHazardInterpolator

logger = logging.getLogger(__name__)


def _linear_warmup_cosine_decay(
    current_step: int,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 1e-2,
) -> float:
    """LR multiplier: linear warmup then cosine decay to min_lr_ratio."""
    if current_step < warmup_steps:
        return current_step / max(warmup_steps, 1)
    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (1 + math.cos(math.pi * progress))


class LPFTTrainer:
    """Two-stage Linear Probe then Fine-Tune trainer.

    Manages the full training lifecycle:
        - AMP (FP16) with GradScaler
        - Gradient accumulation (8 steps)
        - Gradient clipping (max_norm=1.0)
        - Early stopping on validation C_td
        - Checkpoint save/load with full state
    """

    def __init__(
        self,
        model: nn.Module,
        config: ModelConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        censoring_estimator: CensoringSurvivalEstimator,
        chi_interpolator: ConstantHazardInterpolator,
        train_durations: np.ndarray,
        train_events: np.ndarray,
        norm_stats=None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.censoring_estimator = censoring_estimator
        self.chi = chi_interpolator
        self.train_durations = train_durations
        self.train_events = train_events
        self.norm_stats = norm_stats

        # Device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # AMP scaler
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # t_grid tensor
        self.t_grid = torch.tensor(
            config.t_grid, dtype=torch.float32, device=self.device
        )

        # Tracking
        self.best_ctd = -1.0
        self.best_epoch = -1
        self.patience_counter = 0
        self.global_step = 0

        # Ensure checkpoint dir exists
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "LPFTTrainer initialized: device=%s, accum=%d, AMP=%s",
            self.device,
            config.gradient_accumulation_steps,
            self.device.type == "cuda",
        )

    # Stage 1: Linear Probe
    def run_stage1(self, n_epochs: Optional[int] = None) -> dict:
        """Stage 1: freeze backbone + sequence model, train PMA + head.

        Args:
            n_epochs: Override config.max_epochs_stage1.

        Returns:
            Dict of final validation metrics.
        """
        n_epochs = n_epochs or self.config.max_epochs_stage1
        logger.info("═══ Stage 1: Linear Probe (%d epochs) ═══", n_epochs)

        param_groups = self.model.get_stage1_param_groups()
        optimizer = AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        total_steps = n_epochs * len(self.train_loader)
        warmup_steps = int(0.05 * total_steps)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_warmup_cosine_decay(
                step, warmup_steps, total_steps
            ),
        )

        metrics = {}
        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch(optimizer, scheduler)
            metrics = self._validate_epoch()
            logger.info(
                "Stage1 Epoch %d/%d — loss=%.4f, C_td=%.4f, "
                "Uno_C=%.4f, IBS=%.4f",
                epoch, n_epochs, train_loss,
                metrics["c_td"], metrics["uno_c"], metrics["ibs"],
            )

        self._save_checkpoint(n_epochs, metrics, stage="stage1")
        return metrics

    # Stage 2: Fine-Tune

    def run_stage2(self, n_epochs: Optional[int] = None) -> dict:
        """Stage 2: unfreeze sequence model with differential LRs.

        Uses early stopping on validation C_td with configured patience.

        Args:
            n_epochs: Override config.max_epochs_stage2.

        Returns:
            Dict of best validation metrics.
        """
        n_epochs = n_epochs or self.config.max_epochs_stage2
        logger.info("═══ Stage 2: Fine-Tune (%d max epochs) ═══", n_epochs)

        param_groups = self.model.get_stage2_param_groups()
        optimizer = AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        total_steps = n_epochs * len(self.train_loader)
        warmup_steps = int(0.05 * total_steps)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_warmup_cosine_decay(
                step, warmup_steps, total_steps
            ),
        )

        self.best_ctd = -1.0
        self.patience_counter = 0
        best_metrics = {}

        for epoch in range(1, n_epochs + 1):
            train_loss = self._train_epoch(optimizer, scheduler)
            metrics = self._validate_epoch()

            c_td = metrics["c_td"]
            improved = c_td > self.best_ctd

            if improved:
                self.best_ctd = c_td
                self.best_epoch = epoch
                self.patience_counter = 0
                best_metrics = metrics.copy()
                self._save_checkpoint(epoch, metrics, stage="best")
                logger.info(
                    "Stage2 Epoch %d/%d — loss=%.4f, C_td=%.4f ★ "
                    "(new best), Uno_C=%.4f, IBS=%.4f",
                    epoch, n_epochs, train_loss,
                    c_td, metrics["uno_c"], metrics["ibs"],
                )
            else:
                self.patience_counter += 1
                logger.info(
                    "Stage2 Epoch %d/%d — loss=%.4f, C_td=%.4f "
                    "(patience %d/%d), Uno_C=%.4f, IBS=%.4f",
                    epoch, n_epochs, train_loss,
                    c_td, self.patience_counter,
                    self.config.early_stopping_patience,
                    metrics["uno_c"], metrics["ibs"],
                )

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d. Best C_td=%.4f at epoch %d",
                    epoch, self.best_ctd, self.best_epoch,
                )
                break

        self._save_checkpoint(epoch, metrics, stage="final")
        return best_metrics

    # Training epoch

    def _train_epoch(self, optimizer, scheduler) -> float:
        """Run one training epoch with gradient accumulation and AMP.

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        accum = self.config.gradient_accumulation_steps
        total_loss = 0.0
        n_batches = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            dvf_seq = batch["dvf_sequence"].to(self.device)
            time_deltas = batch["time_deltas"].to(self.device)
            missing_mask = batch["missing_mask"].to(self.device)
            durations = batch["duration"].to(self.device)
            events = batch["event"].to(self.device)

            # Forward with AMP
            with autocast(enabled=(self.device.type == "cuda")):
                outputs = self.model(dvf_seq, time_deltas, missing_mask)
                hazards = outputs["hazards"]

                loss = ipcw_survival_loss(
                    hazards, durations, events,
                    self.censoring_estimator, self.t_grid,
                )
                loss = loss / accum  # Scale for accumulation

            # Backward
            self.scaler.scale(loss).backward()

            # Step every accum batches or at last batch
            if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(self.train_loader):
                # Unscale for grad clipping
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_max_norm,
                )
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                self.global_step += 1

            total_loss += loss.item() * accum  # Undo scaling for logging
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # Validation epoch

    @torch.no_grad()
    def _validate_epoch(self) -> dict:
        """Run validation and compute all three metrics.

        Returns:
            Dict with c_td, uno_c, ibs, tau.
        """
        self.model.eval()

        all_hazards = []
        all_durations = []
        all_events = []

        for batch in self.val_loader:
            dvf_seq = batch["dvf_sequence"].to(self.device)
            time_deltas = batch["time_deltas"].to(self.device)
            missing_mask = batch["missing_mask"].to(self.device)

            with autocast(enabled=(self.device.type == "cuda")):
                outputs = self.model(dvf_seq, time_deltas, missing_mask)

            all_hazards.append(outputs["hazards"].float().cpu())
            all_durations.append(batch["duration"].numpy())
            all_events.append(batch["event"].numpy())

        hazards = torch.cat(all_hazards, dim=0)
        durations = np.concatenate(all_durations)
        events = np.concatenate(all_events)

        # Get survival head for hazards_to_survival
        survival_head = self.model.survival_head

        metrics = compute_all_metrics(
            hazards, durations, events,
            self.train_durations, self.train_events,
            self.chi, survival_head, self.config,
        )

        return metrics

    # Checkpoint management
    def _save_checkpoint(self, epoch: int, metrics: dict, stage: str = "best") -> None:
        """Save full checkpoint with model state, config, and t_grid.

        Checkpoint contents:
            - model_state_dict: full pipeline parameters
            - optimizer_state_dict: (not saved here, caller can add)
            - scaler_state_dict: AMP scaler state
            - epoch, metrics, stage
            - config: full ModelConfig as dict
            - t_grid: temporal grid points
            - norm_stats: normalization statistics (if available)
        """
        path = self.config.checkpoint_dir / f"checkpoint_{stage}.pt"
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "stage": stage,
            "config": asdict(self.config),
            "t_grid": self.config.t_grid,
            "best_ctd": self.best_ctd,
            "global_step": self.global_step,
        }
        if self.norm_stats is not None:
            checkpoint["norm_stats"] = self.norm_stats

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s (epoch=%d, C_td=%.4f)", path, epoch,
                     metrics.get("c_td", 0.0))

    def load_checkpoint(self, path: Path) -> dict:
        """Restore model and scaler state from checkpoint.

        Args:
            path: Path to checkpoint .pt file.

        Returns:
            Checkpoint dict with metadata (epoch, metrics, etc.).
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_ctd = checkpoint.get("best_ctd", -1.0)
        self.global_step = checkpoint.get("global_step", 0)
        logger.info(
            "Checkpoint loaded: %s (epoch=%d, C_td=%.4f)",
            path, checkpoint["epoch"], checkpoint["metrics"].get("c_td", 0.0),
        )
        return checkpoint
