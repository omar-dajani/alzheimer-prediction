"""
trainer — Two-stage LP-FT trainer with device-aware AMP, gradient
accumulation, MPS cache management, and early stopping on Antolini C_td.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# PyTorch-version-tolerant imports. torch.amp is present in 2.2.x;
# GradScaler moved namespaces at 2.3. We only ever use GradScaler on
# CUDA, so the import is guarded.
try:
    from torch.amp import autocast as _autocast
except ImportError:  # extremely old torch
    from torch.cuda.amp import autocast as _autocast  # type: ignore

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Transformer.config.model_config import ModelConfig, DEVICE
from Transformer.losses.ipcw_loss import (
    CensoringSurvivalEstimator,
    ipcw_survival_loss,
)
from Transformer.metrics.concordance import compute_all_metrics
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
    return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
        1 + math.cos(math.pi * progress)
    )


def _torch_version_tuple() -> tuple:
    """Return (major, minor) of the installed torch, for feature gating."""
    parts = torch.__version__.split(".")
    try:
        return (int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return (0, 0)


def _mps_autocast_available() -> bool:
    return _torch_version_tuple() >= (2, 5)


class LPFTTrainer:
    """Two-stage Linear Probe then Fine-Tune trainer."""

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
        self.device = torch.device(device) if device is not None else config.device
        self.model = self.model.to(self.device)

        # MPS memory sizing (advisory, never a hard cap)
        if self.device.type == "mps":
            try:
                torch.mps.set_per_process_memory_fraction(
                    config.mps_memory_fraction
                )
                logger.info(
                    "MPS memory fraction set to %.0f%% (advisory). "
                    "For full unified memory access, also export "
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 before Python.",
                    config.mps_memory_fraction * 100,
                )
            except AttributeError:
                logger.warning(
                    "torch.mps.set_per_process_memory_fraction missing — "
                    "needs PyTorch >= 2.1. Skipping memory hint."
                )

        # AMP configuration
        self._amp_enabled = False
        self._amp_device_type = "cpu"
        self._amp_dtype = torch.float32

        if self.device.type == "cuda":
            self._amp_enabled = True
            self._amp_device_type = "cuda"
            self._amp_dtype = torch.bfloat16
            logger.info("AMP: CUDA + bfloat16")
        elif self.device.type == "mps":
            if _mps_autocast_available():
                self._amp_enabled = True
                self._amp_device_type = "mps"
                self._amp_dtype = torch.float16  # bf16 still guarded on MPS
                logger.info("AMP: MPS + float16 (PyTorch %s)", torch.__version__)
            else:
                logger.info(
                    "AMP: disabled on MPS (PyTorch %s < 2.5). "
                    "Training in float32 — this is fast on Apple Silicon's "
                    "unified memory and avoids the GradScaler/MPS bug.",
                    torch.__version__,
                )
        # else: CPU / unknown — stays at float32.

        # GradScaler: CUDA only
        if self._amp_enabled and self._amp_device_type == "cuda":
            try:
                self.scaler = torch.amp.GradScaler(
                    device="cuda", enabled=True,
                )
            except TypeError:
                from torch.cuda.amp import GradScaler as _CudaScaler
                self.scaler = _CudaScaler(enabled=True)
        else:
            self.scaler = _NoopScaler()

        self.t_grid = torch.tensor(
            config.t_grid, dtype=torch.float32, device=self.device,
        )

        self.best_ctd = -1.0
        self.best_epoch = -1
        self.patience_counter = 0
        self.global_step = 0

        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "LPFTTrainer ready: device=%s, AMP=%s (%s), accum=%d, grad_clip=%.1f",
            self.device, self._amp_enabled, self._amp_dtype,
            config.gradient_accumulation_steps, config.grad_clip_max_norm,
        )

    # Stage 1
    def run_stage1(self, n_epochs: Optional[int] = None) -> dict:
        n_epochs = n_epochs or self.config.max_epochs_stage1
        logger.info("═══ Stage 1: Linear Probe (%d epochs) ═══", n_epochs)

        optimizer = AdamW(
            self.model.get_stage1_param_groups(),
            weight_decay=self.config.weight_decay,
        )
        total_steps = n_epochs * len(self.train_loader)
        warmup_steps = int(0.05 * total_steps)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda s: _linear_warmup_cosine_decay(
                s, warmup_steps, total_steps,
            ),
        )

        metrics: dict = {}
        for epoch in range(1, n_epochs + 1):
            self._epoch_start_mps_cache()
            train_loss = self._train_epoch(optimizer, scheduler)
            metrics = self._validate_epoch()
            logger.info(
                "Stage1 Epoch %d/%d — loss=%.4f, C_td=%.4f, Uno_C=%.4f, IBS=%.4f",
                epoch, n_epochs, train_loss,
                metrics["c_td"], metrics["uno_c"], metrics["ibs"],
            )
        self._save_checkpoint(n_epochs, metrics, stage="stage1")
        return metrics

    # Stage 2
    def run_stage2(self, n_epochs: Optional[int] = None) -> dict:
        n_epochs = n_epochs or self.config.max_epochs_stage2
        logger.info("═══ Stage 2: Fine-Tune (up to %d epochs) ═══", n_epochs)

        # MPS Conv3D workaround
        if self.device.type == "mps" and _torch_version_tuple() < (2, 5):
            self.model.extractor.cpu()
            logger.info(
                "MPS (torch %s < 2.5): BrainIAC extractor moved to CPU — "
                "Conv3D autograd not supported on MPS. Temporal model + "
                "survival head remain on MPS.",
                torch.__version__,
            )

        optimizer = AdamW(
            self.model.get_stage2_param_groups(),
            weight_decay=self.config.weight_decay,
        )
        total_steps = n_epochs * len(self.train_loader)
        warmup_steps = int(0.05 * total_steps)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda s: _linear_warmup_cosine_decay(
                s, warmup_steps, total_steps,
            ),
        )

        self.best_ctd = -1.0
        self.patience_counter = 0
        best_metrics: dict = {}
        epoch = 0

        for epoch in range(1, n_epochs + 1):
            self._epoch_start_mps_cache()
            train_loss = self._train_epoch(optimizer, scheduler)
            metrics = self._validate_epoch()
            c_td = metrics["c_td"]

            if c_td > self.best_ctd:
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
                    "Stage2 Epoch %d/%d — loss=%.4f, C_td=%.4f (patience %d/%d)",
                    epoch, n_epochs, train_loss, c_td,
                    self.patience_counter, self.config.early_stopping_patience,
                )

            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    "Early stop @ epoch %d. Best C_td=%.4f @ epoch %d",
                    epoch, self.best_ctd, self.best_epoch,
                )
                break

        self._save_checkpoint(epoch, metrics, stage="final")
        return best_metrics

    # Internals
    def _epoch_start_mps_cache(self) -> None:
        """Called once per epoch on MPS to release cached allocations.

        Calling this every batch (the old behavior) was a perf bug:
        empty_cache forces the next tensor allocation to round-trip
        through the system allocator.
        """
        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    def _autocast_ctx(self):
        if not self._amp_enabled:
            return _NullCtx()
        return _autocast(
            device_type=self._amp_device_type,
            dtype=self._amp_dtype,
            enabled=True,
        )

    def _train_epoch(self, optimizer, scheduler) -> float:
        self.model.train()
        accum = self.config.gradient_accumulation_steps
        total_loss, n_batches = 0.0, 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            dvf_seq = batch["dvf_sequence"].to(self.device, non_blocking=True)
            time_deltas = batch["time_deltas"].to(self.device, non_blocking=True)
            missing_mask = batch["missing_mask"].to(self.device, non_blocking=True)
            durations = batch["duration"].to(self.device, non_blocking=True)
            events = batch["event"].to(self.device, non_blocking=True)

            # Forward under autocast iff AMP is enabled.
            with self._autocast_ctx():
                outputs = self.model(dvf_seq, time_deltas, missing_mask)
                hazards = outputs["hazards"]

            # Loss ALWAYS in float32 for numerical stability.
            loss = ipcw_survival_loss(
                hazards.float(),
                durations.float(),
                events,
                self.censoring_estimator,
                self.t_grid,
            )
            loss = loss / accum

            self.scaler.scale(loss).backward()

            last_batch = (batch_idx + 1) == len(self.train_loader)
            if (batch_idx + 1) % accum == 0 or last_batch:
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

            total_loss += loss.item() * accum
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate_epoch(self) -> dict:
        self.model.eval()

        all_hazards, all_durations, all_events = [], [], []
        for batch in self.val_loader:
            dvf_seq = batch["dvf_sequence"].to(self.device, non_blocking=True)
            time_deltas = batch["time_deltas"].to(self.device, non_blocking=True)
            missing_mask = batch["missing_mask"].to(self.device, non_blocking=True)

            with self._autocast_ctx():
                outputs = self.model(dvf_seq, time_deltas, missing_mask)

            all_hazards.append(outputs["hazards"].float().cpu())
            all_durations.append(batch["duration"].numpy())
            all_events.append(batch["event"].numpy())

        hazards = torch.cat(all_hazards, dim=0)
        durations = np.concatenate(all_durations)
        events = np.concatenate(all_events)

        return compute_all_metrics(
            hazards, durations, events,
            self.train_durations, self.train_events,
            self.chi, self.model.survival_head, self.config,
        )

    def _save_checkpoint(self, epoch: int, metrics: dict, stage: str) -> None:
        path = self.config.checkpoint_dir / f"checkpoint_{stage}.pt"
        ckpt = {
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
            ckpt["norm_stats"] = self.norm_stats
        torch.save(ckpt, path)
        logger.info(
            "Checkpoint: %s (epoch=%d, C_td=%.4f)",
            path, epoch, metrics.get("c_td", 0.0),
        )

    def load_checkpoint(self, path: Path) -> dict:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.best_ctd = ckpt.get("best_ctd", -1.0)
        self.global_step = ckpt.get("global_step", 0)
        logger.info(
            "Loaded: %s (epoch=%d, C_td=%.4f)",
            path, ckpt["epoch"], ckpt["metrics"].get("c_td", 0.0),
        )
        return ckpt

# Helpers
class _NullCtx:
    """No-op context manager used when AMP is disabled."""
    def __enter__(self): return self
    def __exit__(self, *_): return False


class _NoopScaler:
    """Drop-in replacement for torch.amp.GradScaler for the disabled path.

    Used whenever AMP is off or the device is MPS (where GradScaler has
    an open upstream bug — see the module docstring). All methods are
    transparent forwards that keep the training loop identical.
    """
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):  # no-op
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):  # noqa: D401
        pass
