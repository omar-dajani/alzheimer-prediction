"""
Transformer/training/trainer.py
================================
LPFTTrainer — Linear Probe then Fine-Tune (LP-FT) two-stage trainer for
the ADNISurvivalPipeline.

Training schedule
-----------------
Stage 1 — Linear Probe (5 epochs, from ModelConfig.max_epochs_stage1)
  Freeze:  BrainIAC (backbone + projection) + sequence model
  Train:   PMA + survival head only
  LR:      config.lr_survival_head (1e-3), flat — too few epochs for a schedule

Stage 2 — Fine-Tune (up to 100 epochs, ModelConfig.max_epochs_stage2)
  Unfreeze: sequence model + BrainIAC projection
  LRs (differential per group, read from ModelConfig):
    brainiac_backbone:   0.0  (stays frozen)
    brainiac_projection: config.lr_projection_head  (1e-4)
    sequence_model:      config.lr_sequence_model   (5e-5)
    pma:                 config.lr_survival_head    (1e-3)
    survival_head:       config.lr_survival_head    (1e-3)
  Schedule: linear warmup (5% of total steps) → cosine decay to 1e-6
  Early stopping: patience=config.early_stopping_patience on val Antolini C_td

Training mechanics
------------------
- Physical batch size: config.batch_size_physical (2, GPU RAM-limited)
- Gradient accumulation: config.gradient_accumulation_steps (8)
  → effective batch ≈ 16
- AMP (FP16) via torch.cuda.amp.GradScaler
- Gradient clipping: clip_grad_norm_(params, config.grad_clip_max_norm)
  applied after scaler.unscale_() so we clip real (not scaled) gradients

Loss function
-------------
ipcw_survival_loss(hazards, durations, events, censoring_estimator, t_grid)
from Transformer/losses/ipcw_loss.py. The CensoringSurvivalEstimator is fitted
once on training-set labels before Stage 1 begins and is reused for all
subsequent training steps. It must NEVER be refit on val/test data.

Checkpoint format
-----------------
Mirrors Baseline/modeling.py conventions (print-based logging, descriptive
path echoing) but uses torch.save instead of pickle to preserve PyTorch
optimizer and scaler state alongside the model weights:

  {
    'epoch':                int,
    'stage':                int (1 or 2),
    'model_state':          OrderedDict,
    'optimizer_state':      dict,
    'scaler_state':         dict,          # GradScaler — required for exact AMP repro
    'scheduler_state':      dict | None,
    'best_c_td':            float,
    'global_step':          int,
    'config':               ModelConfig,   # full config — no separate file at inference
    't_grid':               np.ndarray,    # time grid for CHI interpolation
    'norm_stats':           any,           # NormalizationStats from training dataset
    'censoring_estimator':  CensoringSurvivalEstimator,  # pickle-serializable
    'sequence_model':       str,           # 'longformer' | 'mamba'
  }

Logging
-------
Matches Baseline/modeling.py's print-based style.
Per-epoch line format:
  [Stage 2 | Ep  12/100] train_loss=0.4231  val_C_td=0.7103  best=0.7210  patience=3/15
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from Transformer.models.pipeline import ADNISurvivalPipeline
from Transformer.losses.ipcw_loss import (
    ipcw_survival_loss,
    CensoringSurvivalEstimator,
)
from Transformer.metrics.concordance import compute_all_metrics
from Transformer.utils.chi_interpolation import ConstantHazardInterpolator
from Transformer.config.model_config import ModelConfig


# ── Learning rate schedule ────────────────────────────────────────────────────

def _build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    n_warmup_steps: int,
    n_total_steps: int,
    min_lr_ratio: float = 1e-6,
) -> LambdaLR:
    """
    Linear warmup (0 → 1) for n_warmup_steps steps, then cosine decay
    from 1 → min_lr_ratio over the remaining steps.

    min_lr_ratio is expressed as a fraction of base LR so it is
    independent of the actual LR values — 1e-6 means "decay to 1e-6
    times whatever base LR the optimizer group has."
    """
    def lr_lambda(step: int) -> float:
        if step < n_warmup_steps:
            return float(step) / float(max(1, n_warmup_steps))
        progress = float(step - n_warmup_steps) / float(
            max(1, n_total_steps - n_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ── Trainer ───────────────────────────────────────────────────────────────────

class LPFTTrainer:
    """
    Two-stage LP-FT trainer for ADNISurvivalPipeline.

    Args:
        pipeline (ADNISurvivalPipeline):
            The assembled model to train.
        train_loader (DataLoader):
            Yields batches of dicts with keys:
              dvf_sequence [B, V, 3, 128, 128, 128]
              time_deltas  [B, V]
              missing_mask [B, V]
              duration     [B]   — observed time in months
              event        [B]   — 0=censored, 1=dementia, 2=mortality
        val_loader (DataLoader):
            Same schema as train_loader. Used for early stopping and
            metric computation at each epoch.
        config (ModelConfig):
            Hyperparameter store. LRs, patience, grad_accum steps,
            early stopping patience are all read from here.
        checkpoint_dir (str | Path):
            Directory for saving .pt checkpoint files.
            Created automatically if it does not exist.
        t_grid (np.ndarray):
            Time grid for CHI interpolation and loss bucketing.
            Saved into every checkpoint for inference portability.
        norm_stats:
            NormalizationStats from the training dataset.
            Saved into every checkpoint for inference portability.
        train_durations (np.ndarray):
            Training-set observed times in months — used to fit the
            CensoringSurvivalEstimator (IPCW weights) and to compute
            Uno's C tau (75th percentile of training event times).
        train_events (np.ndarray):
            Training-set event labels {0, 1, 2} — used with
            train_durations to fit the CensoringSurvivalEstimator.
        sequence_model_name (str):
            'longformer' or 'mamba' — stored in checkpoints for human
            inspection. Does not affect training logic.
        device (str | torch.device | None):
            Target device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        pipeline: ADNISurvivalPipeline,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig,
        checkpoint_dir: str | Path,
        t_grid: np.ndarray,
        norm_stats,
        train_durations: np.ndarray,
        train_events: np.ndarray,
        sequence_model_name: str = "longformer",
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.pipeline = pipeline
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.t_grid = t_grid
        self.norm_stats = norm_stats
        self.train_durations = train_durations
        self.train_events = train_events
        self.sequence_model_name = sequence_model_name

        # Derived from config — single source of truth
        self.grad_accum_steps = config.gradient_accumulation_steps

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.pipeline.to(self.device)

        # ── AMP scaler — shared across both stages; state saved in checkpoints ─
        self.scaler = GradScaler()

        # ── Loss: t_grid as a float32 tensor for torch.bucketize ─────────────
        self.t_grid_tensor = torch.tensor(t_grid, dtype=torch.float32, device=self.device)

        # ── CHI interpolator — needed by compute_all_metrics ──────────────────
        self.chi = ConstantHazardInterpolator.from_config(config)

        # ── CensoringSurvivalEstimator — fitted once on training labels ───────
        # Must NEVER be refit on val/test data (information leakage).
        print("  Fitting CensoringSurvivalEstimator on training labels...")
        self.censoring_estimator = CensoringSurvivalEstimator()
        self.censoring_estimator.fit(train_durations, train_events)

        # ── Optimizer / scheduler — populated by run_stage1 / run_stage2 ──────
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[LambdaLR] = None

        # ── Tracking state ─────────────────────────────────────────────────────
        self.best_c_td: float = 0.0
        self.patience_counter: int = 0
        self.global_step: int = 0

        print(f"  LPFTTrainer initialised on {self.device}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(
            f"  Grad accum: {self.grad_accum_steps} steps "
            f"(physical batch={train_loader.batch_size}, "
            f"effective≈{self.grad_accum_steps * train_loader.batch_size})"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run_stage1(self, n_epochs: Optional[int] = None) -> float:
        """
        Stage 1 — Linear Probe.

        Freezes BrainIAC + sequence model, trains only PMA + survival head
        at a flat LR of config.lr_survival_head. No LR schedule (too few
        epochs for warmup/decay to matter).

        Args:
            n_epochs: Epochs to run. Defaults to config.max_epochs_stage1 (5).

        Returns:
            val Antolini C_td at end of Stage 1.
        """
        if n_epochs is None:
            n_epochs = self.config.max_epochs_stage1

        print("\n" + "=" * 65)
        print(
            f"  Stage 1 — Linear Probe  "
            f"({n_epochs} epochs, LR={self.config.lr_survival_head})"
        )
        print("=" * 65)

        self.pipeline.freeze_for_linear_probe()

        trainable = [p for p in self.pipeline.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=self.config.lr_survival_head,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = None  # flat LR for Stage 1

        val_c_td = 0.0
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(stage=1, epoch=epoch)
            val_c_td, _ = self._validate_epoch()
            elapsed = time.time() - t0

            print(
                f"  [Stage 1 | Ep {epoch:2d}/{n_epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_C_td={val_c_td:.4f}  "
                f"({elapsed:.1f}s)"
            )

        self._save_checkpoint(tag="stage1_final", epoch=n_epochs, stage=1, val_c_td=val_c_td)
        return val_c_td

    def run_stage2(self, n_epochs: Optional[int] = None) -> float:
        """
        Stage 2 — Fine-Tune with differential LRs and early stopping.

        Unfreezes sequence model + BrainIAC projection. Backbone stays frozen.
        LR schedule: linear warmup (5% of total steps) → cosine decay to 1e-6.
        Early stopping on val Antolini C_td with patience=config.early_stopping_patience.

        Args:
            n_epochs: Max epochs. Defaults to config.max_epochs_stage2 (100).

        Returns:
            Best val Antolini C_td achieved during Stage 2.
        """
        if n_epochs is None:
            n_epochs = self.config.max_epochs_stage2

        patience = self.config.early_stopping_patience

        print("\n" + "=" * 65)
        print(f"  Stage 2 — Fine-Tune  (max {n_epochs} epochs, patience={patience})")
        print("=" * 65)

        self.pipeline.unfreeze_for_finetuning()

        # ── Differential LR param groups ──────────────────────────────────────
        # Backbone is excluded (LR=0 → keep frozen, don't add to optimizer).
        # LRs come from ModelConfig — never hardcoded here.
        stage2_lrs: dict[str, float] = {
            "brainiac_backbone":   0.0,                          # stays frozen
            "brainiac_projection": self.config.lr_projection_head,  # 1e-4
            "sequence_model":      self.config.lr_sequence_model,   # 5e-5
            "pma":                 self.config.lr_survival_head,    # 1e-3
            "survival_head":       self.config.lr_survival_head,    # 1e-3
        }

        optim_groups = []
        for g in self.pipeline.param_groups():
            lr = stage2_lrs.get(g["name"], self.config.lr_projection_head)
            if lr == 0.0:
                continue  # backbone stays frozen; not added to optimizer
            optim_groups.append({
                "params": g["params"],
                "lr": lr,
                "name": g["name"],  # stored for debugging; AdamW ignores unknown keys
            })

        self.optimizer = AdamW(optim_groups, weight_decay=self.config.weight_decay)

        # ── LR schedule: warmup 5% → cosine decay ────────────────────────────
        steps_per_epoch = math.ceil(len(self.train_loader) / self.grad_accum_steps)
        total_steps = n_epochs * steps_per_epoch
        warmup_steps = max(1, int(0.05 * total_steps))
        self.scheduler = _build_warmup_cosine_scheduler(
            self.optimizer, warmup_steps, total_steps, min_lr_ratio=1e-6
        )
        print(f"  LR schedule: {warmup_steps} warmup / {total_steps} total optimizer steps")

        # ── Reset early-stopping state for Stage 2 ────────────────────────────
        self.best_c_td = 0.0
        self.patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(stage=2, epoch=epoch)
            val_c_td, _ = self._validate_epoch()
            elapsed = time.time() - t0

            improved = val_c_td > self.best_c_td
            if improved:
                self.best_c_td = val_c_td
                self.patience_counter = 0
                self._save_checkpoint(tag="best", epoch=epoch, stage=2, val_c_td=val_c_td)
            else:
                self.patience_counter += 1

            if epoch % 10 == 0:
                self._save_checkpoint(
                    tag=f"stage2_ep{epoch:03d}", epoch=epoch, stage=2, val_c_td=val_c_td
                )

            print(
                f"  [Stage 2 | Ep {epoch:3d}/{n_epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_C_td={val_c_td:.4f}  "
                f"best={self.best_c_td:.4f}  "
                f"patience={self.patience_counter}/{patience}  "
                f"({'✓' if improved else '–'})  "
                f"({elapsed:.1f}s)"
            )

            if self.patience_counter >= patience:
                print(
                    f"\n  Early stopping triggered at epoch {epoch} "
                    f"(patience={patience} exhausted). "
                    f"Best val C_td: {self.best_c_td:.4f}"
                )
                break

        print(f"\n  Stage 2 complete. Best val C_td: {self.best_c_td:.4f}")
        return self.best_c_td

    # ── Core training loop ────────────────────────────────────────────────────

    def _train_epoch(self, stage: int, epoch: int) -> float:
        """
        One epoch of training with gradient accumulation, AMP, and grad clipping.

        Gradient accumulation:
          - Loss is divided by grad_accum_steps so the effective gradient
            magnitude is independent of the accumulation count.
          - optimizer.step() fires every grad_accum_steps micro-batches,
            plus once at the end of the epoch for any trailing remainder.
          - GradScaler: if a step is skipped due to inf/nan, the LR scheduler
            is NOT stepped (detected by comparing scale before/after).

        Returns:
            Mean training loss averaged over all optimizer update steps.
        """
        self.pipeline.train()
        total_loss = 0.0
        n_updates = 0

        self.optimizer.zero_grad(set_to_none=True)

        for micro_step, batch in enumerate(self.train_loader):
            dvf          = batch["dvf_sequence"].to(self.device, non_blocking=True)
            time_deltas  = batch["time_deltas"].to(self.device, non_blocking=True)
            missing_mask = batch["missing_mask"].to(self.device, non_blocking=True)
            duration     = batch["duration"].to(self.device, non_blocking=True)
            event        = batch["event"].to(self.device, non_blocking=True)

            with autocast():
                out = self.pipeline(dvf, time_deltas, missing_mask)
                # ipcw_survival_loss signature:
                #   (hazards, durations, events, censoring_estimator, t_grid)
                # t_grid must be a float32 tensor (for torch.bucketize).
                loss = ipcw_survival_loss(
                    hazards=out["hazards"],
                    durations=duration,
                    events=event,
                    censoring_estimator=self.censoring_estimator,
                    t_grid=self.t_grid_tensor,
                )
                loss_scaled = loss / self.grad_accum_steps

            self.scaler.scale(loss_scaled).backward()

            is_update_step = (
                (micro_step + 1) % self.grad_accum_steps == 0
                or (micro_step + 1) == len(self.train_loader)
            )

            if is_update_step:
                # Unscale before clipping — clip_grad_norm_ operates on
                # real gradient magnitudes, not AMP-scaled ones.
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.pipeline.parameters(),
                    max_norm=self.config.grad_clip_max_norm,
                )

                # Detect if scaler skips the step due to inf/nan gradients.
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()

                # Only advance scheduler when an actual optimizer step occurred.
                if self.scheduler is not None and scale_after >= scale_before:
                    self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                total_loss += loss.item()
                n_updates += 1

        return total_loss / max(1, n_updates)

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_epoch(self) -> tuple[float, dict]:
        """
        Run the full validation set and compute all three metrics.

        compute_all_metrics signature (from concordance.py):
          (hazards, durations, events, train_durations, train_events,
           chi_interpolator, survival_head, config)
        → returns dict with keys: c_td, uno_c, ibs, tau

        Returns:
            (val_c_td, full_metrics_dict)
        """
        self.pipeline.eval()
        all_hazards, all_durations, all_events = [], [], []

        with torch.no_grad():
            for batch in self.val_loader:
                dvf          = batch["dvf_sequence"].to(self.device, non_blocking=True)
                time_deltas  = batch["time_deltas"].to(self.device, non_blocking=True)
                missing_mask = batch["missing_mask"].to(self.device, non_blocking=True)
                duration     = batch["duration"].to(self.device, non_blocking=True)
                event        = batch["event"].to(self.device, non_blocking=True)

                with autocast():
                    out = self.pipeline(dvf, time_deltas, missing_mask)

                all_hazards.append(out["hazards"].cpu().float())
                all_durations.append(duration.cpu())
                all_events.append(event.cpu())

        hazards_all  = torch.cat(all_hazards, dim=0)          # [N, n_risks, n_grid]
        durations_np = torch.cat(all_durations).numpy().astype(np.float64)
        events_np    = torch.cat(all_events).numpy().astype(np.int64)

        # compute_all_metrics needs train labels to compute Uno's tau
        # and to build the IPCW weights for IBS.
        metrics = compute_all_metrics(
            hazards=hazards_all,
            durations=durations_np,
            events=events_np,
            train_durations=self.train_durations,
            train_events=self.train_events,
            chi_interpolator=self.chi,
            survival_head=self.pipeline.survival_head,
            config=self.config,
        )
        val_c_td = float(metrics.get("c_td", 0.0))
        return val_c_td, metrics

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        tag: str,
        epoch: int,
        stage: int,
        val_c_td: float,
    ) -> None:
        """
        Save full training state to a .pt file.

        Checkpoint layout (all metadata inline — no separate config file
        required at inference time, mirroring Baseline/modeling.py's
        "everything self-contained" convention):

            model_state          → pipeline weights
            optimizer_state      → AdamW state
            scaler_state         → GradScaler state (AMP repro)
            scheduler_state      → LambdaLR state (or None for Stage 1)
            config               → full ModelConfig
            t_grid               → time grid for CHI interpolation
            norm_stats           → per-channel normalisation stats
            censoring_estimator  → fitted CensoringSurvivalEstimator
            epoch / stage / best_c_td / global_step / sequence_model
        """
        path = self.checkpoint_dir / f"advanced_pipeline_{tag}.pt"

        checkpoint = {
            "epoch":               epoch,
            "stage":               stage,
            "model_state":         self.pipeline.state_dict(),
            "optimizer_state":     self.optimizer.state_dict() if self.optimizer else None,
            "scaler_state":        self.scaler.state_dict(),
            "scheduler_state":     self.scheduler.state_dict() if self.scheduler else None,
            "best_c_td":           self.best_c_td,
            "val_c_td_at_save":    val_c_td,
            "global_step":         self.global_step,
            # ── Inference portability ──────────────────────────────────────────
            "config":              self.config,
            "t_grid":              self.t_grid,
            "norm_stats":          self.norm_stats,
            "censoring_estimator": self.censoring_estimator,  # pickle-serializable
            "sequence_model":      self.sequence_model_name,
        }

        torch.save(checkpoint, path)
        print(f"  Checkpointed: advanced_pipeline_{tag} -> {path}  (C_td={val_c_td:.4f})")

    def _load_checkpoint(self, tag: str) -> Optional[dict]:
        """
        Load a checkpoint from disk by tag name.

        Mirrors Baseline/modeling.py's load_checkpoint() — returns None
        silently if the file does not exist so the caller can fall back to
        a fresh start without crashing.

        Restores: model weights, optimizer, scaler, scheduler, tracking state.

        Args:
            tag: Checkpoint stem (e.g. 'best', 'stage1_final').

        Returns:
            Full checkpoint dict, or None if not found.
        """
        path = self.checkpoint_dir / f"advanced_pipeline_{tag}.pt"
        if not path.exists():
            print(f"  Checkpoint not found: {path}  (starting fresh)")
            return None

        checkpoint = torch.load(path, map_location=self.device)

        self.pipeline.load_state_dict(checkpoint["model_state"])

        if self.optimizer and checkpoint.get("optimizer_state"):
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])

        if checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])

        if self.scheduler and checkpoint.get("scheduler_state"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        self.best_c_td    = checkpoint.get("best_c_td", 0.0)
        self.global_step  = checkpoint.get("global_step", 0)

        print(
            f"  Loaded checkpoint: advanced_pipeline_{tag}  "
            f"(epoch={checkpoint['epoch']}, stage={checkpoint['stage']}, "
            f"best_C_td={self.best_c_td:.4f})"
        )
        return checkpoint

    def resume_from(self, tag: str = "best") -> Optional[dict]:
        """
        Public-facing resume helper. Returns the checkpoint dict so the
        caller can inspect which stage to resume from.
        """
        return self._load_checkpoint(tag)


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile
    import numpy as np
    from torch.utils.data import Dataset, DataLoader as TDL
    from Transformer.config.model_config import ModelConfig
    from Transformer.models.pipeline import ADNISurvivalPipeline
    from Transformer.models.longformer_sequence import LongformerSequence

    print("=" * 65)
    print("LPFTTrainer — smoke test (synthetic data, 2 epochs each stage)")
    print("=" * 65)

    cfg = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B_phys, V = cfg.batch_size_physical, 2  # small V for speed

    # ── Synthetic Dataset ─────────────────────────────────────────────────────
    class SyntheticDS(Dataset):
        def __init__(self, n):
            self.n = n
        def __len__(self):
            return self.n
        def __getitem__(self, idx):
            return {
                "dvf_sequence":  torch.randn(V, 3, 128, 128, 128),
                "time_deltas":   torch.rand(V) * 12,
                "missing_mask":  torch.ones(V),
                "duration":      torch.tensor(float(np.random.randint(6, 60))),
                "event":         torch.tensor(float(np.random.choice([0, 1, 2]))),
            }

    n_train, n_val = 8, 4
    train_loader = TDL(SyntheticDS(n_train), batch_size=B_phys, shuffle=False)
    val_loader   = TDL(SyntheticDS(n_val),   batch_size=B_phys, shuffle=False)

    # ── Synthetic survival labels (for censoring estimator) ───────────────────
    np.random.seed(42)
    train_durations = np.random.uniform(6, 60, n_train)
    train_events    = np.random.choice([0, 1, 2], n_train)

    # ── Synthetic t_grid and norm_stats ───────────────────────────────────────
    t_grid     = np.array(cfg.t_grid, dtype=np.float64)
    norm_stats = {"mean": np.zeros(3), "std": np.ones(3)}  # placeholder

    # ── Construct pipeline and trainer ────────────────────────────────────────
    print("\n[1] Constructing pipeline + trainer...")
    pipeline = ADNISurvivalPipeline(
        config=cfg,
        sequence_model_cls=LongformerSequence,
        pretrained_brainiac_path=None,
    ).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = LPFTTrainer(
            pipeline=pipeline,
            train_loader=train_loader,
            val_loader=val_loader,
            config=cfg,
            checkpoint_dir=tmpdir,
            t_grid=t_grid,
            norm_stats=norm_stats,
            train_durations=train_durations,
            train_events=train_events,
            sequence_model_name="longformer",
            device=device,
        )
        print("  PASS — trainer constructed, censoring estimator fitted")

        # ── Stage 1 smoke test ────────────────────────────────────────────────
        print("\n[2] run_stage1 (2 epochs)...")
        c_td_s1 = trainer.run_stage1(n_epochs=2)
        assert isinstance(c_td_s1, float)
        print(f"  PASS — Stage 1 complete, val C_td={c_td_s1:.4f}")

        ckpt_path = Path(tmpdir) / "advanced_pipeline_stage1_final.pt"
        assert ckpt_path.exists(), "Stage 1 checkpoint not found"
        print("  PASS — checkpoint file exists on disk")

        # ── Checkpoint content validation ─────────────────────────────────────
        print("\n[3] Checkpoint content validation...")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        required_keys = {
            "epoch", "stage", "model_state", "optimizer_state",
            "scaler_state", "config", "t_grid", "norm_stats",
            "censoring_estimator", "sequence_model",
        }
        missing = required_keys - set(ckpt.keys())
        assert not missing, f"Checkpoint missing keys: {missing}"
        assert ckpt["stage"] == 1
        assert ckpt["sequence_model"] == "longformer"
        assert ckpt["censoring_estimator"].is_fitted
        print(f"  PASS — all required keys present, censoring_estimator is fitted")

        # ── Stage 2 smoke test ────────────────────────────────────────────────
        print("\n[4] run_stage2 (2 epochs, patience=1 override)...")
        trainer.config.early_stopping_patience = 1  # accelerate for smoke test
        c_td_s2 = trainer.run_stage2(n_epochs=2)
        assert isinstance(c_td_s2, float)
        print(f"  PASS — Stage 2 complete, best val C_td={c_td_s2:.4f}")

        # ── _load_checkpoint ──────────────────────────────────────────────────
        print("\n[5] _load_checkpoint (existing)...")
        loaded = trainer._load_checkpoint("stage1_final")
        assert loaded is not None
        assert loaded["stage"] == 1
        print("  PASS — loaded and state restored")

        print("\n[6] _load_checkpoint (nonexistent)...")
        result = trainer._load_checkpoint("does_not_exist")
        assert result is None
        print("  PASS — returns None gracefully")

    print("\n" + "=" * 65)
    print("All 6 smoke tests PASSED")
    print("=" * 65)