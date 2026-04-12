"""
Pipeline position:
    Phase 0 of the ADNI Advanced Survival Pipeline. This module is the
    entry point for all data flow into the Transformer stack. It runs
    before any model component is instantiated.

Inputs:
    - DVF .npy arrays: [3, 128, 128, 128] float32, one per visit per subject
    - Survival labels: mci_y_duration.npy, mci_y_event.npy from Baseline/outputs/
    - (Optional) ComBat-harmonized tabular features from Baseline pipeline

Outputs:
    - LongitudinalDVFDataset: yields per-subject dicts with keys:
        dvf_sequence, visit_times, time_deltas, missing_mask, tabular,
        duration, event, subject_id
    - NormalizationStats: serializable stats for inference-time normalization

Dependencies:
    - Baseline/outputs/ must exist and contain mci_y_duration.npy, mci_y_event.npy
    - No other Transformer modules are required at this phase
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ModelConfig:
    """Centralized hyperparameters for the ADNI Advanced Survival Pipeline.

    This dataclass consolidates every tunable parameter from the
    advanced_model_blueprint.md training table into a single,
    type-checked, pickle-serializable object. A dataclass is preferred
    over argparse or a plain dict because it provides static type hints,
    IDE auto-complete, and prevents typo-errors in key names. Compared
    to YAML-based configs, it keeps defaults co-located with the code
    that consumes them, eliminating out-of-sync files.

    Tensor shapes consumed / produced:
        This module does not consume or produce tensors directly. It
        parameterizes every downstream module that does.
    """


    d_model: int = 512              # Token embedding dimension throughout the pipeline
    n_tokens_per_visit: int = 512   # Spatial tokens per visit (8×8×8 BrainIAC grid)
    v_max: int = 10                 # Maximum visits per subject (pad shorter sequences)
    n_grid: int = 5                 # TraCeR temporal grid points
    n_risks: int = 2                # Competing risks: 0=dementia, 1=mortality
    pma_seeds: int = 8              # PMA pooling seed vectors
    longformer_window: int = 512    # Sliding-window attention width (= n_tokens_per_visit)
    modality_dropout_p: float = 0.25  # Probability of dropping an entire visit during training
    mamba_d_state: int = 16         # Mamba SSM state dimension
    mamba_expand: int = 2           # Mamba inner dimension multiplier (d_inner = d_model * expand)


    t_grid: List[int] = field(
        default_factory=lambda: [12, 24, 36, 48, 60]
    )


    random_seed: int = 42           # Must match Baseline/config.py RANDOM_SEED
    batch_size_physical: int = 2    # Per-GPU batch size (memory-limited by 128^3 volumes)
    gradient_accumulation_steps: int = 8 # Effective batch = physical x accumulation
    max_epochs_stage1: int = 5      # LP-FT Stage 1: linear probing (head only)
    max_epochs_stage2: int = 100    # LP-FT Stage 2: sequence model + head fine-tuning
    early_stopping_patience: int = 15  # Epochs without C_td improvement before stopping
    lr_projection_head: float = 1e-4   # BrainIAC projection head learning rate
    lr_sequence_model: float = 5e-5    # Longformer/Mamba learning rate
    lr_survival_head: float = 1e-3     # TraCeR head learning rate (small module, higher LR)
    weight_decay: float = 0.01         # AdamW weight decay
    grad_clip_max_norm: float = 1.0    # Gradient clipping threshold


    dvf_dir: Path = field(
        default_factory=lambda: Path("data/dvf")
    )
    tabular_path: Path = field(
        default_factory=lambda: Path("Baseline/outputs/mci_tensor.npy")
    )
    survival_labels_dir: Path = field(
        default_factory=lambda: Path("Baseline/outputs")
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: Path("Transformer/checkpoints")
    )
    figures_dir: Path = field(
        default_factory=lambda: Path("Transformer/figures")
    )

    def validate(self) -> None:
        """Check that every field is within its valid range.

        Raises:
            ValueError: If any hyperparameter is invalid, with a
                descriptive message indicating which field failed and
                what the valid range is.
        """
        if self.d_model % 8 != 0:
            raise ValueError(
                f"d_model must be divisible by 8 for multi-head attention "
                f"compatibility, got {self.d_model}"
            )
        if self.d_model < 1:
            raise ValueError(
                f"d_model must be positive, got {self.d_model}"
            )
        if self.n_tokens_per_visit < 1:
            raise ValueError(
                f"n_tokens_per_visit must be >= 1, got "
                f"{self.n_tokens_per_visit}"
            )
        if self.v_max < 1:
            raise ValueError(
                f"v_max must be >= 1, got {self.v_max}"
            )
        if self.n_grid < 1:
            raise ValueError(
                f"n_grid must be >= 1, got {self.n_grid}"
            )
        if self.n_risks < 1:
            raise ValueError(
                f"n_risks must be >= 1, got {self.n_risks}"
            )
        if self.pma_seeds < 1:
            raise ValueError(
                f"pma_seeds must be >= 1, got {self.pma_seeds}"
            )
        if self.longformer_window < 1:
            raise ValueError(
                f"longformer_window must be >= 1, got "
                f"{self.longformer_window}"
            )
        if not (0.0 <= self.modality_dropout_p <= 1.0):
            raise ValueError(
                f"modality_dropout_p must be in [0, 1], got "
                f"{self.modality_dropout_p}"
            )
        if self.mamba_d_state < 1:
            raise ValueError(
                f"mamba_d_state must be >= 1, got {self.mamba_d_state}"
            )
        if self.mamba_expand < 1:
            raise ValueError(
                f"mamba_expand must be >= 1, got {self.mamba_expand}"
            )
        if len(self.t_grid) != self.n_grid:
            raise ValueError(
                f"t_grid length ({len(self.t_grid)}) must equal n_grid "
                f"({self.n_grid})"
            )
        if self.random_seed < 0:
            raise ValueError(
                f"random_seed must be non-negative, got {self.random_seed}"
            )
        if self.batch_size_physical < 1:
            raise ValueError(
                f"batch_size_physical must be >= 1, got "
                f"{self.batch_size_physical}"
            )
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got "
                f"{self.gradient_accumulation_steps}"
            )
        if self.max_epochs_stage1 < 1:
            raise ValueError(
                f"max_epochs_stage1 must be >= 1, got "
                f"{self.max_epochs_stage1}"
            )
        if self.max_epochs_stage2 < 1:
            raise ValueError(
                f"max_epochs_stage2 must be >= 1, got "
                f"{self.max_epochs_stage2}"
            )
        if self.early_stopping_patience < 1:
            raise ValueError(
                f"early_stopping_patience must be >= 1, got "
                f"{self.early_stopping_patience}"
            )
        if self.lr_projection_head <= 0:
            raise ValueError(
                f"lr_projection_head must be positive, got "
                f"{self.lr_projection_head}"
            )
        if self.lr_sequence_model <= 0:
            raise ValueError(
                f"lr_sequence_model must be positive, got "
                f"{self.lr_sequence_model}"
            )
        if self.lr_survival_head <= 0:
            raise ValueError(
                f"lr_survival_head must be positive, got "
                f"{self.lr_survival_head}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got "
                f"{self.weight_decay}"
            )
        if self.grad_clip_max_norm <= 0:
            raise ValueError(
                f"grad_clip_max_norm must be positive, got "
                f"{self.grad_clip_max_norm}"
            )
