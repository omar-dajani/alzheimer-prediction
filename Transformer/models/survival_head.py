"""
survival_head — TraCeR competing-risks discrete hazard head for MCI->Dementia prediction.

Inputs:
    x: [B, d_input] — flattened PMA output, where d_input = pma.output_dim()
       For default config: d_input = 8 * 512 = 4096.

Outputs:
    hazards: [B, K, G] — cause-specific discrete hazard rates after sigmoid
             and constraint enforcement. Each value in (0, 1).
             K = n_risks (default 1 for single-event dementia-only survival)
             G = n_grid  = 5: time intervals [0,12), [12,24), [24,36), [36,48), [48,60]

Architecture:
    Shared trunk: Linear(d_input, 256) -> GELU -> Dropout(0.3) -> Linear(256, 128) -> GELU
    Per-cause nets: K x [Linear(128, 64) -> GELU -> Linear(64, G)]
    Post-processing: sigmoid -> total hazard constraint enforcement

Inference helpers:
    hazards_to_survival(hazards) -> overall survival S(t_j) [B, G]
    hazards_to_cif(hazards) -> cause-specific CIF F_k(t_j) [B, K, G]
    These are consumed by chi_interpolation.py (Phase 7) for continuous S(t).

Dependencies:
    - Transformer/config/model_config.py
    - Transformer/models/pooling.py (for d_input via pma.output_dim())
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)
from Transformer.config.model_config import ModelConfig


logger = logging.getLogger(__name__)

# Temporal grid default (months) — must match ModelConfig.t_grid
DEFAULT_T_GRID = [12, 24, 36, 48, 60]

# Shared trunk architecture constants
TRUNK_HIDDEN_1 = 256
TRUNK_HIDDEN_2 = 128
TRUNK_DROPOUT = 0.3

# Cause-specific subnetwork constants
CAUSE_HIDDEN = 64

# Numerical stability — hazard bounds after sigmoid
# Prevents log(0) in downstream IPCW loss and ensures constraint
# enforcement doesn't produce exactly 0.0 or 1.0
HAZARD_EPS = 1e-4
HAZARD_MAX = 1.0 - HAZARD_EPS


class TraCeRSurvivalHead(nn.Module):
    """TraCeR competing-risks discrete hazard head.

    Produces cause-specific discrete hazard rates from the PMA pooling
    summary. Uses sigmoid activation to allow each hazard
    to be independently bounded in (0, 1).

    Architecture:
        Shared trunk learns domain-general features common to both
        competing risks. Cause-specific subnetworks learn what
        distinguishes MCI->Dementia conversion (cause 0) from
        non-dementia mortality (cause 1).

        shared: Linear(d_input, 256) -> GELU -> Dropout(0.3) -> Linear(256, 128) -> GELU
        cause_nets: K x [Linear(128, 64) -> GELU -> Linear(64, G)]

    Tensor shapes:
        Input:  x — [B, d_input] (from PMA: d_input = pma.output_dim())
        Output: hazards — [B, K, G] where K=n_risks, G=n_grid

    Args:
        d_input: Input dimension from PMA (use pma.output_dim(), never hardcode).
        config: ModelConfig providing n_risks and n_grid.
    """

    def __init__(self, d_input: int, config: ModelConfig) -> None:
        """Initialize shared trunk and cause-specific subnetworks.

        Args:
            d_input: Dimension of PMA output vector. Passed explicitly
                from pma.output_dim() to avoid hardcoded dependencies.
            config: ModelConfig providing n_risks (default K=1) and n_grid (G=5).
        """
        super().__init__()
        self.n_risks = config.n_risks
        self.n_grid = config.n_grid

        # Shared trunk — learns features common to both competing risks
        self.shared_trunk = nn.Sequential(
            nn.Linear(d_input, TRUNK_HIDDEN_1),
            nn.GELU(),
            nn.Dropout(TRUNK_DROPOUT),
            nn.Linear(TRUNK_HIDDEN_1, TRUNK_HIDDEN_2),
            nn.GELU(),
        )

        # Per-cause subnetworks — learn cause-specific hazard patterns
        # nn.ModuleList ensures all K networks are registered as
        # submodules and included in state_dict() (a Python list would not)
        self.cause_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(TRUNK_HIDDEN_2, CAUSE_HIDDEN),
                nn.GELU(),
                nn.Linear(CAUSE_HIDDEN, config.n_grid),
            )
            for _ in range(config.n_risks)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """Produce cause-specific discrete hazard rates.

        Args:
            x: PMA output [B, d_input].

        Returns:
            Hazard rates [B, K, G] after sigmoid, clamping, and
            constraint enforcement. Each value is in [HAZARD_EPS, HAZARD_MAX].
            Sum across K at each time point <= 1.
        """
        B = x.shape[0]

        # Shared feature extraction
        shared = self.shared_trunk(x)  # [B, 128]

        # Per-cause hazard logits
        cause_logits = []
        for k, cause_net in enumerate(self.cause_nets):
            logits_k = cause_net(shared)  # [B, G]
            cause_logits.append(logits_k)

        # Stack into [B, K, G]
        hazard_logits = torch.stack(cause_logits, dim=1)  # [B, K, G]

        # sigmoid, NOT softmax — each hazard is independently bounded in (0, 1).
        # softmax would force sum=1 across time, implying certain eventual event = wrong
        hazards = torch.sigmoid(hazard_logits)  # [B, K, G]

        # Phase 5: Clamp to [eps, 1-eps] to prevent log(0) in IPCW loss
        # and ensure constraint enforcement doesn't produce exact 0/1
        hazards = hazards.clamp(min=HAZARD_EPS, max=HAZARD_MAX)

        # Total hazard constraint enforcement: sum_k h_k(t_j) <= 1
        # Rescale only when sum > 1 — preserves hazard magnitudes in the
        # normal operating range and only intervenes when violated
        total = hazards.sum(dim=1, keepdim=True) # [B, 1, G]
        scale = torch.where(
            total > 1.0,
            1.0 / total.clamp(min=HAZARD_EPS),  # Prevent div by zero
            torch.ones_like(total),
        )
        hazards = hazards * scale # [B, K, G]

        # Re-clamp after scaling (scaling can push below eps)
        hazards = hazards.clamp(min=HAZARD_EPS, max=HAZARD_MAX)

        assert hazards.shape == (B, self.n_risks, self.n_grid), (
            f"Hazard shape mismatch: got {tuple(hazards.shape)}, "
            f"expected ({B}, {self.n_risks}, {self.n_grid})."
        )

        return hazards

    def hazards_to_survival(self, hazards: Tensor) -> Tensor:
        """Compute overall survival function from cause-specific hazards.

        Mathematical formula:
            S(t_j) = prod_{l<=j} (1 - sum_k h_k(t_l))

        The product is computed in log-space for numerical stability:
            log S(t_j) = sum_{l<=j} log(1 - sum_k h_k(t_l))

        Phase 5 hardening: explicit eps prevents log(0) when total
        hazard approaches 1.0 due to float32 precision.

        Args:
            hazards: Cause-specific hazard rates [B, K, G].

        Returns:
            Overall survival probabilities [B, G]. Each value is in (0, 1].
            S is monotonically non-increasing across the G time points.
        """
        # Sum hazards across all K causes at each time point
        total_hazard = hazards.sum(dim=1)  # [B, G]

        # Clamp to [0, 1 - eps] to prevent log(0) or log(negative)
        total_hazard = total_hazard.clamp(max=1.0 - HAZARD_EPS)

        # Log-product form for numerical stability
        log_surv = torch.log(
            1.0 - total_hazard
        ).cumsum(dim=1)  # [B, G]

        return torch.exp(log_surv)  # [B, G]

    def hazards_to_cif(self, hazards: Tensor) -> Tensor:
        """Compute cause-specific Cumulative Incidence Function (CIF).

        Mathematical formula:
            F_k(t_j) = sum_{l<=j} h_k(t_l) * S(t_{l-1})

        where S(t_0) = 1.0 by definition (everyone at risk at time 0).

        Args:
            hazards: Cause-specific hazard rates [B, K, G].

        Returns:
            Cause-specific CIF [B, K, G]. F_k is monotonically
            non-decreasing across time for each cause k.
        """
        B = hazards.shape[0]

        # Overall survival at each grid point
        S = self.hazards_to_survival(hazards) # [B, G]

        # Lagged survival: S(t_{j-1}) with S(t_0) = 1.0
        S_lagged = torch.cat(
            [torch.ones(B, 1, device=hazards.device), S[:, :-1]],
            dim=1,
        ) # [B, G]

        # CIF increments: h_k(t_l) * S(t_{l-1})
        # S_lagged [B, G] -> [B, 1, G] for broadcasting with hazards [B, K, G]
        cif_increments = hazards * S_lagged.unsqueeze(1) # [B, K, G]

        # Cumulative sum gives F_k(t_j) = sum_{l<=j} increments
        cif = cif_increments.cumsum(dim=-1) # [B, K, G]

        return cif


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("Phase 6a — TraCeRSurvivalHead Smoke Test")

    config = ModelConfig()
    d_input = 4096  # pma.output_dim() for default config
    pass_count = 0
    fail_count = 0

    # Test 1: Instantiation
    try:
        head = TraCeRSurvivalHead(d_input=d_input, config=config)
        head.eval()
        print("PASS: TraCeRSurvivalHead instantiated")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: instantiation — {e}")
        fail_count += 1

    # Test 2: Forward pass shape
    try:
        x = torch.randn(4, d_input)  # B=4 subjects
        with torch.no_grad():
            hazards = head(x)
        expected_shape = (4, config.n_risks, config.n_grid)
        assert hazards.shape == expected_shape, (
            f"Shape: {hazards.shape} != {expected_shape}"
        )
        print(f"PASS: forward shape — {tuple(hazards.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: forward shape — {e}")
        fail_count += 1

    # Test 3: Hazard bounds — all values in (0, 1)
    try:
        assert hazards.min() >= 0.0, f"Min hazard {hazards.min()} < 0"
        assert hazards.max() <= 1.0, f"Max hazard {hazards.max()} > 1"
        print("PASS: hazard bounds — all in [0, 1]")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: hazard bounds — {e}")
        fail_count += 1

    # Test 4: Total hazard constraint sum_k h_k(t_j) <= 1
    try:
        total = hazards.sum(dim=1)  # [B, G]
        assert total.max() <= 1.0 + 1e-5, (
            f"Total hazard max {total.max()} > 1"
        )
        print(f"PASS: total hazard constraint — max={total.max():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: total hazard constraint — {e}")
        fail_count += 1

    # Test 5: Survival shape
    try:
        S = head.hazards_to_survival(hazards)
        assert S.shape == (4, 5), f"Survival shape: {S.shape}"
        print(f"PASS: survival shape — {tuple(S.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: survival shape — {e}")
        fail_count += 1

    # Test 6: Survival bounds — all in (0, 1]
    try:
        assert S.min() > 0.0, f"Min survival {S.min()} <= 0"
        assert S.max() <= 1.0 + 1e-5, f"Max survival {S.max()} > 1"
        print("PASS: survival bounds — all in (0, 1]")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: survival bounds — {e}")
        fail_count += 1

    # Test 7: Survival monotonicity (non-increasing across time)
    try:
        diffs = S[:, 1:] - S[:, :-1]  # [B, G-1]
        assert (diffs <= 1e-5).all(), (
            f"Survival not monotone: max increase = {diffs.max()}"
        )
        print("PASS: survival monotonicity")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: survival monotonicity — {e}")
        fail_count += 1

    # Test 8: CIF shape
    try:
        cif = head.hazards_to_cif(hazards)
        expected_cif_shape = (4, config.n_risks, config.n_grid)
        assert cif.shape == expected_cif_shape, f"CIF shape: {cif.shape} != {expected_cif_shape}"
        print(f"PASS: CIF shape — {tuple(cif.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: CIF shape — {e}")
        fail_count += 1

    # Test 9: CIF non-negative
    try:
        assert cif.min() >= -1e-7, f"CIF min {cif.min()} < 0"
        print("PASS: CIF non-negative")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: CIF non-negative — {e}")
        fail_count += 1

    # Test 10: CIF monotonically non-decreasing
    try:
        cif_diffs = cif[:, :, 1:] - cif[:, :, :-1]  # [B, K, G-1]
        assert (cif_diffs >= -1e-5).all(), (
            f"CIF not monotone: min diff = {cif_diffs.min()}"
        )
        print("PASS: CIF monotonicity")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: CIF monotonicity — {e}")
        fail_count += 1

    # Summary
    print(f"\nResults: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("PASS — All assertions hold.")
    else:
        print("FAIL — Some tests did not pass.")
