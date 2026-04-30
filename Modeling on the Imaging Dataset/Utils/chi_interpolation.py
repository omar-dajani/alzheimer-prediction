"""
chi_interpolation — Constant Hazard Interpolation for continuous survival curve evaluation.

Pipeline position:
    Phase 7 of the ADNI Advanced Survival Pipeline. Utility layer between the
    TraCeR survival head (Phase 6) and the evaluation metrics (Phase 8).
    Also used during training to compute IBS on the validation set.

The problem this solves:
    TraCeR outputs S(t) at G=5 discrete grid points: S(12), S(24), ..., S(60) months.
    Evaluation metrics (Antolini C_td, IBS) query S(t) at arbitrary continuous times —
    the actual observed event/censoring times in the dataset, which almost never fall
    exactly on a grid point. CHI provides the continuous bridge.

Dependencies:
    - Transformer/config/model_config.py
    - Transformer/models/survival_head.py (consumes hazards_to_survival output)
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


def validate_monotonicity(S: Tensor, tol: float = 1e-5) -> bool:
    """Check whether a survival tensor is monotonically non-increasing.

    Floating-point operations in the survival head (log, exp, cumsum)
    can produce differences of ~1e-7 that are not meaningful violations
    of monotonicity. The default tolerance of 1e-5 is conservative enough
    to catch real violations while ignoring numerical noise.

    Args:
        S: Survival probabilities of shape [B, G] or [G].
            Values should be in (0, 1].
        tol: Maximum allowed increase between consecutive time points.
            Default 1e-5 — any increase larger than this is flagged.

    Returns:
        True if S is non-increasing across the last dimension
        (within tolerance), False otherwise.
    """
    # S[..., 1:] - S[..., :-1] should be <= 0 for non-increasing
    # Allow small positive diffs up to tol for numerical noise
    diffs = S[..., 1:] - S[..., :-1]
    return bool((diffs <= tol).all())


class ConstantHazardInterpolator(nn.Module):
    """Piecewise-exponential interpolation for discrete survival curves.

    Converts the G-point discrete survival output from TraCeR into a
    continuous survival function that can be queried at arbitrary times.
    Uses the constant hazard assumption within each interval, which
    guarantees monotonically non-increasing interpolated S(t) by
    construction.

    CHI vs CDI (linear interpolation):
        CDI: S(t) = S_left + (S_right - S_left) * (t - t_left) / dt
             Can produce S(t) > S_left due to numerical error — violates
             the fundamental property of survival functions.
        CHI: S(t) = S_left * exp(-h * (t - t_left))
             Exponential decay cannot increase — monotonicity guaranteed.

    The t_grid must exactly match the grid used to train the survival
    head. Mismatched grids produce silently wrong interpolation — no
    runtime error is raised because the shapes still match.

    Tensor shapes:
        Input:  S_discrete [B, G] — discrete survival at G grid points
        Input:  t_query [B] or [B, T] — continuous query times in months
        Output: same shape as t_query — S(t_query) for each subject

    Args:
        t_grid: Temporal grid points in months, strictly increasing.
            Default from ModelConfig: [12, 24, 36, 48, 60].
    """

    def __init__(self, t_grid: Tensor) -> None:
        """Initialize the interpolator with a temporal grid.

        Args:
            t_grid: Strictly increasing temporal grid points in months.
                Must match the grid used to train the survival head.

        Raises:
            AssertionError: If t_grid is not strictly increasing.
        """
        super().__init__()
        assert (t_grid[1:] > t_grid[:-1]).all(), (
            "t_grid must be strictly increasing. Check ModelConfig.t_grid. "
            f"Got: {t_grid.tolist()}"
        )
        # Register as buffer — fixed grid, not a learnable parameter.
        # Moves with model to correct device via .to(device) and is
        # included in state_dict for checkpoint reproducibility.
        self.register_buffer("t_grid", t_grid)
        self.G = t_grid.shape[0]

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
    ) -> "ConstantHazardInterpolator":
        """Preferred constructor — ensures grid matches survival head.

        Always use this instead of the raw constructor unless you have
        a specific reason to use a different grid. Mismatched grids
        between the interpolator and the survival head produce silently
        wrong interpolation results.

        Args:
            config: ModelConfig providing t_grid.

        Returns:
            ConstantHazardInterpolator with the config's temporal grid.
        """
        return cls(
            torch.tensor(config.t_grid, dtype=torch.float32)
        )

    def interpolate(
        self,
        S_discrete: Tensor,
        t_query: Tensor,
    ) -> Tensor:
        """Interpolate discrete survival to continuous query times via CHI.

        Uses the piecewise-exponential (constant hazard) formula:
            h_k  = -log(S(t_k) / S(t_{k-1})) / (t_k - t_{k-1})
            S(t) = S(t_{k-1}) * exp(-h_k * (t - t_{k-1}))

        This guarantees monotonically non-increasing S(t) because
        exp(-h*dt) is strictly decreasing for h > 0. Linear
        interpolation (CDI) cannot guarantee this and may produce
        S(t) > S(t_{k-1}) due to numerical error.

        Fully vectorized — no Python loops over batch or time dimensions.
        Uses torch.bucketize to find intervals and torch.gather to
        extract per-query values.

        Args:
            S_discrete: Discrete survival probabilities [B, G] from
                TraCeRSurvivalHead.hazards_to_survival().
            t_query: Continuous query times in months.
                Shape [B] for one query per subject, or
                shape [B, T] for T queries per subject.

        Returns:
            Interpolated survival S(t) with same shape as t_query.
        """
        B = S_discrete.shape[0]
        # Track whether input was 1D to restore shape at the end
        squeeze_output = False
        if t_query.ndim == 1:
            t_query = t_query.unsqueeze(-1) # [B] -> [B, 1]
            squeeze_output = True
        T = t_query.shape[1]

        # Step 1: Prepend S(0)=1 and t=0 as left boundary of first interval
        # S(0) = 1.0 by definition — everyone is at risk at time zero
        S_full = torch.cat(
            [torch.ones(B, 1, device=S_discrete.device), S_discrete],
            dim=1,
        )  # [B, G+1]
        t_full = torch.cat(
            [torch.zeros(1, device=self.t_grid.device), self.t_grid],
        )  # [G+1]

        # Step 2: Clamp query times to the grid range
        # t > t_G: return S(t_G) — no extrapolation beyond last grid point
        t_query = t_query.clamp(max=self.t_grid[-1].item())

        # Find which interval each query time falls in
        # bucketize returns the index of the right boundary in t_full
        # Subtract 1 to get the LEFT boundary index (start of interval)
        k = torch.bucketize(t_query, t_full) - 1  # [B, T]
        # Clamp to valid range [0, G-1]:
        # t < t_1 -> use first interval (extrapolate left)
        k = k.clamp(0, self.G - 1)  # [B, T]

        # Step 3: Compute interval-specific constant hazards
        # delta_t: interval widths [G]
        delta_t = t_full[1:] - t_full[:-1]  # [G]
        # Survival ratio S(t_k) / S(t_{k-1}) for each interval
        # 1e-10 guard prevents log(0) on exactly-zero survival edges
        ratio = S_full[:, 1:] / (S_full[:, :-1] + 1e-10) # [B, G]
        # h_k = -log(ratio) / delta_t — interval-specific constant hazard
        # h_k >= 0 because ratio in (0, 1] -> log <= 0 -> -log >= 0
        log_ratio = torch.log(ratio + 1e-10) # [B, G]
        h = -log_ratio / delta_t.unsqueeze(0) # [B, G]

        # Step 4: Gather values for each query's interval using indices k
        # Expand k for gather: [B, T] -> index into [B, G] tensors
        k_expanded = k # [B, T]

        # S_left: survival at left boundary of each query's interval
        S_left = torch.gather(S_full[:, :-1], 1, k_expanded) # [B, T]
        # t_left: time at left boundary of each query's interval
        t_left = t_full[:-1][k_expanded[0]] # [T] — same grid for all B
        t_left = t_full[:-1].unsqueeze(0).expand(B, -1) # [B, G]
        t_left = torch.gather(t_left, 1, k_expanded) # [B, T]
        # h_k: hazard for each query's interval
        h_k = torch.gather(h, 1, k_expanded) # [B, T]

        # Step 5: Apply piecewise-exponential formula
        # S(t) = S_left * exp(-h_k * (t - t_left))
        dt = t_query - t_left # [B, T] — time elapsed within interval
        dt = dt.clamp(min=0.0) # safety: should not be negative
        S_interp = S_left * torch.exp(-h_k * dt) # [B, T]

        if squeeze_output:
            S_interp = S_interp.squeeze(-1) # [B, 1] -> [B]

        return S_interp

    def interpolate_cif(
        self,
        hazards: Tensor,
        survival_head,
        t_query: Tensor,
        cause: int = 0,
    ) -> Tensor:
        """Interpolate cause-specific CIF at arbitrary continuous times.

        Args:
            hazards: Cause-specific hazard rates [B, K, G] from
                TraCeRSurvivalHead.
            survival_head: TraCeRSurvivalHead instance with
                hazards_to_cif method.
            t_query: Query times [B] or [B, T] in months.
            cause: Cause index (default 0 = dementia). Use 1 for
                competing mortality.

        Returns:
            Interpolated CIF with same shape as t_query.
        """
        # Get discrete CIF: [B, K, G]
        cif = survival_head.hazards_to_cif(hazards)  # [B, K, G]
        # Extract specified cause: [B, G]
        cif_cause = cif[:, cause, :]  # [B, G]

        # For CIF interpolation, we interpolate 1 - CIF (which is
        # non-increasing) via CHI, then convert back
        one_minus_cif = 1.0 - cif_cause  # [B, G]
        one_minus_cif_interp = self.interpolate(
            one_minus_cif, t_query
        )
        return 1.0 - one_minus_cif_interp


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("Phase 7 — CHI Interpolation Smoke Test")

    config = ModelConfig()
    pass_count = 0
    fail_count = 0

    # Test 1: from_config constructor
    try:
        chi = ConstantHazardInterpolator.from_config(config)
        assert chi.t_grid.tolist() == [12.0, 24.0, 36.0, 48.0, 60.0]
        print("PASS: from_config grid")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: from_config — {e}")
        fail_count += 1

    # Test 2: validate_monotonicity — valid curve
    try:
        S_valid = torch.tensor([[1.0, 0.8, 0.6, 0.4, 0.2]])
        assert validate_monotonicity(S_valid) is True
        print("PASS: monotone curve detected")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: monotone detection — {e}")
        fail_count += 1

    # Test 3: validate_monotonicity — invalid curve
    try:
        S_invalid = torch.tensor([[1.0, 0.9, 0.95, 0.4, 0.2]])
        assert validate_monotonicity(S_invalid) is False
        print("PASS: non-monotone detected (0.95 > 0.9)")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: non-monotone detection — {e}")
        fail_count += 1

    # Test 4: Interpolation at exact grid point (t=24 -> S=0.8)
    try:
        S_disc = torch.tensor([[1.0, 0.8, 0.6, 0.4, 0.2]])
        t_q = torch.tensor([24.0])
        result = chi.interpolate(S_disc, t_q)
        assert abs(result.item() - 0.8) < 1e-4, (
            f"At t=24: got {result.item()}, expected 0.8"
        )
        print(f"PASS: grid point interpolation — S(24)={result.item():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: grid point — {e}")
        fail_count += 1

    # Test 5: Midpoint interpolation (t=18, between t=12 and t=24)
    try:
        t_mid = torch.tensor([18.0])
        result_mid = chi.interpolate(S_disc, t_mid)
        assert 0.8 < result_mid.item() < 1.0, (
            f"At t=18: got {result_mid.item()}, expected in (0.8, 1.0)"
        )
        print(f"PASS: midpoint interpolation — S(18)={result_mid.item():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: midpoint — {e}")
        fail_count += 1

    # Test 6: CHI < CDI at midpoint (exponential decay < linear)
    try:
        # Linear midpoint between S(12)=1.0 and S(24)=0.8 is 0.9
        cdi_midpoint = 0.9
        assert result_mid.item() < cdi_midpoint, (
            f"CHI {result_mid.item()} should be < CDI {cdi_midpoint}"
        )
        print(
            f"PASS: CHI < CDI at midpoint — "
            f"CHI={result_mid.item():.4f} < CDI={cdi_midpoint}"
        )
        pass_count += 1
    except Exception as e:
        print(f"FAIL: CHI < CDI — {e}")
        fail_count += 1

    # Test 7: Monotonicity of interpolated curve over dense query
    try:
        t_dense = torch.linspace(1, 65, 50).unsqueeze(0) # [1, 50]
        S_dense = chi.interpolate(S_disc, t_dense) # [1, 50]
        assert validate_monotonicity(S_dense), (
            "Interpolated curve is not monotone"
        )
        print("PASS: interpolated curve is monotone (50 points)")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: interpolated monotonicity — {e}")
        fail_count += 1

    # Test 8: Boundary t=0 (before grid start)
    try:
        t_zero = torch.tensor([0.0])
        S_at_zero = chi.interpolate(S_disc, t_zero)
        assert S_at_zero.item() <= 1.0 + 1e-5, (
            f"S(0) = {S_at_zero.item()} > 1"
        )
        print(f"PASS: t=0 boundary — S(0)={S_at_zero.item():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: t=0 boundary — {e}")
        fail_count += 1

    # Test 9: Boundary t=70 (after grid end -> return S(60))
    try:
        t_beyond = torch.tensor([70.0])
        S_at_70 = chi.interpolate(S_disc, t_beyond)
        expected = S_disc[0, -1].item()  # S(60) = 0.2
        assert abs(S_at_70.item() - expected) < 0.05, (
            f"S(70)={S_at_70.item()}, expected ~{expected}"
        )
        print(f"PASS: t>t_G boundary — S(70)={S_at_70.item():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: t>t_G boundary — {e}")
        fail_count += 1

    # Test 10: Batch consistency (4 subjects, query at t=30)
    try:
        S_batch = torch.tensor([
            [1.0, 0.8, 0.6, 0.4, 0.2],
            [0.9, 0.7, 0.5, 0.3, 0.1],
            [0.95, 0.85, 0.75, 0.65, 0.55],
            [0.8, 0.6, 0.4, 0.2, 0.05],
        ])  # [4, 5]
        t_batch = torch.tensor([30.0, 30.0, 30.0, 30.0]) # [4]
        S_result = chi.interpolate(S_batch, t_batch)
        assert S_result.shape == (4,), f"Batch shape: {S_result.shape}"
        # Each result should be between S(24) and S(36)
        for i in range(4):
            s24 = S_batch[i, 1].item() # S at grid point 2 (24 months)
            s36 = S_batch[i, 2].item() # S at grid point 3 (36 months)
            assert s36 - 1e-5 <= S_result[i].item() <= s24 + 1e-5, (
                f"Subject {i}: S(30)={S_result[i]:.4f} not in "
                f"[{s36:.4f}, {s24:.4f}]"
            )
        print(f"PASS: batch query — shape {tuple(S_result.shape)}, values in range")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: batch query — {e}")
        fail_count += 1

    # Test 11: Multi-query mode [B, T]
    try:
        t_multi = torch.tensor([
            [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0],
            [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0],
        ])  # [2, 10]
        S_multi_input = torch.tensor([
            [1.0, 0.8, 0.6, 0.4, 0.2],
            [0.9, 0.7, 0.5, 0.3, 0.1],
        ])  # [2, 5]
        S_multi = chi.interpolate(S_multi_input, t_multi)
        assert S_multi.shape == (2, 10), f"Multi shape: {S_multi.shape}"
        print(f"PASS: multi-query shape — {tuple(S_multi.shape)}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: multi-query — {e}")
        fail_count += 1

    # Summary
    print(f"\nResults: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("PASS — All assertions hold.")
    else:
        print("FAIL — Some tests did not pass.")
