"""
ipcw_loss — IPCW-weighted negative log-likelihood loss for competing-risks survival.

Pipeline position:
    Phase 6 training utility. Called by the trainer (Phase 9) at every training step.
    Uses hazards from TraCeRSurvivalHead and survival labels from Baseline/outputs/.

What IPCW corrects:
    Standard NLL treats all subjects equally regardless of when they were censored.
    Under informative censoring (subjects who drop out early differ systematically
    from those followed longer), this introduces bias. IPCW re-weights each subject's
    loss contribution by 1/G(T_i), the inverse probability that subject i was still
    being observed at their event time T_i. Subjects censored in high-censoring regions
    (small G(T_i)) receive large weights — they are rare, informative observations.

Event label convention (used throughout this module and survival_head.py):
    CENSORED = 0 — subject did not experience either event before last follow-up
    DEMENTIA = 1 — MCI -> Dementia conversion (cause index k=0 in hazard tensor)
    MORTALITY = 2 — non-dementia death (cause index k=1 in hazard tensor)
    Mapping: event_label - 1 = cause_index  (for event_label > 0)

Label source:
    Baseline/outputs/mci_y_event.npy — integer array with values {0, 1, 2}
    Baseline/outputs/mci_y_duration.npy — float array, time in months

Dependencies:
    - Transformer/config/model_config.py
    - lifelines (KaplanMeierFitter for censoring survival estimation)
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)


logger = logging.getLogger(__name__)


# Event label constants — used throughout the survival pipeline.
# These correspond to the encoding in Baseline/outputs/mci_y_event.npy
EVENT_CENSORED = 0 # Subject did not experience either event
EVENT_DEMENTIA = 1 # MCI -> Dementia (maps to cause index k=0 in hazards)
EVENT_MORTALITY = 2 # Non-dementia death (maps to cause index k=1 in hazards)


class CensoringSurvivalEstimator:
    """Reverse Kaplan-Meier estimator for censoring survival G(t).

    Estimates the probability that a subject is still being observed
    (not yet censored) at time t: G(t) = P(C > t), where C is the
    censoring time.

    The "reverse KM" trick swaps event and censoring indicators:
    subjects who experienced an event (dementia or death) are treated
    as "censored" for the censoring process, and subjects who were
    administratively censored are treated as "events." This gives
    a Kaplan-Meier estimate of the censoring survival G(t).

    G(t) is used to compute IPCW weights: w_i = 1 / G(T_i).
    Subjects observed in high-censoring regions (small G(T_i)) receive
    large weights — they are rare, informative observations that
    counterbalance the censoring-induced selection bias.

    This estimator is fitted once on the training set and reused for
    every training step. It must NEVER be refit on the validation or
    test set — doing so introduces information leakage.

    The estimator is pickle-serializable for checkpoint inclusion.
    """

    def __init__(self) -> None:
        """Initialize an unfitted censoring survival estimator."""
        from lifelines import KaplanMeierFitter
        self._kmf = KaplanMeierFitter()
        self.is_fitted = False

    def fit(
        self,
        durations: np.ndarray,
        events: np.ndarray,
    ) -> "CensoringSurvivalEstimator":
        """Fit the reverse Kaplan-Meier on training data.

        The reverse KM swaps the event indicator: censored subjects
        (event=0) become "events" for the censoring process, and
        event subjects (event>0) become "censored." This estimates
        G(t) = P(C > t) rather than S(t) = P(T > t).

        Args:
            durations: Observed times in months [n_subjects].
            events: Event labels {0, 1, 2} [n_subjects].
                0 = censored, 1 = dementia, 2 = mortality.

        Returns:
            Self, for method chaining.
        """
        # Reverse KM: event_observed = 1 when subject was CENSORED (event=0)
        # This estimates the censoring survival G(t), not the event survival
        censoring_events = (events == EVENT_CENSORED).astype(int)
        self._kmf.fit(durations, event_observed=censoring_events)
        self.is_fitted = True
        logger.info(
            "CensoringSurvivalEstimator fitted on %d subjects "
            "(%.1f%% censored).",
            len(durations),
            100.0 * censoring_events.mean(),
        )
        return self

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Evaluate censoring survival G(t) at given times.

        Args:
            t: Times in months at which to evaluate G(t) [n_queries].

        Returns:
            G(t) values [n_queries], each in [1e-7, 1.0].

        Raises:
            RuntimeError: If the estimator has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "CensoringSurvivalEstimator has not been fitted. "
                "Call .fit(durations, events) first."
            )
        # Evaluate KM survival function at each query time
        # lifelines predict() returns a DataFrame; extract values
        g_values = self._kmf.predict(t).values

        # Clip to prevent division by zero in IPCW weights
        g_values = np.clip(g_values, 1e-7, 1.0)

        return g_values

    def save(self, path: Path) -> None:
        """Serialize the estimator to disk for checkpoint inclusion.

        Args:
            path: Path to save the pickle file.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("CensoringSurvivalEstimator saved to %s.", path)

    @classmethod
    def load(cls, path: Path) -> "CensoringSurvivalEstimator":
        """Load a previously saved estimator from disk.

        Args:
            path: Path to the pickle file.

        Returns:
            Restored CensoringSurvivalEstimator instance.
        """
        with open(path, "rb") as f:
            estimator = pickle.load(f)
        logger.info("CensoringSurvivalEstimator loaded from %s.", path)
        return estimator


def ipcw_survival_loss(
    hazards: Tensor,
    durations: Tensor,
    events: Tensor,
    censoring_estimator: CensoringSurvivalEstimator,
    t_grid: Tensor,
) -> Tensor:
    """IPCW-weighted negative log-likelihood loss for competing risks.

    Re-weights each subject's contribution by 1/G(T_i), the inverse
    probability that subject i was still being observed at their event
    time. This corrects for the bias introduced by informative
    censoring, where early dropouts differ systematically from subjects
    followed longer.

    Weight truncation at the 5th percentile of G(t) prevents
    pathological weight inflation in the right tail where few subjects
    remain at risk. This trades a small amount of asymptotic bias for
    finite variance — preferable in practice.

    Loss formulation:
        For event subjects (event > 0):
            L_i = -w_i * [log h_k(t_j) + log S(t_{j-1})]
            where k = event - 1 (cause index), j = time interval index
        For censored subjects (event = 0):
            L_i = -w_i * log S(t_j)
            penalizes low predicted survival at censoring time

    Args:
        hazards: Cause-specific hazard rates [B, K, G] from
            TraCeRSurvivalHead.
        durations: Observed times in months [B].
        events: Event labels [B], using EVENT_* constants:
            0 = censored, 1 = dementia, 2 = mortality.
        censoring_estimator: Fitted CensoringSurvivalEstimator for
            computing IPCW weights.
        t_grid: Temporal grid points in months [G].

    Returns:
        Scalar mean IPCW-NLL loss for backpropagation.
    """
    B, K, G = hazards.shape

    # Compute overall survival in log-space for the loss
    total_hazard = hazards.sum(dim=1)  # [B, G]
    # 1e-7 guard prevents log(0) when total hazard approaches 1.0
    log_surv = torch.log(
        1.0 - total_hazard + 1e-7
    ).cumsum(dim=1)  # [B, G]

    # Map each subject's duration to a discrete time interval index
    interval_idx = torch.bucketize(
        durations, t_grid
    ) - 1  # [B] — 0-indexed interval

    # Compute IPCW weights from censoring survival estimator
    g_values = censoring_estimator(durations.detach().cpu().numpy())
    # Truncate at 5th percentile — G(t) -> 0 in the right tail causes
    # weights to blow up. Truncation trades a small amount of asymptotic
    # bias for finite variance.
    threshold = np.percentile(g_values, 5)
    g_values = np.clip(g_values, threshold, None)
    weights = torch.tensor(
        1.0 / g_values, dtype=torch.float32, device=hazards.device
    )  # [B]

    # Compute per-subject IPCW-weighted NLL
    nll = torch.zeros(B, device=hazards.device)

    for i in range(B):
        j = interval_idx[i].clamp(0, G - 1)

        if events[i] == EVENT_CENSORED:
            # Censored: penalize low predicted survival at censoring time
            nll[i] = -weights[i] * log_surv[i, j]
        else:
            # Event observed: penalize low hazard for the correct cause
            k = int(events[i].item()) - 1  # DEMENTIA(1)->k=0, MORTALITY(2)->k=1
            log_h = torch.log(hazards[i, k, j] + 1e-7)
            if j > 0:
                log_s_prev = log_surv[i, j - 1]
            else:
                # S(t_0) = 1.0 by definition -> log(1) = 0
                log_s_prev = torch.tensor(
                    0.0, device=hazards.device
                )
            nll[i] = -weights[i] * (log_h + log_s_prev)

    return nll.mean()


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("Phase 6b — IPCW Loss Smoke Test")

    pass_count = 0
    fail_count = 0

    # Create dummy survival data: 8 subjects, mix of events
    durations = np.array(
        [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 48.0, 60.0]
    )
    events = np.array(
        [0, 1, 0, 2, 1, 0, 1, 0]
    )  # 0=censored, 1=dementia, 2=mortality

    # Test 1: Fit CensoringSurvivalEstimator
    try:
        estimator = CensoringSurvivalEstimator()
        estimator.fit(durations, events)
        assert estimator.is_fitted
        print("PASS: estimator fitted")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: estimator fitting — {e}")
        fail_count += 1

    # Test 2: Estimator bounds — G(t) in (0, 1]
    try:
        test_times = np.array([6.0, 12.0, 24.0, 36.0, 60.0])
        g_vals = estimator(test_times)
        assert g_vals.min() > 0.0, f"G(t) min {g_vals.min()} <= 0"
        assert g_vals.max() <= 1.0, f"G(t) max {g_vals.max()} > 1"
        print(f"PASS: estimator bounds — G(t) in [{g_vals.min():.4f}, {g_vals.max():.4f}]")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: estimator bounds — {e}")
        fail_count += 1

    # Test 3: Save/load round-trip
    try:
        import tempfile
        tmp_path = Path(tempfile.mktemp(suffix=".pkl"))
        estimator.save(tmp_path)
        loaded = CensoringSurvivalEstimator.load(tmp_path)
        g_before = estimator(test_times)
        g_after = loaded(test_times)
        assert np.allclose(g_before, g_after), (
            f"Mismatch after load: {g_before} vs {g_after}"
        )
        tmp_path.unlink()  # cleanup
        print("PASS: estimator serialization round-trip")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: estimator serialization — {e}")
        fail_count += 1

    # Test 4: IPCW loss is scalar
    try:
        B = 8
        hazards = torch.rand(B, 2, 5) * 0.3  # low hazards to stay < 1
        dur_tensor = torch.tensor(durations, dtype=torch.float32)
        evt_tensor = torch.tensor(events, dtype=torch.long)
        t_grid = torch.tensor(
            [12.0, 24.0, 36.0, 48.0, 60.0], dtype=torch.float32
        )
        loss = ipcw_survival_loss(
            hazards, dur_tensor, evt_tensor, estimator, t_grid
        )
        assert loss.ndim == 0, f"Loss not scalar: shape {loss.shape}"
        print(f"PASS: loss is scalar — value={loss.item():.4f}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: loss scalar — {e}")
        fail_count += 1

    # Test 5: Loss is finite
    try:
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        print("PASS: loss is finite")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: loss finite — {e}")
        fail_count += 1

    # Test 6: Unfitted estimator raises RuntimeError
    try:
        unfitted = CensoringSurvivalEstimator()
        try:
            unfitted(np.array([12.0]))
            print("FAIL: unfitted estimator did not raise")
            fail_count += 1
        except RuntimeError:
            print("PASS: unfitted estimator raises RuntimeError")
            pass_count += 1
    except Exception as e:
        print(f"FAIL: unfitted check — {e}")
        fail_count += 1

    # Summary
    print(f"\nResults: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("PASS — All assertions hold.")
    else:
        print("FAIL — Some tests did not pass.")
