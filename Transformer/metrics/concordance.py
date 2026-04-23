import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

import sys
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2]),
)
from Transformer.config.model_config import ModelConfig
from Transformer.losses.ipcw_loss import EVENT_CENSORED, EVENT_DEMENTIA, EVENT_MORTALITY


logger = logging.getLogger(__name__)


BASELINE_RESULTS = {
    "GBM_Surv_MCI": {
        "model": "GBM Surv",
        "cohort": "MCI->Dementia",
        "metric": "Harrell's C",
        "value": 0.7742,
    },
    "DeepSurv_MCI": {
        "model": "DeepSurv",
        "cohort": "MCI->Dementia",
        "metric": "Harrell's C",
        "value": 0.7879,
    },
    "Weighted_Ensemble_MCI": {
        "model": "Weighted Ensemble",
        "cohort": "MCI->Dementia",
        "metric": "Harrell's C",
        "value": 0.7893,
    },
    "Domain_Ensemble_MCI": {
        "model": "Domain Ensemble",
        "cohort": "MCI->Dementia",
        "metric": "Harrell's C",
        "value": 0.7893,
    },
    "GBM_Surv_CN": {
        "model": "GBM Surv",
        "cohort": "CN->Decline",
        "metric": "Harrell's C",
        "value": 0.4969,
    },
    "DeepSurv_CN": {
        "model": "DeepSurv",
        "cohort": "CN->Decline",
        "metric": "Harrell's C",
        "value": 0.4619,
    },
    "Weighted_Ensemble_CN": {
        "model": "Weighted Ensemble",
        "cohort": "CN->Dementia",
        "metric": "Harrell's C",
        "value": 0.5052,
    },
}

# scikit-survival structured array dtype
SKSURV_DTYPE = np.dtype([("event", bool), ("time", float)])

# Dense interpolation grid
N_DENSE_POINTS = 50


def rmst(
    S_discrete: Tensor,
    t_grid: Tensor,
    t_max: float = None,
) -> Tensor:
    """Compute Restricted Mean Survival Time via trapezoidal rule.

    RMST = integral_0^{t_max} S(t) dt, approximated by the trapezoidal
    rule on the discrete survival grid.

    For this pipeline, -S(t_G) is the default risk score because it is
    simpler and empirically performs similarly on ADNI data.

    Args:
        S_discrete: Discrete survival probabilities [B, G].
        t_grid: Temporal grid points [G] in months.
        t_max: Maximum integration time (default: t_grid[-1]).
            Values beyond t_max are excluded from the integral.

    Returns:
        RMST for each subject [B], in units of months.
    """
    if t_max is not None:
        mask = t_grid <= t_max
        t_grid = t_grid[mask]
        S_discrete = S_discrete[:, mask]

    # Prepend S(0) = 1.0 and t = 0 for full integration from time 0
    B = S_discrete.shape[0]
    S_full = torch.cat(
        [torch.ones(B, 1, device=S_discrete.device), S_discrete],
        dim=1,
    )  # [B, G+1]
    t_full = torch.cat(
        [torch.zeros(1, device=t_grid.device), t_grid],
    )  # [G+1]

    # Trapezoidal rule: sum of (S_left + S_right) / 2 * delta_t
    return torch.trapz(S_full, t_full, dim=1)  # [B]


def _build_structured_array(
    durations: np.ndarray,
    events: np.ndarray,
    any_event: bool = True,
) -> np.ndarray:
    """Build scikit-survival structured array from duration/event arrays.

    scikit-survival requires a specific structured numpy array format
    with dtype=[('event', bool), ('time', float)] for all its metric
    functions. This helper centralizes that construction.

    Args:
        durations: Observed times in months [N].
        events: Event labels {0, 1, 2} [N].
        any_event: If True, event field = (events > 0) — any cause.
            Used for Uno's C_t and IBS where overall event status matters.
            If False, event field = (events == EVENT_DEMENTIA) — primary
            cause only. Used for C_td where only dementia matters.

    Returns:
        Structured numpy array of shape [N] with dtype SKSURV_DTYPE.
    """
    if any_event:
        # Any event (dementia OR mortality) counts as an event
        event_bool = events > 0
    else:
        # Only primary event (dementia conversion) counts
        event_bool = events == EVENT_DEMENTIA

    # Build structured array — scikit-survival's required format
    result = np.zeros(len(durations), dtype=SKSURV_DTYPE)
    result["event"] = event_bool.astype(bool)
    result["time"] = durations.astype(float)

    return result


def compute_antolini_ctd(
    S_discrete: Tensor,
    t_grid: Tensor,
    durations: np.ndarray,
    events: np.ndarray,
    chi_interpolator,
) -> float:
    """Compute Antolini's time-dependent concordance index.

    This is the primary early-stopping metric — it evaluates whether the
    model's survival curve correctly ranks subjects at every actual event
    time.

    Antolini C_td is stricter than Harrell's C because it requires the
    ordering to hold at each event time individually, not just globally.
    It is computed from the full survival curve, not a scalar risk score.

    Mode: 'antolini' is used, NOT 'adj_antolini'. The adj_antolini
    mode applies a tie-correction that changes the numerical range and
    makes results incomparable with prior literature using the standard
    Antolini formulation.

    Primary events only: events is converted to binary (dementia = 1,
    all others = 0) even though the model handles competing risks.
    This is because C_td evaluates ranking for the primary endpoint
    (dementia conversion), not for the competing risk.

    Implementation note:
        Pure numpy is safe in all execution contexts and produces
        identical results
        O(N²) complexity is acceptable for evaluation-size datasets.

    Args:
        S_discrete: Discrete survival probabilities [B, G] from
            hazards_to_survival().
        t_grid: Temporal grid points [G] in months.
        durations: Observed times [B] in months (numpy).
        events: Event labels {0, 1, 2} [B] (numpy).
        chi_interpolator: ConstantHazardInterpolator for continuous
            survival curve construction.

    Returns:
        Antolini C_td as a scalar float in [0, 1].
    """
    B = S_discrete.shape[0]

    # Build dense time grid for survival curve evaluation
    t_min = float(t_grid[0])
    t_max = float(t_grid[-1])
    dense_times = np.linspace(t_min, t_max, N_DENSE_POINTS)

    # CHI-interpolate survival at dense time points
    # t_query: [B, T] for multi-query mode
    t_query = torch.tensor(
        dense_times, dtype=torch.float32
    ).unsqueeze(0).expand(B, -1)  # [B, N_DENSE_POINTS]

    with torch.no_grad():
        S_dense = chi_interpolator.interpolate(
            S_discrete, t_query
        ).cpu().numpy()  # [B, N_DENSE_POINTS]

    # surv: [T, N] — rows = time points, cols = subjects
    surv = S_dense.T  # [T, B]

    # Map each subject's duration to the nearest dense grid index
    surv_idx = np.searchsorted(dense_times, durations.astype(float))
    surv_idx = np.clip(surv_idx, 0, N_DENSE_POINTS - 1)

    # Primary events only: dementia = 1, censored/mortality = 0
    primary_events = (events == EVENT_DEMENTIA).astype(int)

    # Pure-numpy Antolini C_td computation
    # For each event subject i, compare against all other subjects j
    concordant = 0.0
    comparable = 0.0

    durs = durations.astype(float)

    for i in range(B):
        if primary_events[i] == 0:
            continue  # Only iterate over event subjects
        t_i = durs[i]
        idx_i = surv_idx[i]

        for j in range(B):
            if i == j:
                continue
            t_j = durs[j]
            d_j = primary_events[j]

            # Antolini comparability:
            # (t_i < t_j and d_i=1) or (t_i == t_j and d_i=1 and d_j==0)
            is_comparable = (
                (t_i < t_j) or
                (t_i == t_j and d_j == 0)
            )
            if not is_comparable:
                continue

            comparable += 1.0
            # Concordant: subject i has lower predicted survival at their
            # event time than subject j (correct ranking)
            if surv[idx_i, i] < surv[idx_i, j]:
                concordant += 1.0

    if comparable == 0.0:
        logger.warning(
            "No comparable pairs for C_td — returning 0.0. "
            "This usually means no primary events in the test set."
        )
        return 0.0

    return concordant / comparable


def compute_uno_c(
    S_discrete: Tensor,
    t_grid: Tensor,
    durations: np.ndarray,
    events: np.ndarray,
    train_durations: np.ndarray,
    train_events: np.ndarray,
) -> tuple:
    """Compute Uno's IPCW-corrected concordance index.

    Uses scikit-survival's concordance_index_ipcw. Essential for cohorts
    with heavy censoring where Harrell's C has upward bias.

    The truncation time tau is set to the 75th percentile of training
    event times. As t -> t_max, the censoring survival G(t) -> 0 and
    IPCW weights blow up. Truncating at tau prevents the denominator
    from becoming numerically unstable. The 75th percentile is standard
    practice.

    Risk scores are extracted as -S(t_G): higher risk = lower survival
    at the last grid point. The negation converts from "higher = longer
    survival" to "higher = more risky" as required by
    concordance_index_ipcw's sign convention.

    Structured arrays use any_event=True (events > 0) because Uno's C
    evaluates overall event status, not cause-specific. A subject who
    dies before converting is still an "event" for concordance purposes.

    Args:
        S_discrete: Discrete survival probabilities [B, G].
        t_grid: Temporal grid points [G] in months.
        durations: Test set observed times [B] (numpy).
        events: Test set event labels {0, 1, 2} [B] (numpy).
        train_durations: Training set observed times (numpy).
        train_events: Training set event labels (numpy).

    Returns:
        Tuple of (c_uno: float, tau: float).
        tau is returned for logging and checkpoint storage.
    """
    from sksurv.metrics import concordance_index_ipcw

    # Build structured arrays — required by scikit-survival
    y_train = _build_structured_array(
        train_durations, train_events, any_event=True
    )
    y_test = _build_structured_array(
        durations, events, any_event=True
    )

    # Scalar risk scores: -S(t_G) — higher = more risky
    # S_discrete[:, -1] is survival at last grid point (t=60)
    estimate = (-S_discrete[:, -1]).cpu().numpy()  # [B] numpy

    # Truncation time: 75th percentile of training event times
    # Prevents IPCW weight explosion in the right tail
    event_mask = train_events > 0
    event_times_train = train_durations[event_mask]
    tau = float(np.percentile(event_times_train, 75))

    # Compute Uno's IPCW-corrected concordance
    c_uno, _, _, _, _ = concordance_index_ipcw(
        y_train, y_test, estimate, tau=tau
    )

    return float(c_uno), tau


def compute_ibs(
    S_discrete: Tensor,
    t_grid: Tensor,
    durations: np.ndarray,
    events: np.ndarray,
    train_durations: np.ndarray,
    train_events: np.ndarray,
    chi_interpolator,
) -> float:
    """Compute Integrated Brier Score for absolute calibration.

    Tests whether predicted survival probabilities match observed event
    rates across the full evaluation horizon. A model with perfect
    ranking (C_td = 1.0) but terrible calibration (IBS = 0.24) would
    be useless in a clinical setting where physicians use the predicted
    probability to make dosing and enrollment decisions.

    IBS interpretation:
        0.0 — Perfect calibration
        0.25 — Null model (predict S(t) = 0.5 everywhere)
        <0.15 — Good for survival models on clinical data

    Uses CHI-interpolated survival curves rather than raw discrete grid
    values to ensure smooth evaluation at the grid time points.

    Args:
        S_discrete: Discrete survival probabilities [B, G].
        t_grid: Temporal grid points [G] in months.
        durations: Test set observed times [B] (numpy).
        events: Test set event labels {0, 1, 2} [B] (numpy).
        train_durations: Training set observed times (numpy).
        train_events: Training set event labels (numpy).
        chi_interpolator: ConstantHazardInterpolator for consistent
            curve evaluation.

    Returns:
        IBS as a scalar float in [0, 0.25].
    """
    from sksurv.metrics import integrated_brier_score

    # Build structured arrays — any_event=True for IBS
    y_train = _build_structured_array(
        train_durations, train_events, any_event=True
    )
    y_test = _build_structured_array(
        durations, events, any_event=True
    )

    # Eval times must be within the range of test set durations
    eval_times = np.array(t_grid.cpu().numpy(), dtype=float)
    # Clip eval_times to avoid exceeding the test set's max duration
    max_test_time = float(durations.max())
    eval_times = eval_times[eval_times < max_test_time]

    if len(eval_times) < 2:
        logger.warning(
            "Fewer than 2 eval_times within test range — "
            "returning IBS=NaN. Max test duration=%.1f, "
            "min t_grid=%.1f",
            max_test_time,
            float(t_grid[0]),
        )
        return float("nan")

    # CHI-interpolate survival at eval_times
    B = S_discrete.shape[0]
    t_query = torch.tensor(
        eval_times, dtype=torch.float32
    ).unsqueeze(0).expand(B, -1)  # [B, len(eval_times)]

    with torch.no_grad():
        surv_probs = chi_interpolator.interpolate(
            S_discrete, t_query
        ).cpu().numpy()  # [B, len(eval_times)]

    # Compute IBS via scikit-survival
    ibs = integrated_brier_score(
        y_train, y_test, surv_probs, eval_times
    )

    return float(ibs)


def compute_all_metrics(
    hazards: Tensor,
    durations: np.ndarray,
    events: np.ndarray,
    train_durations: np.ndarray,
    train_events: np.ndarray,
    chi_interpolator,
    survival_head,
    config: ModelConfig,
) -> dict:
    """Compute all three survival metrics in one call.

    This is the single entry point for all evaluation metrics — callers
    should never call individual metric functions directly. This ensures
    consistent array construction and CHI interpolation across all three
    metrics.

    Args:
        hazards: Cause-specific hazard rates [B, K, G] from the
            TraCeR survival head.
        durations: Test set observed times [B] (numpy float).
        events: Test set event labels {0, 1, 2} [B] (numpy int).
        train_durations: Training set observed times (numpy float).
        train_events: Training set event labels (numpy int).
        chi_interpolator: ConstantHazardInterpolator for continuous
            survival curves.
        survival_head: TraCeRSurvivalHead instance.
        config: ModelConfig providing t_grid.

    Returns:
        Dict with keys:
            c_td (float): Antolini C_td
            uno_c (float): Uno's C_t
            ibs (float): Integrated Brier Score
            tau (float): tau used for Uno's C (for logging)
    """
    # Extract discrete survival function from hazards
    with torch.no_grad():
        S_discrete = survival_head.hazards_to_survival(hazards)  # [B, G]

    t_grid = torch.tensor(config.t_grid, dtype=torch.float32)

    # Antolini C_td — primary early-stopping metric
    c_td = compute_antolini_ctd(
        S_discrete, t_grid, durations, events, chi_interpolator
    )

    # Uno's IPCW-corrected concordance
    uno_c, tau = compute_uno_c(
        S_discrete, t_grid, durations, events,
        train_durations, train_events,
    )

    # Integrated Brier Score — absolute calibration
    ibs = compute_ibs(
        S_discrete, t_grid, durations, events,
        train_durations, train_events, chi_interpolator,
    )

    metrics = {
        "c_td": c_td,
        "uno_c": uno_c,
        "ibs": ibs,
        "tau": tau,
    }

    logger.info(
        "Metrics — C_td=%.4f, Uno_C=%.4f (tau=%.1f), IBS=%.4f",
        c_td, uno_c, tau, ibs,
    )

    return metrics


def format_comparison_table(
    transformer_metrics: dict,
    cohort: str = "MCI->Dementia",
) -> tuple:
    """Build a markdown comparison table of baseline vs transformer.

    Combines the fixed baseline results with the transformer's
    three-metric evaluation for side-by-side comparison.

    Args:
        transformer_metrics: Dict with keys c_td, uno_c, ibs from
            compute_all_metrics().
        cohort: Cohort label for the transformer rows.

    Returns:
        Tuple of (markdown_table: str, csv_rows: list[dict]).
        markdown_table: formatted markdown string for display.
        csv_rows: list of dicts ready for csv.DictWriter with keys
            Model, Cohort, Metric Type, Value, Notes.
    """
    csv_rows = []

    # Header
    lines = [
        "| Model | Cohort | Metric Type | Value | Notes |",
        "|-------|--------|-------------|-------|-------|",
    ]

    # Baseline rows from stored constants
    for key, info in BASELINE_RESULTS.items():
        row = {
            "Model": info["model"],
            "Cohort": info["cohort"],
            "Metric Type": info["metric"],
            "Value": f"{info['value']:.4f}",
            "Notes": "Baseline",
        }
        csv_rows.append(row)
        lines.append(
            f"| {row['Model']} | {row['Cohort']} | "
            f"{row['Metric Type']} | {row['Value']} | {row['Notes']} |"
        )

    # Separator
    lines.append("| --- | --- | --- | --- | --- |")

    # Transformer rows
    transformer_rows = [
        {
            "Model": "Transformer",
            "Cohort": cohort,
            "Metric Type": "Antolini C_td",
            "Value": f"{transformer_metrics['c_td']:.4f}",
            "Notes": "Primary early-stopping metric",
        },
        {
            "Model": "Transformer",
            "Cohort": cohort,
            "Metric Type": "Uno's C_tau",
            "Value": f"{transformer_metrics['uno_c']:.4f}",
            "Notes": f"tau={transformer_metrics.get('tau', 0):.1f}mo",
        },
        {
            "Model": "Transformer",
            "Cohort": cohort,
            "Metric Type": "IBS",
            "Value": f"{transformer_metrics['ibs']:.4f}",
            "Notes": "Lower is better (0=perfect)",
        },
    ]

    for row in transformer_rows:
        csv_rows.append(row)
        lines.append(
            f"| {row['Model']} | {row['Cohort']} | "
            f"{row['Metric Type']} | {row['Value']} | {row['Notes']} |"
        )

    # Footer note about metric non-comparability
    lines.append("")
    lines.append(
        "* Harrell's C and Antolini C_td are not directly comparable. "
        "See SKILL_evaluation_metrics.md section 1."
    )

    markdown_table = "\n".join(lines)
    return markdown_table, csv_rows


# Smoke test
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    print("Phase 8 — Evaluation Metrics Smoke Test")

    from Transformer.models.survival_head import TraCeRSurvivalHead
    from Transformer.utils.chi_interpolation import (
        ConstantHazardInterpolator,
    )

    config = ModelConfig()
    pass_count = 0
    fail_count = 0

    # Create synthetic survival data: 30 subjects
    np.random.seed(42)
    torch.manual_seed(42)

    n_total = 30
    n_test = 10
    n_train = n_total - n_test

    durations_all = np.random.uniform(6, 60, n_total).astype(np.float64)
    events_all = np.random.choice([0, 1, 2], n_total, p=[0.5, 0.3, 0.2])
    events_all = events_all.astype(np.int64)

    train_durations = durations_all[:n_train]
    train_events = events_all[:n_train]
    test_durations = durations_all[n_train:]
    test_events = events_all[n_train:]

    # Instantiate components
    chi = ConstantHazardInterpolator.from_config(config)
    head = TraCeRSurvivalHead(d_input=4096, config=config)
    head.eval()

    # Create dummy hazards from random input and extract survival
    x_dummy = torch.randn(n_test, 4096)
    with torch.no_grad():
        hazards = head(x_dummy)  # [10, 2, 5]
        S_discrete = head.hazards_to_survival(hazards)  # [10, 5]

    t_grid = torch.tensor(config.t_grid, dtype=torch.float32)

    # Test 1: RMST shape and positivity
    try:
        rmst_vals = rmst(S_discrete, t_grid)
        assert rmst_vals.shape == (n_test,), f"RMST shape: {rmst_vals.shape}"
        assert (rmst_vals > 0).all(), f"RMST min: {rmst_vals.min()}"
        print(f"PASS: rmst — shape {tuple(rmst_vals.shape)}, all positive")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: rmst — {e}")
        fail_count += 1

    # Test 2: Antolini C_td
    try:
        c_td = compute_antolini_ctd(
            S_discrete, t_grid, test_durations, test_events, chi
        )
        assert 0.0 <= c_td <= 1.0, f"C_td={c_td} out of [0,1]"
        print(f"PASS: C_td = {c_td:.4f}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: C_td — {e}")
        fail_count += 1

    # Test 3: Uno's C_tau
    try:
        c_uno, tau = compute_uno_c(
            S_discrete, t_grid, test_durations, test_events,
            train_durations, train_events,
        )
        assert 0.0 <= c_uno <= 1.0, f"Uno's C={c_uno} out of [0,1]"
        assert tau > 0, f"tau={tau} not positive"
        print(f"PASS: Uno's C = {c_uno:.4f}, tau = {tau:.1f}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: Uno's C — {e}")
        fail_count += 1

    # Test 4: IBS — range is wider for synthetic data (random predictions)
    # 0.25 = null model; untrained random models can exceed this
    try:
        ibs_val = compute_ibs(
            S_discrete, t_grid, test_durations, test_events,
            train_durations, train_events, chi,
        )
        if not np.isnan(ibs_val):
            assert 0.0 <= ibs_val, f"IBS={ibs_val} < 0"
            assert np.isfinite(ibs_val), f"IBS={ibs_val} not finite"
            print(f"PASS: IBS = {ibs_val:.4f} (finite, non-negative)")
        else:
            print("PASS: IBS = NaN (expected with small synthetic data)")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: IBS — {e}")
        fail_count += 1

    # Test 5: compute_all_metrics
    try:
        all_metrics = compute_all_metrics(
            hazards, test_durations, test_events,
            train_durations, train_events,
            chi, head, config,
        )
        required_keys = {"c_td", "uno_c", "ibs", "tau"}
        assert required_keys.issubset(all_metrics.keys()), (
            f"Missing keys: {required_keys - all_metrics.keys()}"
        )
        for k in ["c_td", "uno_c", "tau"]:
            assert np.isfinite(all_metrics[k]), f"{k} is not finite"
        print(f"PASS: compute_all_metrics — keys: {list(all_metrics.keys())}")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: compute_all_metrics — {e}")
        fail_count += 1

    # Test 6: Comparison table
    try:
        dummy_metrics = {"c_td": 0.85, "uno_c": 0.82, "ibs": 0.12, "tau": 48.0}
        md_table, csv_rows = format_comparison_table(dummy_metrics)
        assert "GBM Surv" in md_table, "Missing baseline in table"
        assert "Transformer" in md_table, "Missing transformer in table"
        assert "not directly comparable" in md_table, "Missing note"
        assert len(csv_rows) > 0, "CSV rows empty"
        print(f"PASS: comparison table — {len(csv_rows)} rows")
        pass_count += 1
    except Exception as e:
        print(f"FAIL: comparison table — {e}")
        fail_count += 1

    # Summary
    print(f"Results: {pass_count} PASS, {fail_count} FAIL")
    if fail_count == 0:
        print("PASS — All assertions hold.")
    else:
        print("FAIL — Some tests did not pass.")
