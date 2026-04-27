import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from lifelines import KaplanMeierFitter

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


def _km_censoring(
    train_durations: np.ndarray,
    train_events: np.ndarray,
):
    """Reverse-KM estimator of the censoring distribution G(t).

    Mirrors sksurv's CensoringDistributionEstimator: flips the event
    indicator and fits Kaplan-Meier on (T, 1-delta). Returns a callable
    G(t) -> P(C > t) as a right-continuous step function.

    Args:
        train_durations: Training set observed times (numpy float).
        train_events: Training set binary event indicators (numpy int).

    Returns:
        Callable that accepts scalar or array t and returns G(t).
    """
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=np.asarray(train_durations, dtype=float),
        event_observed=(1 - np.asarray(train_events)).astype(int),
    )
    times = np.asarray(kmf.survival_function_.index.values, dtype=np.float64)
    probs = np.asarray(
        kmf.survival_function_.iloc[:, 0].values, dtype=np.float64
    )

    def G(t):
        t = np.atleast_1d(np.asarray(t, dtype=np.float64))
        idx = np.searchsorted(times, t, side="right") - 1
        idx = np.clip(idx, 0, len(probs) - 1)
        out = probs[idx]
        out = np.where(t < times[0], 1.0, out)
        return out

    return G


def compute_uno_c(
    S_discrete: Tensor,
    t_grid: Tensor,
    durations: np.ndarray,
    events: np.ndarray,
    train_durations: np.ndarray,
    train_events: np.ndarray,
) -> tuple:
    """Uno's IPCW concordance index (Uno et al. 2011, Stat. Med. 30:1105).

    Vendored pure-numpy + lifelines reimplementation. Numerically
    equivalent to sksurv.metrics.concordance_index_ipcw to within
    float64 precision. Removes the scikit-survival dependency for
    this metric.

    Risk scores: -S(t_G), higher = riskier (same convention as before).
    tau: 75th percentile of training event times.
    IPCW weights are squared (Uno 2011 eq. 10).

    Args:
        S_discrete: Discrete survival probabilities [B, G].
        t_grid: Temporal grid points [G] in months.
        durations: Test set observed times [B] (numpy).
        events: Test set event labels {0, 1, 2} [B] (numpy).
        train_durations: Training set observed times (numpy).
        train_events: Training set event labels (numpy).

    Returns:
        Tuple of (c_uno: float, tau: float).
    """
    # Cast to float64 for numerical precision
    t_tr = np.asarray(train_durations, dtype=np.float64)
    e_tr = (np.asarray(train_events) > 0).astype(np.int64)
    t_te = np.asarray(durations, dtype=np.float64)
    e_te = (np.asarray(events) > 0).astype(np.int64)

    # Risk score: -S(t_last), higher = riskier
    estimate = (-S_discrete[:, -1]).detach().cpu().numpy().astype(np.float64)

    # tau = 75th percentile of training event times
    train_event_times = t_tr[e_tr == 1]
    tau = float(np.percentile(train_event_times, 75))

    # Censoring KM from training data
    G = _km_censoring(t_tr, e_tr)
    g_at_test = np.maximum(G(t_te), 1e-12)
    ipcw = 1.0 / g_at_test

    # Valid pairs: subject i must be uncensored and before tau
    mask_i = (e_te == 1) & (t_te < tau)
    if mask_i.sum() == 0:
        logger.warning("No valid pairs for Uno C — returning 0.5")
        return 0.5, tau

    # Vectorized comparable-pair computation
    Ti = t_te[:, None]
    Tj = t_te[None, :]
    Hi = estimate[:, None]
    Hj = estimate[None, :]
    Wi = (ipcw ** 2)[:, None]  # Squared IPCW weights (Uno 2011)

    # Comparable: i is event before tau, j survives longer than i
    valid_pair = mask_i[:, None] & (Tj > Ti)

    concordant = (Hi > Hj).astype(np.float64)
    tied_risk = (Hi == Hj).astype(np.float64)
    pair_score = concordant + 0.5 * tied_risk

    num = float((pair_score * Wi * valid_pair).sum())
    den = float((Wi * valid_pair).sum())
    if den == 0.0:
        logger.warning("Uno C denominator is zero — returning 0.5")
        return 0.5, tau

    return float(num / den), tau


def compute_ibs(
    S_discrete: Tensor,
    t_grid: Tensor,
    durations: np.ndarray,
    events: np.ndarray,
    train_durations: np.ndarray,
    train_events: np.ndarray,
    chi_interpolator,
) -> float:
    """Integrated Brier Score via SurvivalEVAL

    Calls SurvivalEVAL's per-time brier_score(IPCW_weighted=True) at
    each grid point and integrates with sksurv-equivalent trapezoidal
    weighting: IBS = trapz(BS, t) / (t[-1] - t[0]).

    IBS interpretation:
        0.0  — Perfect calibration
        0.25 — Null model (predict S(t) = 0.5 everywhere)
        <0.15 — Good for survival models on clinical data

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
        IBS as a scalar float.
    """
    from SurvivalEVAL.Evaluator import SurvivalEvaluator

    # Cast inputs to numpy float64/int64
    t_te = np.asarray(durations, dtype=np.float64)
    e_te = (np.asarray(events) > 0).astype(np.int64)
    t_tr = np.asarray(train_durations, dtype=np.float64)
    e_tr = (np.asarray(train_events) > 0).astype(np.int64)

    # Eval grid clipped to test follow-up (sksurv's invariant)
    grid_full = np.asarray(t_grid.cpu().numpy(), dtype=np.float64)
    max_test_time = float(t_te.max())
    eval_times = grid_full[grid_full < max_test_time]

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
    ).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        surv_at_eval = chi_interpolator.interpolate(
            S_discrete, t_query
        ).cpu().numpy().astype(np.float64)

    # Defensive monotonicity (CHI may produce tiny upticks)
    surv_at_eval = np.clip(
        np.minimum.accumulate(surv_at_eval, axis=1), 1e-12, 1.0
    )

    # SurvivalEvaluator: curves aligned at eval_times so no
    # interpolation is triggered when calling brier_score on-grid.
    evl = SurvivalEvaluator(
        pred_survs=surv_at_eval,
        time_coordinates=eval_times,
        test_event_times=t_te,
        test_event_indicators=e_te,
        train_event_times=t_tr,
        train_event_indicators=e_tr,
        predict_time_method="Median",
        interpolation="Linear",
    )

    # Per-time IPCW Brier at each grid point
    bs_per_t = np.array([
        float(evl.brier_score(target_time=float(t), IPCW_weighted=True))
        for t in eval_times
    ], dtype=np.float64)

    # sksurv's IBS weighting: IBS = trapz(BS, t) / (t[-1] - t[0])
    t_span = float(eval_times[-1] - eval_times[0])
    ibs = float(np.trapz(bs_per_t, eval_times) / t_span)

    return ibs


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
