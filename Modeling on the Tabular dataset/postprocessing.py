from lifelines import KaplanMeierFitter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from modeling import binary_horizon_dataset
from config import RANDOM_SEED, N_FOLDS, HORIZONS, FIG_DIR, CHECKPOINT_DIR, OUT_DIR, MRI_HARMONIZE_COLS, BASE_DIR
import itertools
from typing import List, Optional

def km_risk_quartile(risk_scores, y_event, y_duration, model_name, cohort):
    """
    Plot Kaplan-Meier survival curves stratified by predicted risk quartile.

    Subjects are divided into four equal-sized quartiles (Q1=lowest risk,
    Q4=highest risk) based on model risk scores. A well-discriminating model
    produces widely separated KM curves between Q1 and Q4.

    Quartiles with fewer than 5 subjects are skipped to avoid unstable
    KM estimates.

    Args:
        risk_scores (np.ndarray): Continuous risk scores output by a model,
            where higher values indicate higher predicted risk.
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring in years.
        model_name (str): Model label used in the plot title and output filename
            (spaces replaced with underscores in filename).
        cohort (str): Cohort label e.g. 'MCI->Dementia', used in title and filename.

    Returns:
        None. Saves figure to FIG_DIR as
        'km_quartile_<model_name>_<cohort>.png'.
    """
    quartile = pd.qcut(risk_scores, 4, labels=['Q1 (low)','Q2','Q3','Q4 (high)'])
    colors   = ['#2ecc71','#f1c40f','#e67e22','#e74c3c']
    fig, ax  = plt.subplots(figsize=(9, 6))
    kmf = KaplanMeierFitter()
    for q, col in zip(['Q1 (low)','Q2','Q3','Q4 (high)'], colors):
        mask = quartile == q
        if mask.sum() < 5: continue
        kmf.fit(y_duration[mask], event_observed=y_event[mask], label=q)
        kmf.plot_survival_function(ax=ax, color=col, ci_show=True, ci_alpha=0.15)
    ax.set(xlabel='Years from Baseline', ylabel='P(No Event)',
           title=f'KM by Risk Quartile: {model_name} [{cohort}]', ylim=(0,1))
    ax.axhline(0.5, color='gray', ls=':', alpha=0.6)
    plt.tight_layout()
    cohort_clean = cohort.replace('>','')
    fname = FIG_DIR / f'km_quartile_{model_name.replace(" ","_")}_{cohort_clean}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()


def build_subject_time_matrix(df_all, rids, time_grid, features, window=0.25):
    '''
    Returns:
      tensor : (n_subjects, n_timepoints, n_features)  float32
      mask   : (n_subjects, n_timepoints)  bool — True where real observation exists
    '''
    n_subj = len(rids)
    n_time = len(time_grid)
    n_feat = len(features)
    tensor = np.full((n_subj, n_time, n_feat), np.nan, dtype=np.float32)
    mask   = np.zeros((n_subj, n_time), dtype=bool)

    for s_idx, rid in enumerate(rids):
        subj = df_all[df_all['RID'] == rid].sort_values('Years_bl')
        for t_idx, t in enumerate(time_grid):
            # Find nearest visit within ±window
            diffs = np.abs(subj['Years_bl'].values - t)
            if diffs.min() <= window:
                nearest = subj.iloc[diffs.argmin()]
                for f_idx, feat in enumerate(features):
                    val = nearest[feat]
                    if not pd.isna(val):
                        tensor[s_idx, t_idx, f_idx] = val
                mask[s_idx, t_idx] = (diffs.min() <= window)

    # Forward-fill within each subject to handle remaining NaNs
    for s_idx in range(n_subj):
        for f_idx in range(n_feat):
            arr = tensor[s_idx, :, f_idx]
            # ffill
            last = np.nan
            for t_idx in range(n_time):
                if not np.isnan(arr[t_idx]):
                    last = arr[t_idx]
                elif not np.isnan(last):
                    tensor[s_idx, t_idx, f_idx] = last
        # Fill remaining NaN with feature median across subjects
    for f_idx in range(n_feat):
        feat_median = np.nanmedian(tensor[:, :, f_idx])
        nan_mask = np.isnan(tensor[:, :, f_idx])
        tensor[:, :, f_idx][nan_mask] = feat_median

    return tensor, mask

def plot_individual_survival_curves(
    curves: List[pd.Series],
    duration: float,
    event: int,
    labels: Optional[List[str]] = None,
    title: str = "Survival Curves"
) -> None:
    """
    Plot multiple survival curves with a duration marker and event/censoring indicator.

    Parameters
    ----------
    curves : List[pd.Series]
        List of survival probabilities for each curve, indexed by time.
    duration : float
        Scalar time point to mark on the plot.
    event : int
        0 = censored exit (open circle marker)
        1 = event exit (diamond marker)
    labels : Optional[List[str]]
        Display names for the curves. Defaults to "Curve 1", "Curve 2", etc.
    title : str
        Plot title.

    Returns
    -------
    fig : plt.Figure
    """
    if event not in (0, 1):
        raise ValueError("`event` must be 0 (censored) or 1 (event).")

    # Handle dynamic labels
    if labels is None:
        labels = [f"Curve {i+1}" for i in range(len(curves))]
    elif len(labels) != len(curves):
        raise ValueError("The number of labels must match the number of curves.")

    # Colors and Styling
    GRAY = "#888780"
    RED  = "#E24B4A"
    
    # Pre-defined palette covering original colors and extending for more curves
    PALETTE = ["#378ADD", "#D85A30", "#3B6D11", "#9B59B6", "#1ABC9C", "#F39C12"]
    color_cycler = itertools.cycle(PALETTE)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#FAFAF9")
    ax.set_facecolor("#FAFAF9")

    # --- draw step curves ---
    curve_colors = []
    for series, label in zip(curves, labels):
        color = next(color_cycler)
        curve_colors.append(color)
        ax.step(series.index, series.values, where="post",
                color=color, linewidth=2.2, label=label)

    # --- vertical duration line ---
    ax.axvline(duration, color=GRAY, linewidth=1.2,
               linestyle="--", alpha=0.8, label=f"t = {duration}")

    # --- interpolate S(duration) for each curve ---
    def interp(series: pd.Series, t: float) -> float:
        times = series.index.to_numpy(dtype=float)
        probs = series.values.astype(float)
        if t <= times[0]:
            return float(probs[0])
        if t >= times[-1]:
            return float(probs[-1])
        return float(np.interp(t, times, probs))

    # --- plot markers ---
    event_label = "Event" if event == 1 else "Censored"
    marker = "D" if event == 1 else "o"
    mfc = RED if event == 1 else "none"  # Marker face color

    for series, label, color in zip(curves, labels, curve_colors):
        surv_val = interp(series, duration)
        
        ax.plot(duration, surv_val, marker=marker, markersize=10,
                markerfacecolor=mfc, markeredgecolor=color,
                markeredgewidth=2, zorder=5,
                label=f"{label} S({duration}) = {surv_val:.3f}  [{event_label}]")

    # --- labels & styling ---
    ax.set_xlabel("Time", fontsize=12, color=GRAY)
    ax.set_ylabel("Survival probability S(t)", fontsize=12, color=GRAY)
    ax.set_title(title, fontsize=14, fontweight="normal", pad=14)
    ax.set_ylim(-0.02, 1.08)
    ax.tick_params(colors=GRAY, labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_edgecolor("#D3D1C7")

    ax.legend(frameon=True, framealpha=0.9, fontsize=8,
              edgecolor="#D3D1C7",
              bbox_to_anchor=(0.5, -0.18), loc='upper center',
              borderaxespad=0, ncol=3)
    fig.tight_layout(pad=3.0)
    plt.show()
