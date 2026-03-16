from lifelines import KaplanMeierFitter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
from modeling import binary_horizon_dataset
from config import RANDOM_SEED, N_FOLDS, HORIZONS, FIG_DIR, CHECKPOINT_DIR, OUT_DIR, MRI_HARMONIZE_COLS, BASE_DIR


def calibration_plot(X_imp, y_event, y_duration, predict_proba_fn,
                      horizon, model_name, cohort, n_bins=10):
    '''
    Decile calibration plot: predicted P(event ≤ horizon) vs. observed rate.
    predict_proba_fn(X) -> P(event <= horizon) for each subject
    '''
    y_bin, include = binary_horizon_dataset(y_event, y_duration, horizon)
    X_h = X_imp.iloc[include]
    probs = predict_proba_fn(X_h)

    # Sort by predicted probability and bin into deciles
    order = np.argsort(probs)
    bin_size = len(probs) // n_bins
    pred_means, obs_means, ci_lower, ci_upper = [], [], [], []

    for i in range(n_bins):
        idx = order[i*bin_size:(i+1)*bin_size]
        pred_means.append(probs[idx].mean())
        obs_means.append(y_bin[idx].mean())
        n, p = len(idx), y_bin[idx].mean()
        se = np.sqrt(p*(1-p)/max(n,1))
        ci_lower.append(p - 1.96*se)
        ci_upper.append(p + 1.96*se)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0,1],[0,1],'k--', lw=1, label='Perfect calibration')
    ax.errorbar(pred_means, obs_means,
                yerr=[np.array(obs_means)-np.array(ci_lower),
                      np.array(ci_upper)-np.array(obs_means)],
                fmt='o', color='#e74c3c', capsize=4, label=model_name)
    ax.set(xlabel=f'Predicted P(event ≤ {horizon}yr)',
           ylabel='Observed event rate',
           title=f'Calibration: {model_name} [{cohort}] {horizon}yr')
    ax.legend(); plt.tight_layout()
    cohort_clean = cohort.replace('>','')
    fname = FIG_DIR / f'calibration_{model_name.replace(" ","_")}_{cohort_clean}_{horizon}yr.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'  Saved calibration plot: {fname.name}')


def lgb_horizon_proba(X, horizon, y_ev, y_dur, X_train, y_ev_tr, y_dur_tr):
    """
    Train a LightGBM binary classifier for fixed-horizon survival prediction
    and return predicted event probabilities on a test set.

    Uses binary_horizon_dataset to construct labels, excluding censored
    subjects with follow-up shorter than the horizon. Class imbalance is
    addressed via scale_pos_weight.

    Args:
        X (pd.DataFrame): Test feature matrix to generate predictions for.
        horizon (int or float): Prediction horizon in years (e.g. 3 or 5).
        y_ev (np.ndarray): Event indicators for the test set.
        y_dur (np.ndarray): Durations for the test set in years.
        X_train (pd.DataFrame): Training feature matrix.
        y_ev_tr (np.ndarray): Event indicators for the training set.
        y_dur_tr (np.ndarray): Durations for the training set in years.

    Returns:
        np.ndarray: Predicted probabilities of event within `horizon` years,
            for the subset of test subjects with determinable outcomes.
    """
    y_bin_tr, include_tr = binary_horizon_dataset(y_ev_tr, y_dur_tr, horizon)
    X_tr_h = X_train.iloc[include_tr]
    scale_pos = (y_bin_tr==0).sum() / max((y_bin_tr==1).sum(), 1)
    params = dict(objective='binary', metric='auc', n_estimators=300,
                  learning_rate=0.05, num_leaves=31, verbose=-1,
                  scale_pos_weight=scale_pos, random_state=RANDOM_SEED)
    m = lgb.LGBMClassifier(**params)
    m.fit(X_tr_h, y_bin_tr)
    y_bin_te, include_te = binary_horizon_dataset(y_ev, y_dur, horizon)
    return m.predict_proba(X.iloc[include_te])[:,1]

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
