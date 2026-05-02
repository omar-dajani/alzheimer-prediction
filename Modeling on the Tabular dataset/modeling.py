"""
modeling.py — survival model training, evaluation, and bootstrap utilities
for the ADNI tabular survival analysis pipeline.

Four models are supported:
  1. Cox PH           (lifelines CoxPHFitter, regularised with elastic-net)
  2. GBSA             (scikit-survival GradientBoostingSurvivalAnalysis)
  3. DeepSurv         (pycox neural Cox PH via torchtuples MLP)
  4. Weighted Ensemble (Optuna-optimised blend of survival curves from 1–3)

All training functions return both an OOF (out-of-fold) C-index and an
in-sample C-index so the caller can inspect the train-vs-validation gap as
an overfitting diagnostic.  Final evaluation should always be done on the
held-out test set returned by ``make_holdout_split``.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import optuna
from tqdm.notebook import tqdm
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from lifelines.utils import concordance_index
import lightgbm as lgb  # used only for CSF imputation, not survival modelling

# DeepSurv imports
import torch
import torchtuples as tt
from pycox.models import CoxPH as PycoxCoxPH
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
from concordance import concordance_td

# Cox PH imports
from lifelines import CoxPHFitter

# Ensemble meta-learner
from sklearn.linear_model import RidgeCV

from config import (
    RANDOM_SEED, N_FOLDS, HORIZONS,
    FIG_DIR, CHECKPOINT_DIR, OUT_DIR, MRI_HARMONIZE_COLS, BASE_DIR,
)

# ── GPU detection ─────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb_test
    lgb_test.LGBMRegressor(device='gpu', verbose=-1, n_estimators=1).fit([[1]], [1])
    LGB_DEVICE = 'gpu'
except Exception:
    LGB_DEVICE = 'cpu'
print(f'LightGBM device: {LGB_DEVICE}')


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(name, obj):
    """
    Serialise and save a Python object to disk as a ``.pkl`` file.

    Checkpoints allow expensive training runs to be skipped on re-runs by
    setting ``RETRAIN = False`` in the notebook.  Any pickle-serialisable
    object can be stored (fitted models, result dicts, numpy arrays, etc.).

    Args:
        name (str): Checkpoint identifier used as the file stem.
            For example, ``'gbsa_mci'`` saves to ``CHECKPOINT_DIR/gbsa_mci.pkl``.
        obj (any): Any pickle-serialisable Python object.

    Returns:
        None.  Prints the saved path on success.

    Raises:
        OSError: If ``CHECKPOINT_DIR`` does not exist or is not writable.
    """
    path = CHECKPOINT_DIR / f'{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  Checkpointed: {name} -> {path}')


def load_checkpoint(name):
    """
    Load a previously saved ``.pkl`` checkpoint from ``CHECKPOINT_DIR``.

    Returns ``None`` silently if the file does not exist, allowing the caller
    to fall back to retraining without an explicit existence check.

    Args:
        name (str): Checkpoint identifier matching the stem passed to
            ``save_checkpoint`` (e.g. ``'gbsa_mci'``).

    Returns:
        any: The deserialised Python object, or ``None`` if the checkpoint
            file does not exist.
    """
    path = CHECKPOINT_DIR / f'{name}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f'  Loaded checkpoint: {name}')
        return obj
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Holdout split
# ═══════════════════════════════════════════════════════════════════════════════

def make_holdout_split(X, y_event, y_duration, test_size=0.20, seed=RANDOM_SEED):
    """
    Create a stratified holdout split that is kept completely separate from all
    model training, hyperparameter optimisation, and cross-validation.

    Stratification is performed on a joint label combining event status and
    duration quartile so the holdout set mirrors the full dataset's event rate
    *and* follow-up time distribution.  This prevents pathological splits where
    all late events land in training and all early events land in test.

    The returned ``idx_dev`` and ``idx_test`` are **positional** indices into the
    original (un-reset) ``X`` DataFrame, useful for tracing subjects back to
    ``df_baseline`` or ``surv_labels``.

    Args:
        X (pd.DataFrame): Fully imputed feature matrix — no NaN values allowed.
            Shape ``(n_subjects, n_features)``.
        y_event (np.ndarray): Binary event indicators, shape ``(n_subjects,)``.
            ``1`` = subject experienced the event; ``0`` = censored.
        y_duration (np.ndarray): Time to event or censoring in years,
            shape ``(n_subjects,)``.  Must be positive.
        test_size (float): Fraction of subjects to hold out.  Default ``0.20``
            (20 % test, 80 % development).
        seed (int): Random seed for reproducibility.  Default ``RANDOM_SEED``.

    Returns:
        tuple: Eight elements in the following order:

            - **X_dev** (pd.DataFrame): Development feature matrix
              (index reset to 0..n_dev-1).
            - **X_test** (pd.DataFrame): Holdout test feature matrix
              (index reset to 0..n_test-1).
            - **y_ev_dev** (np.ndarray): Event indicators for the dev set.
            - **y_ev_test** (np.ndarray): Event indicators for the test set.
            - **y_dur_dev** (np.ndarray): Durations for the dev set.
            - **y_dur_test** (np.ndarray): Durations for the test set.
            - **idx_dev** (np.ndarray[int]): Positional indices into the
              original ``X`` for the development subjects.
            - **idx_test** (np.ndarray[int]): Positional indices into the
              original ``X`` for the test subjects.
    """
    from sklearn.model_selection import train_test_split

    dur_q = pd.qcut(y_duration, q=4, labels=False, duplicates='drop')
    strat = y_event * 10 + dur_q  # unique per (event, quartile) cell

    idx = np.arange(len(X))
    idx_dev, idx_test = train_test_split(
        idx, test_size=test_size, stratify=strat, random_state=seed
    )

    return (
        X.iloc[idx_dev].reset_index(drop=True),
        X.iloc[idx_test].reset_index(drop=True),
        y_event[idx_dev],
        y_event[idx_test],
        y_duration[idx_dev],
        y_duration[idx_test],
        idx_dev,
        idx_test,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_cindex_harrell(y_event, y_duration, risk_scores,
                              n_bootstrap=1000, seed=RANDOM_SEED, alpha=0.05):
    """
    Compute a non-parametric bootstrap 95 % confidence interval for
    Harrell's concordance index (C-index).

    The bootstrap resamples subjects *with replacement* ``n_bootstrap`` times
    and evaluates Harrell's C on each resample using the **pre-computed risk
    scores** — no model refit is performed.  This approach quantifies sampling
    variability in the concordance estimate without the computational cost of
    repeated training.

    Harrell's C is the probability that a randomly chosen pair of subjects is
    ranked correctly (subject with the shorter survival time receives a higher
    risk score).  It equals 0.5 for a random model and 1.0 for a perfect model.

    Args:
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
            ``1`` = event observed; ``0`` = censored.
        y_duration (np.ndarray): Time to event or censoring in years,
            shape ``(n,)``.
        risk_scores (np.ndarray): Continuous model-predicted risk scores,
            shape ``(n,)``.  Higher values must indicate higher predicted risk
            (i.e. shorter expected survival).
        n_bootstrap (int): Number of bootstrap resamples.  Default ``1000``.
            Increase to 5000 for publication-quality CIs.
        seed (int): Random seed for reproducibility.  Default ``RANDOM_SEED``.
        alpha (float): Two-tailed significance level.  Default ``0.05``
            produces a 95 % CI.

    Returns:
        dict: Four keys:

            - ``'point'``  (float): C-index on the original (unperturbed) data.
            - ``'mean'``   (float): Mean of the bootstrap distribution.
            - ``'lower'``  (float): Lower ``alpha/2`` percentile CI bound.
            - ``'upper'``  (float): Upper ``1-alpha/2`` percentile CI bound.
            - ``'boot'``   (np.ndarray): All ``n_bootstrap`` C-index values —
              useful for plotting the full distribution.
    """
    rng = np.random.RandomState(seed)
    n = len(y_event)
    point = concordance_index(y_duration, -risk_scores, y_event)

    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if y_event[idx].sum() == 0:
            continue
        c = concordance_index(y_duration[idx], -risk_scores[idx], y_event[idx])
        boot.append(c)

    boot = np.array(boot)
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return {'point': point, 'mean': boot.mean(), 'lower': lo, 'upper': hi, 'boot': boot}


def bootstrap_cindex_td(y_event, y_duration, surv_df,
                         n_bootstrap=1000, seed=RANDOM_SEED, alpha=0.05):
    """
    Compute a non-parametric bootstrap 95 % confidence interval for the
    IPCW-weighted Antolini time-dependent concordance index (C-td).

    C-td extends Harrell's C to handle time-dependent survival predictions
    by comparing ``S(t_i | x_i)`` against ``S(t_i | x_j)`` for all concordant
    pairs ``(i, j)``.  The IPCW (Inverse Probability of Censoring Weighting)
    adjustment down-weights heavily censored pairs to reduce bias.

    The bootstrap resamples subjects with replacement and recomputes C-td on
    each resample using the **pre-computed survival matrix** — no model refit.

    Args:
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.
        surv_df (pd.DataFrame): Predicted survival matrix with shape
            ``(n_timepoints, n_subjects)``.  Index values are time points in
            years; columns correspond to subjects in the same order as
            ``y_event`` and ``y_duration``.
        n_bootstrap (int): Number of bootstrap resamples.  Default ``1000``.
        seed (int): Random seed.  Default ``RANDOM_SEED``.
        alpha (float): Two-tailed significance level.  Default ``0.05``.

    Returns:
        dict: Same structure as ``bootstrap_cindex_harrell``:
            ``'point'``, ``'mean'``, ``'lower'``, ``'upper'``, ``'boot'``.

    Raises:
        TypeError: If ``surv_df`` is not a ``pd.DataFrame``.
    """
    if isinstance(surv_df, pd.DataFrame):
        surv_arr = surv_df.values.astype(np.float64)
        time_grid = surv_df.index.to_numpy(dtype=np.float64)
    else:
        raise TypeError('surv_df must be a pd.DataFrame')

    def _c_td(idx):
        dur_b = y_duration[idx]
        ev_b  = y_event[idx]
        surv_b = surv_arr[:, idx]
        s_idx = np.clip(
            np.searchsorted(time_grid, dur_b, side='right') - 1,
            0, len(time_grid) - 1
        ).astype(np.int64)
        return concordance_td(
            dur_b.astype(np.float64), ev_b.astype(np.int32),
            surv_b, s_idx, method='adj_antolini', ipcw=True
        )

    n = len(y_event)
    point_s_idx = np.clip(
        np.searchsorted(time_grid, y_duration, side='right') - 1,
        0, len(time_grid) - 1
    ).astype(np.int64)
    point = concordance_td(
        y_duration.astype(np.float64), y_event.astype(np.int32),
        surv_arr, point_s_idx, method='adj_antolini', ipcw=True
    )

    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if y_event[idx].sum() < 3:
            continue
        try:
            boot.append(_c_td(idx))
        except Exception:
            pass

    boot = np.array(boot)
    lo = np.percentile(boot, 100 * alpha / 2)
    hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return {'point': point, 'mean': boot.mean(), 'lower': lo, 'upper': hi, 'boot': boot}


# ═══════════════════════════════════════════════════════════════════════════════
# Permutation importance
# ═══════════════════════════════════════════════════════════════════════════════

def permutation_importance_survival(predict_fn, X, y_event, y_duration,
                                     n_repeats=20, seed=RANDOM_SEED):
    """
    Model-agnostic permutation feature importance for any survival model.

    For each feature, the values in that column are **randomly shuffled**
    ``n_repeats`` times while all other columns remain unchanged.  The mean
    drop in Harrell's C-index relative to the baseline (unshuffled) score
    quantifies how much the model depends on that feature.

    Permutation importance is more reliable than native tree-based importance
    (which is biased toward high-cardinality features) because it directly
    measures the effect of removing a feature's information from predictions.

    A feature with **high permutation importance** is one the model genuinely
    relies on for ranking patients.  A feature with **near-zero importance**
    provides redundant or uninformative signal and could be dropped without
    harming performance.

    Args:
        predict_fn (callable): Function with signature ``(X_df) -> risk_scores``
            where ``X_df`` is a ``pd.DataFrame`` and the returned ``np.ndarray``
            contains continuous risk scores (higher = higher risk).
            For GBSA: ``lambda X: gbsa_model.predict(X)``.
            For CoxPH: ``lambda X: -cox_model.predict_log_partial_hazard(X_scaled)``.
        X (pd.DataFrame): Feature matrix to evaluate on (NaN-free).
            Typically the holdout test set to avoid overfitting the importance
            estimate to the training distribution.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.
        n_repeats (int): Number of shuffles per feature.  Default ``20``.
            Higher values reduce variance in the importance estimate.
        seed (int): Random seed.  Default ``RANDOM_SEED``.

    Returns:
        pd.DataFrame: One row per feature, sorted descending by
            ``importance_mean``.  Columns:

            - ``'feature'``         (str): Feature name.
            - ``'importance_mean'`` (float): Mean C-index drop across repeats.
              Positive = removing the feature hurts; negative = shuffling
              accidentally improves the score (feature adds noise).
            - ``'importance_std'``  (float): Standard deviation across repeats —
              a proxy for stability of the importance estimate.
    """
    rng = np.random.RandomState(seed)
    baseline_scores = predict_fn(X)
    baseline_c = concordance_index(y_duration, -baseline_scores, y_event)

    results = []
    for feat in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            perm_scores = predict_fn(X_perm)
            perm_c = concordance_index(y_duration, -perm_scores, y_event)
            drops.append(baseline_c - perm_c)
        results.append({'feature': feat,
                        'importance_mean': np.mean(drops),
                        'importance_std':  np.std(drops)})

    df = pd.DataFrame(results).sort_values('importance_mean', ascending=False)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CSF imputer
# ═══════════════════════════════════════════════════════════════════════════════

def build_csf_imputer(df_baseline, target_col='ABETA', seed=RANDOM_SEED):
    """
    Train a LightGBM regression model to predict a missing CSF biomarker
    (default: Aβ42) from non-invasive features available for all subjects.

    Lumbar puncture is required for CSF collection and is refused by
    approximately 40–50 % of ADNI participants, creating systematic missing
    data in CSF columns.  This imputer uses MRI volumes, PET, cognition, and
    genetics — which are available for nearly all subjects — to predict the
    missing CSF values.

    The model is trained on subjects with known CSF values only.  A 15 %
    random holdout within that subset evaluates imputation quality without
    re-using training data.  Holdout RMSE and R² are printed; an R² > 0.40
    is considered acceptable for this task.

    **Leakage safeguard:** predictor features are restricted to measurements
    that do not require lumbar puncture.  The target column itself is never
    used as a predictor.

    Args:
        df_baseline (pd.DataFrame): Baseline visit DataFrame containing both
            the target CSF column and the predictor columns listed below.
            Rows where ``target_col`` is NaN are excluded from training.
        target_col (str): Name of the CSF column to impute.
            Default ``'ABETA'`` (Aβ42 in pg/mL).  Also works for ``'TAU'``
            and ``'PTAU'``.
        seed (int): Random seed for model training and train/test split.
            Default ``RANDOM_SEED``.

    Returns:
        tuple:
            - **model** (lgb.LGBMRegressor): Fitted LightGBM regressor.
              Call ``model.predict(X[predictor_cols].fillna(0))`` to impute.
            - **predictor_cols** (list[str]): List of feature columns actually
              used (intersection of the desired predictors and columns
              present in ``df_baseline``).
    """
    predictor_cols = [
        'AGE', 'PTGENDER_num', 'PTEDUCAT', 'APOE4',
        'Hippocampus_ICV', 'Entorhinal_ICV', 'Ventricles_ICV',
        'MMSE', 'CDRSB', 'ADAS13', 'FAQ',
        'AV45', 'FDG',
    ]
    predictor_cols = [c for c in predictor_cols if c in df_baseline.columns]

    known = df_baseline[df_baseline[target_col].notna()].copy()
    X_csf = known[predictor_cols].copy().fillna(known[predictor_cols].median())
    y_csf = known[target_col].values

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_csf, y_csf, test_size=0.15, random_state=seed)

    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=15, random_state=seed, verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)])

    preds = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    print(f'  CSF imputer [{target_col}]: holdout RMSE={rmse:.1f}, R²={r2:.3f}')
    return model, predictor_cols


# ═══════════════════════════════════════════════════════════════════════════════
# CV helpers
# ═══════════════════════════════════════════════════════════════════════════════

def cv_cindex(X, y_event, y_duration, predict_fn, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """
    Compute the mean and standard deviation of Harrell's C-index via
    stratified K-fold cross-validation using out-of-fold (OOF) predictions.

    In each fold, ``predict_fn`` is called with the training split to fit a
    model and the validation split to generate predictions.  The C-index is
    computed on the validation fold only, so the result is an unbiased
    estimate of generalisation performance.

    Args:
        X (pd.DataFrame): Fully imputed feature matrix, shape ``(n, p)``.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring, shape ``(n,)``.
        predict_fn (callable): Function with signature
            ``(X_train, y_ev_train, y_dur_train, X_val) -> risk_scores``
            where ``risk_scores`` is an ``np.ndarray`` of length ``n_val``
            (higher = higher risk).
        n_folds (int): Number of stratified CV folds.  Default ``N_FOLDS`` (5).
        seed (int): Random seed for fold assignment.  Default ``RANDOM_SEED``.

    Returns:
        tuple:
            - **mean_c** (float): Mean C-index across folds.
            - **std_c** (float): Standard deviation across folds —
              a proxy for estimate stability; large std (> 0.05) means
              performance is sensitive to which subjects are in each fold.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cindices = []
    for tr, va in skf.split(X, y_event):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        ev_tr, dur_tr = y_event[tr], y_duration[tr]
        ev_va, dur_va = y_event[va], y_duration[va]
        scores = predict_fn(X_tr, ev_tr, dur_tr, X_va)
        if ev_va.sum() > 0:
            c = concordance_index(dur_va, -scores, ev_va)
            cindices.append(c)
    return np.mean(cindices), np.std(cindices)


def binary_horizon_dataset(y_event, y_duration, horizon_yr):
    """
    Construct binary classification labels for fixed-horizon survival prediction.

    Converts the survival problem into a binary classification problem at a
    specific time horizon: "did the patient experience the event before
    ``horizon_yr``?"

    Subjects are assigned label ``1`` if they converted **before** the horizon,
    label ``0`` if they were event-free at or beyond the horizon.  Censored
    subjects whose follow-up ended *before* the horizon are **excluded** because
    their outcome at the horizon is unknown — including them would introduce
    label noise that inflates false-negative rates.

    Args:
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.
        horizon_yr (float): Prediction horizon in years (e.g. ``3.0`` or ``5.0``).

    Returns:
        tuple:
            - **label** (np.ndarray[int]): Binary labels (0 or 1) for the
              included subjects only.  Length equals ``include.sum()``.
            - **include** (np.ndarray[bool]): Boolean mask of length ``n``
              selecting subjects with determinable outcomes at the horizon.
              Pass to ``X.iloc[include]`` to align features.
    """
    label  = np.full(len(y_event), -1, dtype=int)
    is_ev  = y_event == 1
    is_cen = y_event == 0
    label[is_ev  & (y_duration <= horizon_yr)] = 1
    label[is_ev  & (y_duration >  horizon_yr)] = 0
    label[is_cen & (y_duration >= horizon_yr)] = 0
    include = label != -1
    return label[include], include


def horizon_aucs(X_imp, y_event, y_duration, train_predict_fn, horizons=HORIZONS):
    """
    Compute cross-validated AUC at each fixed prediction horizon.

    For each horizon, ``binary_horizon_dataset`` converts survival labels into
    binary outcomes, then ``N_FOLDS``-fold stratified CV computes AUC using
    OOF predicted probabilities.  Horizons with fewer than 15 events are
    skipped to avoid degenerate AUC estimates.

    Args:
        X_imp (pd.DataFrame): Fully imputed feature matrix.
        y_event (np.ndarray): Binary event indicators.
        y_duration (np.ndarray): Time to event or censoring in years.
        train_predict_fn (callable): Function with signature
            ``(X_train, y_bin_train, X_val) -> predicted_probabilities``
            where ``y_bin_train`` is the binary horizon label.
        horizons (list[int]): Prediction horizons in years to evaluate.
            Default ``HORIZONS`` (e.g. ``[3, 5]``).

    Returns:
        dict: Maps each horizon (int) to a tuple ``(mean_auc, std_auc)``
            across folds.  Horizons with too few events are omitted.
    """
    aucs = {}
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    for h in horizons:
        y_bin, include = binary_horizon_dataset(y_event, y_duration, h)
        if y_bin.sum() < 15:
            print(f'  Skipping {h}yr AUC — too few events')
            continue
        X_h = X_imp.iloc[include]
        fold_aucs = []
        for tr, va in skf.split(X_h, y_bin):
            X_tr, X_va = X_h.iloc[tr], X_h.iloc[va]
            y_tr, y_va = y_bin[tr], y_bin[va]
            probs = train_predict_fn(X_tr, y_tr, X_va)
            if len(np.unique(y_va)) > 1:
                fold_aucs.append(roc_auc_score(y_va, probs))
        if fold_aucs:
            aucs[h] = (np.mean(fold_aucs), np.std(fold_aucs))
            print(f'  AUC {h}yr: {aucs[h][0]:.4f} ± {aucs[h][1]:.4f}')
    return aucs


# ═══════════════════════════════════════════════════════════════════════════════
# GBSA model
# ═══════════════════════════════════════════════════════════════════════════════

def gbsa_survival_cv(X_imp, y_event, y_duration, feature_names, label,
                     n_trials=40, seed=RANDOM_SEED):
    """
    Train and tune a ``GradientBoostingSurvivalAnalysis`` model using Optuna
    Bayesian hyperparameter optimisation with stratified cross-validation.

    GBSA fits an ensemble of decision trees to minimise the Cox partial
    likelihood, enabling non-linear relationships and feature interactions
    that the linear Cox PH model cannot capture.  The Antolini time-dependent
    C-index is used as the optimisation objective rather than Harrell's C
    because it is more appropriate when the proportional hazards assumption
    is violated.

    **Overfitting safeguards:**

    - The Optuna objective evaluates **OOF C-td** (validation fold only) in
      every CV split.  Hyperparameters that overfit training folds but fail on
      validation folds are penalised.
    - Per-fold training C-td is also tracked and stored as a trial user
      attribute, so the train-vs-OOF gap is visible for the best trial.
    - The final model is refit on **all** development data.  Its in-sample
      C-td is returned alongside the OOF estimate so the caller can inspect
      the gap.  A gap > 0.06–0.08 warrants additional regularisation.

    HPO search space:

    | Parameter | Range | Scale |
    |-----------|-------|-------|
    | ``learning_rate`` | [0.005, 0.20] | log-uniform |
    | ``n_estimators``  | [100, 800] | step 50 |
    | ``max_depth``     | [1, 6] | integer |
    | ``min_samples_split`` | [2, 30] | integer |
    | ``min_samples_leaf``  | [1, 30] | integer |
    | ``max_features``  | [0.3, 1.0] | uniform |
    | ``subsample``     | [0.5, 1.0] | uniform |

    Args:
        X_imp (pd.DataFrame): Fully imputed feature matrix, shape ``(n, p)``.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.
        feature_names (list[str]): Feature column names in the same order as
            ``X_imp.columns``.  Used to label the importance series.
        label (str): Cohort label for log messages (e.g. ``'MCI->Dementia'``).
        n_trials (int): Number of Optuna HPO trials.  Default ``40``.
            Increase to 60–80 for more thorough search.
        seed (int): Random seed.  Default ``RANDOM_SEED``.

    Returns:
        tuple of six elements:

            - **oof_c** (float): Mean OOF Antolini C-td from the best Optuna
              trial — the honest cross-validation estimate.
            - **train_c** (float): In-sample Antolini C-td of the final model
              on all development data.  Compare against ``oof_c`` to gauge
              overfitting.
            - **imp** (pd.Series): Feature importances (mean decrease in
              impurity across all trees), indexed by feature name, sorted
              descending.
            - **final_model** (GradientBoostingSurvivalAnalysis): Final model
              fitted on all development data with the best hyperparameters.
            - **study** (optuna.Study): Completed Optuna study — use for HPO
              history and hyperparameter importance plots.
            - **oof_preds** (np.ndarray): OOF risk scores (``model.predict(X)``
              on the held-out fold in each CV split), shape ``(n,)``.  Useful
              for ensemble construction without data leakage.
    """
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from pycox.evaluation import EvalSurv

    def get_antolini_c(surv_funcs, durations, events):
        time_grid   = surv_funcs[0].x
        surv_matrix = np.row_stack([fn(time_grid) for fn in surv_funcs]).T
        ev = EvalSurv(
            surv=pd.DataFrame(surv_matrix, index=time_grid),
            durations=durations,
            events=events
        )
        return ev.concordance_td()

    y_struct = np.array(
        [(bool(e), t) for e, t in zip(y_event, y_duration)],
        dtype=[('event', bool), ('time', float)]
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    def objective(trial):
        params = dict(
            loss='coxph',
            learning_rate=trial.suggest_float('learning_rate', 0.005, 0.20, log=True),
            n_estimators=trial.suggest_int('n_estimators', 100, 800, step=50),
            max_depth=trial.suggest_int('max_depth', 1, 6),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 30),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 30),
            max_features=trial.suggest_float('max_features', 0.3, 1.0),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=seed,
        )

        cs_val, cs_train = [], []
        for tr, va in skf.split(X_imp, y_event):
            model = GradientBoostingSurvivalAnalysis(**params)
            model.fit(X_imp.iloc[tr], y_struct[tr])
            surv_funcs_va = model.predict_survival_function(X_imp.iloc[va])
            c_val = get_antolini_c(surv_funcs_va, y_duration[va], y_event[va])
            cs_val.append(c_val)
            surv_funcs_tr = model.predict_survival_function(X_imp.iloc[tr])
            c_tr = get_antolini_c(surv_funcs_tr, y_duration[tr], y_event[tr])
            cs_train.append(c_tr)

        trial.set_user_attr('mean_train_c', float(np.mean(cs_train)))
        trial.set_user_attr('mean_val_c',   float(np.mean(cs_val)))
        return np.mean(cs_val)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best      = study.best_params
    best_trial = study.best_trial
    oof_c     = study.best_value
    print(f'  [{label}] GBSA best OOF C-td: {oof_c:.4f} | '
          f'train-C: {best_trial.user_attrs.get("mean_train_c", float("nan")):.4f} | '
          f'params: {best}')

    # OOF predictions with best params
    oof_preds = np.zeros(len(X_imp))
    for tr, va in skf.split(X_imp, y_event):
        m = GradientBoostingSurvivalAnalysis(loss='coxph', random_state=seed, **best)
        m.fit(X_imp.iloc[tr], y_struct[tr])
        oof_preds[va] = m.predict(X_imp.iloc[va])

    # Final model on all dev data
    final_model = GradientBoostingSurvivalAnalysis(loss='coxph', random_state=seed, **best)
    final_model.fit(X_imp, y_struct)

    imp = pd.Series(
        final_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    surv_funcs_all = final_model.predict_survival_function(X_imp)
    train_c = get_antolini_c(surv_funcs_all, y_duration, y_event)
    print(f'  [{label}] GBSA in-sample C-td: {train_c:.4f}  '
          f'(OOF={oof_c:.4f}, gap={train_c - oof_c:+.4f})')

    return oof_c, train_c, imp, final_model, study, oof_preds


# ═══════════════════════════════════════════════════════════════════════════════
# DeepSurv model
# ═══════════════════════════════════════════════════════════════════════════════

def run_deepsurv(X_imp, y_event, y_duration, label,
                  n_trials=25, seed=RANDOM_SEED):
    """
    Train and tune a DeepSurv (neural Cox PH) model using pycox and
    torchtuples, with Optuna hyperparameter optimisation over architecture
    and training hyperparameters.

    DeepSurv replaces the linear ``βᵀx`` term in Cox PH with a multi-layer
    perceptron (MLP), allowing the model to learn non-linear feature
    representations.  The Cox partial log-likelihood loss is backpropagated
    through the network.

    **Overfitting safeguards:**

    - Early stopping with ``patience=10`` halts each CV fold when validation
      loss stops improving, preventing epoch-level overfitting.
    - Per-fold training C-td is tracked inside the Optuna objective, allowing
      the train-vs-OOF gap to be inspected for the best trial.
    - Dropout and L2 weight decay are jointly tuned by Optuna.
    - Features are standardised using a ``StandardScaler`` fitted only on the
      training portion in each fold (no leakage of validation statistics).
    - The final model uses an 80/20 stratified split for early stopping; the
      scaler for the final model is fitted on the 80 % training split only.
    - The final refit uses lr * 0.5 relative to the Optuna-tuned lr, since
      Optuna tuned on fold-sized data and the full training set benefits from
      a more conservative learning rate.
    - A convergence check after the final refit automatically rebuilds with
      lr * 0.3 if the model diverged (in-sample C-td < 0.55).

    HPO search space:

    | Parameter | Range | Scale |
    |-----------|-------|-------|
    | Architecture | 6 MLP options from [32,32] to [256,256,128] | categorical |
    | ``dropout`` | [0.05, 0.50] | uniform |
    | ``lr`` (learning rate) | [5e-5, 5e-2] | log-uniform |
    | ``wd`` (weight decay) | [1e-5, 1e-2] | log-uniform |
    | ``batch`` size | {32, 64, 128, 256} | categorical |

    Args:
        X_imp (pd.DataFrame): Fully imputed feature matrix, shape ``(n, p)``.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.
        label (str): Cohort label for log messages (e.g. ``'MCI->Dementia'``).
        n_trials (int): Number of Optuna HPO trials.  Default ``25``.
        seed (int): Random seed.  Default ``RANDOM_SEED``.

    Returns:
        tuple of seven elements:

            - **oof_c** (float): Best OOF C-td from Optuna HPO.
            - **train_c** (float): In-sample C-td of the final model on all
              development data.
            - **final** (PycoxCoxPH): Final DeepSurv model with baseline
              hazards computed.
            - **scaler** (StandardScaler): Fitted on the 80 % training split of
              the dev set.  **Must** be applied to any new data before inference.
            - **loss_history** (dict): ``{'train': [...], 'val': [...]}`` lists
              of per-epoch negative partial log-likelihood from the final refit.
              Empty lists if the pycox log format is incompatible.
            - **study** (optuna.Study): Completed Optuna study.
            - **oof_preds** (np.ndarray): OOF predicted risk scores (negative
              mean survival probability), shape ``(n,)``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    ARCH_OPTIONS = [
        [32, 32],
        [64, 64],
        [128, 128],
        [256, 128, 64],
        [256, 256, 128],
        [128, 64, 32],
    ]

    scaler_hpo = StandardScaler()
    X_scaled = scaler_hpo.fit_transform(X_imp.values).astype(np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    def objective(trial):
        arch_idx = trial.suggest_categorical('arch_idx', list(range(len(ARCH_OPTIONS))))
        hidden   = ARCH_OPTIONS[arch_idx]
        dropout  = trial.suggest_float('dropout', 0.05, 0.50)
        lr       = trial.suggest_float('lr', 5e-5, 5e-2, log=True)
        wd       = trial.suggest_float('wd', 1e-5, 1e-2, log=True)
        batch_sz = trial.suggest_categorical('batch', [32, 64, 128, 256])

        fold_cs_val, fold_cs_train = [], []
        for tr, va in skf.split(X_scaled, y_event):
            net = tt.practical.MLPVanilla(
                X_scaled.shape[1], hidden, 1,
                batch_norm=True, dropout=dropout)
            model = PycoxCoxPH(net, tt.optim.Adam(lr=lr, weight_decay=wd))
            y_tr = (y_duration[tr].astype(np.float32), y_event[tr].astype(np.float32))
            y_va = (y_duration[va].astype(np.float32), y_event[va].astype(np.float32))
            model.fit(X_scaled[tr], y_tr, batch_sz, 60,
                      val_data=(X_scaled[va], y_va),
                      callbacks=[tt.callbacks.EarlyStopping(patience=10)],
                      verbose=False)
            model.compute_baseline_hazards()
            surv_va = model.predict_surv_df(X_scaled[va])
            ev_va = EvalSurv(surv_va, y_duration[va].astype(np.float64),
                             y_event[va].astype(bool))
            fold_cs_val.append(ev_va.concordance_td())
            surv_tr = model.predict_surv_df(X_scaled[tr])
            ev_tr = EvalSurv(surv_tr, y_duration[tr].astype(np.float64),
                             y_event[tr].astype(bool))
            fold_cs_train.append(ev_tr.concordance_td())

        trial.set_user_attr('mean_train_c', float(np.mean(fold_cs_train)))
        return np.mean(fold_cs_val)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best  = study.best_params
    oof_c = study.best_value
    print(f'  [{label}] DeepSurv best OOF C-td: {oof_c:.4f} | '
          f'train-C: {study.best_trial.user_attrs.get("mean_train_c", float("nan")):.4f} | '
          f'{best}')

    # OOF predictions
    hidden_final = ARCH_OPTIONS[best['arch_idx']]
    oof_preds = np.zeros(len(X_scaled))
    for tr, va in skf.split(X_scaled, y_event):
        net = tt.practical.MLPVanilla(
            X_scaled.shape[1], hidden_final, 1,
            batch_norm=True, dropout=best['dropout'])
        m = PycoxCoxPH(net, tt.optim.Adam(lr=best['lr'], weight_decay=best['wd']))
        y_tr = (y_duration[tr].astype(np.float32), y_event[tr].astype(np.float32))
        y_va = (y_duration[va].astype(np.float32), y_event[va].astype(np.float32))
        m.fit(X_scaled[tr], y_tr, best['batch'], 60,
              val_data=(X_scaled[va], y_va),
              callbacks=[tt.callbacks.EarlyStopping(patience=10)],
              verbose=False)
        m.compute_baseline_hazards()
        surv_tmp = m.predict_surv_df(X_scaled[va])
        oof_preds[va] = -surv_tmp.values.mean(axis=0)

    # Final model on 80/20 split (scaler fitted on 80 % only)
    from sklearn.model_selection import train_test_split as _tts
    idx = np.arange(len(X_imp))
    idx_tr, idx_va = _tts(idx, test_size=0.20, stratify=y_event, random_state=seed)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_imp.iloc[idx_tr].values).astype(np.float32)
    X_va_sc = scaler.transform(X_imp.iloc[idx_va].values).astype(np.float32)

    def _build_and_fit(lr_scale):
        net = tt.practical.MLPVanilla(
            X_tr_sc.shape[1], hidden_final, 1,
            batch_norm=True, dropout=best['dropout'])
        mdl = PycoxCoxPH(net, tt.optim.Adam(
            lr=best['lr'] * lr_scale, weight_decay=best['wd']))
        mdl.fit(
            X_tr_sc, y_tr_f, best['batch'], 200,
            val_data=(X_va_sc, y_va_f),
            callbacks=[tt.callbacks.EarlyStopping(patience=15)],
            verbose=False,
        )
        mdl.compute_baseline_hazards(
            input=X_tr_sc,
            target=(
                y_duration[idx_tr].astype(np.float32),
                y_event[idx_tr].astype(np.float32)
            ),
            batch_size=256
        )
        return mdl

    y_tr_f = (y_duration[idx_tr].astype(np.float32), y_event[idx_tr].astype(np.float32))
    y_va_f = (y_duration[idx_va].astype(np.float32), y_event[idx_va].astype(np.float32))

    # First attempt at lr * 0.5
    final = _build_and_fit(0.5)

    # Convergence check — if model diverged, retry at lr * 0.3 then lr * 0.1
    for lr_scale in [0.3, 0.1]:
        _surv_check = final.predict_surv_df(X_tr_sc[:10])
        _ev_check = EvalSurv(_surv_check,
                             y_duration[idx_tr[:10]].astype(np.float64),
                             y_event[idx_tr[:10]].astype(bool))
        _c_check = _ev_check.concordance_td()
        if _c_check >= 0.55:
            break
        print(f'  WARNING: final refit C={_c_check:.4f} — diverged, '
              f'rebuilding with lr*{lr_scale}')
        final = _build_and_fit(lr_scale)

    loss_history = {'train': [], 'val': []}

    surv_all = final.predict_surv_df(scaler.transform(X_imp.values).astype(np.float32))
    ev_all = EvalSurv(surv_all, y_duration.astype(np.float64), y_event.astype(bool))
    train_c = ev_all.concordance_td()
    print(f'  [{label}] DeepSurv in-sample C-td: {train_c:.4f}  '
          f'(OOF={oof_c:.4f}, gap={train_c - oof_c:+.4f})')

    return oof_c, train_c, final, scaler, loss_history, study, oof_preds


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def calc_deepsurv_c(model, scaler, X, y_event, y_duration):
    """
    Evaluate a fitted DeepSurv model on an arbitrary dataset and return the
    IPCW time-dependent concordance index together with the full survival matrix.

    This function should be called on the **holdout test set** returned by
    ``make_holdout_split`` to obtain an unbiased final performance estimate.
    Calling it on training data produces an overly optimistic score.

    Args:
        model (PycoxCoxPH): Fitted DeepSurv model with baseline hazards
            already computed via ``model.compute_baseline_hazards()``.
        scaler (StandardScaler): The scaler returned by ``run_deepsurv`` —
            must be the same scaler used during training.  Applies
            ``scaler.transform`` (not ``fit_transform``) to avoid data leakage.
        X (pd.DataFrame): Feature matrix (NaN-free, unscaled), shape ``(n, p)``.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.

    Returns:
        tuple:
            - **c_td** (float): IPCW Antolini adj_antolini C-td on the provided data.
            - **surv** (pd.DataFrame): Predicted survival matrix, shape
              ``(n_timepoints, n_subjects)``.  Row index = time points in years;
              columns = subject indices 0..n-1.  Compatible with
              ``weighted_ensemble_td`` and ``bootstrap_cindex_td``.
    """
    X_scaled = scaler.transform(X.values).astype(np.float32)
    surv = model.predict_surv_df(X_scaled)

    times     = surv.index.values.astype(np.float64)
    surv_arr  = surv.values.astype(np.float64)
    durations = y_duration.astype(np.float64)
    events    = y_event.astype(np.int32)

    surv_idx = np.clip(np.searchsorted(times, durations), 0, len(times) - 1)
    c_td = concordance_td(durations, events, surv_arr, surv_idx,
                           method='adj_antolini', ipcw=True)
    print(f'  DeepSurv test C-td: {c_td:.4f}')
    return c_td, surv


def calc_gbsa_c(model, X, y_event, y_duration):
    """
    Evaluate a fitted GBSA model on an arbitrary dataset and return the
    IPCW time-dependent concordance index together with the full survival matrix.

    This function should be called on the **holdout test set** to obtain an
    unbiased performance estimate.

    Args:
        model (GradientBoostingSurvivalAnalysis): Fitted sksurv GBSA model
            returned by ``gbsa_survival_cv``.
        X (pd.DataFrame): Feature matrix (NaN-free), shape ``(n, p)``.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.

    Returns:
        tuple:
            - **c_td** (float): IPCW Antolini C-td on the provided data.
            - **surv** (pd.DataFrame): Predicted survival matrix, shape
              ``(n_timepoints, n_subjects)``.  Compatible with
              ``weighted_ensemble_td`` and ``bootstrap_cindex_td``.
    """
    surv_funcs  = model.predict_survival_function(X)
    time_grid   = surv_funcs[0].x
    surv_matrix = np.row_stack([fn(time_grid) for fn in surv_funcs]).T
    surv        = pd.DataFrame(surv_matrix, index=time_grid)
    surv_idx    = np.clip(np.searchsorted(time_grid, y_duration), 0, len(time_grid) - 1)

    c_td = concordance_td(y_duration, y_event, surv, surv_idx,
                           method='adj_antolini', ipcw=True)
    print(f'  GBSA test C-td: {c_td:.4f}')
    return c_td, surv


def run_cox_ph(X_imp, y_event, y_duration, label, n_trials=30, seed=RANDOM_SEED):
    """
    Train and tune a regularised Cox PH model using lifelines ``CoxPHFitter``
    and Optuna hyperparameter search with stratified cross-validation.

    Features are standardised with ``StandardScaler`` before fitting so that
    coefficient magnitudes are directly comparable across features.  The
    penaliser weight and L1/L2 ratio are jointly optimised; ``l1_ratio=0`` is
    pure ridge (retains all features), ``l1_ratio=1`` is pure LASSO (zeroes
    out irrelevant features).  The search space caps ``l1_ratio`` at 0.5 to
    prevent full coefficient collapse in high-dimensional settings.

    **Overfitting safeguards:**

    - OOF Harrell C-index is maximised by Optuna (validation fold only).
    - Per-fold training C is also tracked for gap inspection.
    - The penaliser floor is set to 0.1 to ensure meaningful regularisation.
    - Collapsed models (all-zero coefficients) receive a score of 0.0.

    HPO search space:

    | Parameter | Range | Scale |
    |-----------|-------|-------|
    | ``penalizer`` | [0.1, 10.0] | log-uniform |
    | ``l1_ratio``  | [0.0, 0.5] | uniform |

    Args:
        X_imp (pd.DataFrame): Fully imputed feature matrix, shape ``(n, p)``.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.
        label (str): Cohort label for log messages (e.g. ``'MCI->Dementia'``).
        n_trials (int): Number of Optuna HPO trials.  Default ``30``.
        seed (int): Random seed.  Default ``RANDOM_SEED``.

    Returns:
        tuple of five elements:

            - **oof_c** (float): Best OOF Harrell C from HPO cross-validation.
            - **train_c** (float): In-sample Harrell C of the final model on
              all development data.
            - **final_model** (CoxPHFitter): Final model fitted on all dev data.
              Coefficients accessible via ``final_model.params_``.
            - **scaler** (StandardScaler): Fitted on all dev data.  **Must**
              be applied to any new data before passing to ``calc_cox_ph_c``.
            - **study** (optuna.Study): Completed Optuna study.
    """
    np.random.seed(seed)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imp.values), columns=X_imp.columns)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    def _fit(X_df, y_ev, y_dur, penalizer, l1_ratio):
        df = X_df.copy()
        df['_duration'] = y_dur
        df['_event'] = y_ev.astype(int)
        fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        fitter.fit(df, duration_col='_duration', event_col='_event', show_progress=False)
        return fitter

    def objective(trial):
        penalizer = trial.suggest_float('penalizer', 0.1, 10.0, log=True)
        l1_ratio  = trial.suggest_float('l1_ratio', 0.0, 0.5)
        fold_cs_val, fold_cs_train = [], []
        for tr, va in skf.split(X_scaled, y_event):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    fitter = _fit(X_scaled.iloc[tr], y_event[tr], y_duration[tr],
                                  penalizer, l1_ratio)
                    if np.all(np.abs(fitter.params_.values) < 1e-6):
                        fold_cs_val.append(0.0)
                        fold_cs_train.append(0.0)
                        continue
                    val_df = X_scaled.iloc[va].copy()
                    val_df['_duration'] = y_duration[va]
                    val_df['_event'] = y_event[va].astype(int)
                    c_val = fitter.score(val_df, scoring_method='concordance_index')
                    if c_val < 0.45:
                        c_val = 0.0
                    fold_cs_val.append(c_val)
                    tr_df = X_scaled.iloc[tr].copy()
                    tr_df['_duration'] = y_duration[tr]
                    tr_df['_event'] = y_event[tr].astype(int)
                    c_tr = fitter.score(tr_df, scoring_method='concordance_index')
                    fold_cs_train.append(c_tr)
                except Exception:
                    fold_cs_val.append(0.0)
                    fold_cs_train.append(0.0)
        trial.set_user_attr('mean_train_c', float(np.mean(fold_cs_train)))
        return np.mean(fold_cs_val)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best  = study.best_params
    oof_c = study.best_value
    print(f'  [{label}] CoxPH best OOF C: {oof_c:.4f} | '
          f'train-C: {study.best_trial.user_attrs.get("mean_train_c", float("nan")):.4f} | '
          f'params: {best}')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        final_model = _fit(X_scaled, y_event, y_duration,
                           best['penalizer'], best['l1_ratio'])

    full_df = X_scaled.copy()
    full_df['_duration'] = y_duration
    full_df['_event'] = y_event.astype(int)
    train_c = final_model.score(full_df, scoring_method='concordance_index')
    print(f'  [{label}] CoxPH in-sample C: {train_c:.4f}  '
          f'(OOF={oof_c:.4f}, gap={train_c - oof_c:+.4f})')

    return oof_c, train_c, final_model, scaler, study


def calc_cox_ph_c(model, scaler, X, y_event, y_duration):
    """
    Evaluate a fitted ``CoxPHFitter`` on an arbitrary dataset and return the
    IPCW time-dependent concordance index together with the full survival matrix.

    Survival curves are predicted on a time grid consisting of observed event
    times in ``y_duration``, producing a ``(n_times × n_subjects)`` DataFrame
    that matches the output format of ``calc_gbsa_c`` and ``calc_deepsurv_c``
    for use in ``weighted_ensemble_td``.

    Args:
        model (CoxPHFitter): Fitted lifelines CoxPHFitter returned by ``run_cox_ph``.
        scaler (StandardScaler): The scaler returned by ``run_cox_ph`` — must be
            the same object used during training.
        X (pd.DataFrame): Feature matrix (NaN-free, unscaled), shape ``(n, p)``.
        y_event (np.ndarray): Binary event indicators, shape ``(n,)``.
        y_duration (np.ndarray): Time to event or censoring in years, shape ``(n,)``.

    Returns:
        tuple:
            - **c_td** (float): IPCW Antolini adj_antolini C-td on the provided data.
            - **surv** (pd.DataFrame): Predicted survival matrix, shape
              ``(n_timepoints, n_subjects)``.  Index = observed event times in years;
              columns = subject indices 0..n-1.  Values clipped to ``[0, 1]``.
    """
    X_scaled = pd.DataFrame(scaler.transform(X.values), columns=X.columns)
    time_grid = np.sort(np.unique(y_duration[y_event == 1])).astype(np.float64)

    surv = model.predict_survival_function(
        X_scaled.reset_index(drop=True), times=time_grid)
    surv.index = time_grid
    surv.columns = range(len(X_scaled))
    surv = surv.clip(lower=0.0, upper=1.0)

    surv_arr = surv.values.astype(np.float64)
    surv_idx = np.clip(
        np.searchsorted(time_grid, y_duration), 0, len(time_grid) - 1
    ).astype(np.int64)

    c_td = concordance_td(y_duration.astype(np.float64), y_event.astype(np.int32),
                           surv_arr, surv_idx, method='adj_antolini', ipcw=True)
    print(f'  CoxPH test C-td: {c_td:.4f}')
    return c_td, surv


# ═══════════════════════════════════════════════════════════════════════════════
# Ensemble
# ═══════════════════════════════════════════════════════════════════════════════

def weighted_ensemble(risk_score_dict, weights_dict, y_event, y_duration, label, n_trials=50):
    """
    Combine scalar risk scores from multiple models using Optuna-optimised
    weights, evaluated with Harrell's C-index.

    Each model's scores are normalised to ``[0, 1]`` before blending so that
    models with different score scales contribute equally.  Optuna searches
    for the weight combination that maximises the concordance index on the
    provided data.

    Note: for a more principled ensemble that preserves time-resolution,
    prefer ``weighted_ensemble_td`` which blends full survival curves.

    Args:
        risk_score_dict (dict): ``{model_name: np.ndarray}`` — one risk score
            array per model (higher = higher risk), all the same length.
        weights_dict (dict): Initial weights (not used for optimisation; kept
            for API compatibility).
        y_event (np.ndarray): Binary event indicators.
        y_duration (np.ndarray): Time to event or censoring in years.
        label (str): Cohort label for log messages.
        n_trials (int): Optuna trials.  Default ``50``.

    Returns:
        tuple:
            - **c** (float): Harrell's C of the optimised ensemble.
            - **ensemble_score** (np.ndarray): Optimised blended risk scores.
    """
    model_names = list(risk_score_dict.keys())
    normed = {}
    for name, scores in risk_score_dict.items():
        s_min, s_max = scores.min(), scores.max()
        normed[name] = (scores - s_min) / (s_max - s_min + 1e-9)

    def objective(trial):
        raw_weights = [trial.suggest_float(f'w_{name}', 0.2, 0.8) for name in model_names]
        total = sum(raw_weights) + 1e-9
        ensemble_score = sum(
            (w / total) * normed[name]
            for w, name in zip(raw_weights, model_names)
        )
        return concordance_index(y_duration, -ensemble_score, y_event)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    raw_weights = [best[f'w_{name}'] for name in model_names]
    total = sum(raw_weights) + 1e-9
    ensemble_score = sum(
        (w / total) * normed[name]
        for w, name in zip(raw_weights, model_names)
    )

    c = concordance_index(y_duration, -ensemble_score, y_event)
    best_weights = {name: w / total for name, w in zip(model_names, raw_weights)}
    print(f'  [{label}] Optimized ensemble C-index: {c:.4f} | weights: {best_weights}')
    return c, ensemble_score


def domain_ensemble(X_mci, y_event, y_duration, domains_dict,
                     base_model_fn, label, seed=RANDOM_SEED):
    """
    Train one base survival model per feature domain, stack their OOF risk
    scores, and fit a meta-learner (``RidgeCV``) on the stacked OOF matrix.

    This is a stacking approach that allows different feature domains
    (imaging, cognitive, CSF/PET) to contribute independently.  The
    meta-learner learns which domains are most predictive for this cohort.

    Args:
        X_mci (pd.DataFrame): Full feature matrix (NaN-free).
        y_event (np.ndarray): Binary event indicators.
        y_duration (np.ndarray): Time to event or censoring in years.
        domains_dict (dict): ``{domain_name: list[feature_names]}`` —
            returned by ``get_domain_features``.  The ``'combined'`` key
            is ignored.
        base_model_fn (callable): Function with signature
            ``(X_train, y_event_train, y_duration_train) -> predict_fn``
            where ``predict_fn(X_val)`` returns risk scores.
        label (str): Cohort label for log messages.
        seed (int): Random seed.  Default ``RANDOM_SEED``.

    Returns:
        float: Harrell's C of the meta-learner stacking ensemble on OOF predictions.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    n = len(y_event)
    domain_names = [d for d in domains_dict if d != 'combined']
    oof_matrix = np.zeros((n, len(domain_names)))

    for d_idx, domain in enumerate(domain_names):
        feats = [f for f in domains_dict[domain] if f in X_mci.columns]
        X_d = X_mci[feats]
        for tr, va in skf.split(X_d, y_event):
            model = base_model_fn(X_d.iloc[tr], y_event[tr], y_duration[tr])
            oof_matrix[va, d_idx] = model(X_d.iloc[va])

    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta.fit(oof_matrix, -np.log1p(y_duration))
    meta_pred = meta.predict(oof_matrix)
    c = concordance_index(y_duration, -meta_pred, y_event)
    print(f'  [{label}] Domain ensemble C-index: {c:.4f}')
    print(f'  Meta-learner coefficients: {dict(zip(domain_names, meta.coef_))}')
    return c


def weighted_ensemble_td(risk_score_dict, y_event, y_duration, label, n_trials=50):
    """
    Combine full survival curve predictions from multiple models using
    Optuna-optimised weights, evaluated with the IPCW Antolini C-td.

    Unlike scalar risk score ensembles, this function blends the complete
    predicted survival function ``S(t|x)`` for each patient, preserving
    time-resolution.  The blended survival matrix is a weighted average of
    the individual models' matrices, normalised to satisfy the constraint
    that weights sum to 1.

    Survival matrices with different time grids are aligned to a **union time
    grid** via index-based interpolation, forward/backward fill, and clipping
    to ``[0, 1]``.

    Args:
        risk_score_dict (dict): ``{model_name: pd.DataFrame}`` — one survival
            matrix per model.  Each DataFrame must have:

            - rows  = time points in years (the index)
            - columns = patient identifiers in the same order as ``y_event``

        y_event (np.ndarray): Binary event indicators for the **same** patients
            as the columns of the survival matrices.
        y_duration (np.ndarray): Time to event or censoring in years for those
            patients.
        label (str): Cohort label for log messages.
        n_trials (int): Optuna trials.  Default ``50``.

    Returns:
        tuple of three elements:

            - **c** (float): IPCW Antolini C-td of the optimised ensemble.
            - **ensemble_surv_df** (pd.DataFrame): Blended survival matrix on
              the union time grid.  Shape ``(n_union_times, n_subjects)``.
              Compatible with ``bootstrap_cindex_td`` and ``plot_brier_score``.
            - **best_weights** (dict): ``{model_name: weight}`` normalised to
              sum to 1.  Weights near zero indicate a redundant model.

    Raises:
        AssertionError: If the patient column indices of any two input
            DataFrames do not match.
    """
    model_names = list(risk_score_dict.keys())

    ref_cols = risk_score_dict[model_names[0]].columns
    for name in model_names[1:]:
        assert (risk_score_dict[name].columns == ref_cols).all(), \
            f"Patient ID mismatch: {name} vs {model_names[0]}"

    union_index = risk_score_dict[model_names[0]].index
    for name in model_names[1:]:
        union_index = union_index.union(risk_score_dict[name].index)
    union_index = union_index.sort_values()

    normed = {}
    for name, df in risk_score_dict.items():
        reindexed = (
            df.reindex(union_index)
              .interpolate(method='index', axis=0, limit_area='inside')
              .ffill(axis=0)
              .bfill(axis=0)
              .clip(lower=0.0, upper=1.0)
        )
        normed[name] = reindexed

    time_points = union_index.to_numpy()
    surv_idx = np.searchsorted(time_points, y_duration, side='right') - 1
    surv_idx = np.clip(surv_idx, 0, len(time_points) - 1).astype(np.int64)

    def _blend_surv(weights):
        return sum(w * normed[name] for w, name in zip(weights, model_names))

    def _cindex(surv_df):
        return concordance_td(
            durations=y_duration, events=y_event,
            surv=surv_df.values, surv_idx=surv_idx,
            method='adj_antolini', ipcw=True,
        )

    def objective(trial):
        raw_weights = [trial.suggest_float(f'w_{name}', 0.0, 1.0) for name in model_names]
        total = sum(raw_weights) + 1e-9
        return _cindex(_blend_surv([w / total for w in raw_weights]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    raw_weights = [best[f'w_{name}'] for name in model_names]
    total = sum(raw_weights) + 1e-9
    best_weights = {name: w / total for name, w in zip(model_names, raw_weights)}

    ensemble_surv_df = _blend_surv(list(best_weights.values()))
    c = _cindex(ensemble_surv_df)

    print(f'  [{label}] Ensemble C-td: {c:.4f} | weights: {best_weights}')
    return c, ensemble_surv_df, best_weights
