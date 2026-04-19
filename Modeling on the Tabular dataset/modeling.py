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
import lightgbm as lgb #for CSF imputation not survival modeling

# Deepsurv imports
import torch
import torchtuples as tt
from pycox.models import CoxPH as PycoxCoxPH
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler
from concordance import concordance_td

# Cox PH imports
from lifelines import CoxPHFitter

# Ensemble imports
from sklearn.linear_model import RidgeCV
from config import RANDOM_SEED, N_FOLDS, HORIZONS, FIG_DIR, CHECKPOINT_DIR, OUT_DIR, MRI_HARMONIZE_COLS, BASE_DIR

# ── GPU detection ─────────────────────────────────────────────────────────────
try:
    import lightgbm as lgb_test
    _test_params = {'device': 'gpu', 'verbose': -1, 'n_estimators': 1}
    lgb_test.LGBMRegressor(**_test_params).fit([[1]], [1])
    LGB_DEVICE = 'gpu'
except Exception:
    LGB_DEVICE = 'cpu'
print(f'LightGBM device: {LGB_DEVICE}')


def save_checkpoint(name, obj):
    """
    Serialize and save a Python object to disk as a .pkl file in CHECKPOINT_DIR.

    Used to persist trained models and results between pipeline runs, allowing
    the pipeline to skip expensive retraining steps when RETRAIN=False.

    Args:
        name (str): Checkpoint identifier used as the filename stem
            (e.g. 'lgb_model_mci' saves as 'lgb_model_mci.pkl').
        obj (any): Any pickle-serializable Python object (e.g. trained model,
            results dict, numpy array).

    Returns:
        None. Prints the saved path on success.
    """
    path = CHECKPOINT_DIR / f'{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  Checkpointed: {name} -> {path}')


def load_checkpoint(name):
    """
    Load a previously saved .pkl checkpoint from CHECKPOINT_DIR if it exists.

    Returns None silently if the checkpoint file is not found, allowing the
    caller to fall back to retraining when a checkpoint is missing.

    Args:
        name (str): Checkpoint identifier matching the stem used in
            save_checkpoint (e.g. 'lgb_model_mci' loads 'lgb_model_mci.pkl').

    Returns:
        any: The deserialized Python object if the checkpoint exists,
            or None if the file is not found.
    """
    path = CHECKPOINT_DIR / f'{name}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f'  Loaded checkpoint: {name}')
        return obj
    return None

def build_csf_imputer(df_baseline, target_col='ABETA', seed=RANDOM_SEED):
    '''
    Train a LightGBM model to predict CSF ABETA from available features.
    Returns fitted model and predictor feature list.
    '''
    predictor_cols = [
    'AGE', 'PTGENDER_num', 'PTEDUCAT', 'APOE4',
    'Hippocampus_ICV', 'Entorhinal_ICV', 'Ventricles_ICV',
    'MMSE', 'CDRSB', 'ADAS13', 'FAQ',
    'AV45', 'FDG',   # PET where available — these are legitimately available without LP
    ]
    predictor_cols = [c for c in predictor_cols if c in df_baseline.columns]

    known = df_baseline[df_baseline[target_col].notna()].copy()
    X_csf = known[predictor_cols].copy()
    y_csf = known[target_col].values

    # Impute predictors with median for this sub-model
    X_csf = X_csf.fillna(X_csf.median())

    # Hold out 15% to evaluate imputation quality
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
    print(f'  CSF imputer for {target_col}: holdout RMSE={rmse:.1f}, R²={r2:.3f}')
    return model, predictor_cols

def cv_cindex(X, y_event, y_duration, predict_fn, n_folds=N_FOLDS, seed=RANDOM_SEED):
    '''
    5-fold stratified CV returning mean/std C-index.
    predict_fn(X_train, y_ev_tr, y_dur_tr, X_val) -> risk_scores (higher = higher risk)
    '''
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

    Subjects are assigned label 1 if they experienced the event within
    horizon_yr, label 0 if they were event-free at or beyond horizon_yr.
    Censored subjects followed for less than horizon_yr are excluded because
    their outcome is unknown — including them would introduce label noise.

    Args:
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring in years.
        horizon_yr (float): Prediction horizon in years (e.g. 3.0 or 5.0).

    Returns:
        tuple:
            label (np.ndarray[int]): Binary labels (0 or 1) for included subjects.
            include (np.ndarray[bool]): Boolean mask selecting subjects with
                determinable outcomes at the given horizon.
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
    Compute time-dependent AUC at each fixed prediction horizon using
    stratified cross-validation.

    For each horizon, binary labels are constructed via binary_horizon_dataset,
    then N_FOLDS-fold stratified CV is run using the provided training function.
    Horizons with fewer than 15 events are skipped to ensure stable AUC estimates.

    Args:
        X_imp (pd.DataFrame): Fully imputed feature matrix (no NaNs).
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring in years.
        train_predict_fn (callable): Function with signature
            (X_train, y_bin_train, X_val) -> predicted_probabilities.
        horizons (list[int]): Prediction horizons in years to evaluate.
            Default HORIZONS (e.g. [3, 5]).

    Returns:
        dict: Maps each horizon (int) to a tuple (mean_auc, std_auc) across folds.
            Horizons with too few events are omitted from the dict.
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


def gbsa_survival_cv(X_imp, y_event, y_duration, feature_names, label,
                     n_trials=30, seed=RANDOM_SEED):
    """
    Train and tune a scikit-survival GradientBoostingSurvivalAnalysis model
    using Optuna Bayesian hyperparameter optimization and stratified
    cross-validation.

    Concordance is evaluated using Antolini's time-dependent C-index via
    pycox's EvalSurv, which requires the full survival curve S(t|x) rather
    than a scalar risk score. The survival matrix is built from sksurv's
    predict_survival_function() output and passed to EvalSurv with a KM
    censoring estimator.

    Args:
        X_imp (pd.DataFrame): Fully imputed feature matrix (no NaNs).
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring (e.g. in years).
        feature_names (list[str]): Feature column names (used for importance).
        label (str): Cohort label for progress printing (e.g. 'MCI->Dementia').
        n_trials (int): Number of Optuna hyperparameter search trials. Default 30.
        seed (int): Random seed for reproducibility. Default RANDOM_SEED.

    Returns:
        tuple:
            c (float): Antolini's C-index of the final model on the full
                training set.
            imp (pd.Series): Feature importances (mean decrease in impurity),
                sorted descending.
            final_model (GradientBoostingSurvivalAnalysis): Final model fitted
                on all data.
    """
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from pycox.evaluation import EvalSurv

    def get_antolini_c(surv_funcs, durations, events):
        """Build a pycox EvalSurv object and return Antolini's C-index."""
        time_grid   = surv_funcs[0].x
        surv_matrix = np.row_stack([fn(time_grid) for fn in surv_funcs]).T
        # surv_matrix shape: (n_times, n_subjects) — pycox convention
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
            learning_rate=trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            n_estimators=trial.suggest_int('n_estimators', 100, 600, step=50),
            max_depth=trial.suggest_int('max_depth', 2, 5),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_float('max_features', 0.5, 1.0),
            subsample=trial.suggest_float('subsample', 0.6, 1.0),
            random_state=seed,
        )

        cs = []
        for tr, va in skf.split(X_imp, y_event):
            model = GradientBoostingSurvivalAnalysis(**params)
            model.fit(X_imp.iloc[tr], y_struct[tr])

            surv_funcs = model.predict_survival_function(X_imp.iloc[va])
            c_stat = get_antolini_c(surv_funcs, y_duration[va], y_event[va])
            cs.append(c_stat)

        return np.mean(cs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f'  [{label}] GBSA best Antolini-C: {study.best_value:.4f} | params: {best}')

    # Refit final model on all data
    final_model = GradientBoostingSurvivalAnalysis(
        loss='coxph',
        random_state=seed,
        **best
    )
    final_model.fit(X_imp, y_struct)

    imp = pd.Series(
        final_model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    surv_funcs_all = final_model.predict_survival_function(X_imp)
    c = get_antolini_c(surv_funcs_all, y_duration, y_event)

    return c, imp, final_model

def run_deepsurv(X_imp, y_event, y_duration, label,
                  n_trials=20, seed=RANDOM_SEED):
    """
    Train and tune a DeepSurv (neural Cox PH) model using pycox and torchtuples,
    with Optuna hyperparameter optimization over architecture and training params.

    Features are standardized before training. The final model is refit on the
    full dataset using an 80/20 temporal split for early stopping. Baseline
    hazards are computed after fitting to enable survival function prediction.

    Args:
        X_imp (pd.DataFrame): Fully imputed feature matrix (no NaNs).
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring in years.
        label (str): Cohort label for progress printing (e.g. 'MCI->Dementia').
        n_trials (int): Number of Optuna hyperparameter search trials. Default 20.
        seed (int): Random seed for reproducibility. Default RANDOM_SEED.

    Returns:
        tuple:
            best_value (float): Best time-dependent C-index (C-td) from Optuna CV.
            final (PycoxCoxPH): Final DeepSurv model fitted on full data.
            scaler (StandardScaler): Fitted scaler — must be used to transform
                any new data before passing to this model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp.values).astype(np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    def objective(trial):
        hidden_sizes = trial.suggest_categorical(
            'hidden', [[64,64],[128,128],[256,128,64],[128,64,32]])
        dropout  = trial.suggest_float('dropout', 0.1, 0.5)
        lr       = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_sz = trial.suggest_categorical('batch', [64, 128, 256])
        fold_cs  = []
        for tr, va in skf.split(X_scaled, y_event):
            net = tt.practical.MLPVanilla(
                X_scaled.shape[1], hidden_sizes, 1,
                batch_norm=True, dropout=dropout)
            model = PycoxCoxPH(net, tt.optim.Adam(lr=lr))
            y_tr = (y_duration[tr].astype(np.float32), y_event[tr].astype(np.float32))
            y_va = (y_duration[va].astype(np.float32), y_event[va].astype(np.float32))
            model.fit(X_scaled[tr], y_tr, batch_sz, 50,
                      val_data=(X_scaled[va], y_va),
                      callbacks=[tt.callbacks.EarlyStopping(patience=10)],
                      verbose=False)
            model.compute_baseline_hazards()
            surv = model.predict_surv_df(X_scaled[va])
            ev = EvalSurv(surv,
                          y_duration[va].astype(np.float64),
                          y_event[va].astype(bool))
            fold_cs.append(ev.concordance_td())
        return np.mean(fold_cs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    print(f'  [{label}] DeepSurv best C-td: {study.best_value:.4f} | {best}')

    # Refit final model with explicit val split for early stopping
    net = tt.practical.MLPVanilla(
        X_scaled.shape[1], best['hidden'], 1,
        batch_norm=True, dropout=best['dropout'])
    final = PycoxCoxPH(net, tt.optim.Adam(best['lr']))
    y_all = (y_duration.astype(np.float32), y_event.astype(np.float32))
    n_val = int(len(X_scaled) * 0.2)
    X_tr_f, X_va_f = X_scaled[:-n_val], X_scaled[-n_val:]
    y_tr_f = (y_all[0][:-n_val], y_all[1][:-n_val])
    y_va_f = (y_all[0][-n_val:], y_all[1][-n_val:])
    final.fit(X_tr_f, y_tr_f, best['batch'], 200, verbose=False,
              val_data=(X_va_f, y_va_f),
              callbacks=[tt.callbacks.EarlyStopping(patience=15)])
    final.compute_baseline_hazards()
    return study.best_value, final, scaler


def calc_deepsurv_c(model, scaler, X, y_event, y_duration):
    """
    Evaluate a fitted DeepSurv model by computing the time-dependent C-index
    (C-td) on a given dataset.

    Updated to use an IPCW weighted version (Uno et al.) of the time-dependent
    C-index (Antolini et al.)

    Args:
        model (PycoxCoxPH): Fitted DeepSurv model with baseline hazards computed.
        scaler (StandardScaler): The scaler fitted during model training —
            must match the one returned by run_deepsurv.
        X (pd.DataFrame): Feature matrix to evaluate on (NaN-free).
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring in years.

    Returns:
        tuple:
            c_td (float): Time-dependent concordance index on the provided dataset.
            surv (pd.DataFrame): Predicted survival functions, shape
                (n_timepoints, n_subjects).
    """
    X_scaled = scaler.transform(X.values).astype(np.float32)
    surv = model.predict_surv_df(X_scaled)
    
    # Extract arrays from the surv DataFrame
    times = surv.index.values.astype(np.float64)
    surv_array = surv.values.astype(np.float64)  # shape (n_times, n)
    durations = y_duration.astype(np.float64)
    events = y_event.astype(np.int32)
    
    # Build surv_idx
    surv_idx = np.searchsorted(times, durations)
    surv_idx = np.clip(surv_idx, 0, len(times) - 1)
    
    c_td = concordance_td(durations, events, surv_array, surv_idx, method='adj_antolini', ipcw=True)
    print(f'  DeepSurv final C-td: {c_td:.4f}')
    return c_td, surv

def calc_gbsa_c(model, X, y_event, y_duration):
    """
    Evaluate a fitted GradientBoostingSurvivalAnalysis model by computing
    the time-dependent C-index (C-td) on a given dataset.

    The C-td is based on Antolini et al. with an additional IPCW weight based
    on Uno et al.

    Args:
        model (GradientBoostingSurvivalAnalysis): Fitted sksurv GBSA model.
        X (pd.DataFrame): Feature matrix to evaluate on (NaN-free).
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring in years.

    Returns:
        tuple:
            c_td (float): Time-dependent concordance index on the provided dataset.
            surv (pd.DataFrame): Predicted survival functions, shape
                (n_timepoints, n_subjects).
    """
    surv_funcs  = model.predict_survival_function(X)
    time_grid   = surv_funcs[0].x
    surv_matrix = np.row_stack([fn(time_grid) for fn in surv_funcs]).T
    surv        = pd.DataFrame(surv_matrix, index=time_grid)
    surv_idx = np.searchsorted(time_grid, y_duration)
    surv_idx = np.clip(surv_idx, 0, len(time_grid) - 1)

    c_td = concordance_td(y_duration, y_event, surv, surv_idx, method='adj_antolini',ipcw=True)
    print(f'  GBSA final C-td: {c_td:.4f}')
    return c_td, surv

def run_cox_ph(X_imp, y_event, y_duration, label, n_trials=30, seed=RANDOM_SEED):
    """
    Train and tune a regularised Cox PH model (lifelines CoxPHFitter) with
    Optuna hyperparameter search and stratified cross-validation.
 
    Features are standardised before fitting (same pattern as run_deepsurv).
    The penaliser weight and L1 ratio are optimised jointly via Optuna; l1_ratio=0
    is ridge, l1_ratio=1 is lasso, intermediate values give elastic net.
 
    HPO uses Harrell C on each val fold for speed; the final reported metric uses
    adj_antolini IPCW C-td, consistent with GBSA and DeepSurv evaluation.
 
    Args:
        X_imp (pd.DataFrame): Fully imputed feature matrix (no NaNs).
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring in years.
        label (str): Cohort label for progress printing (e.g. 'MCI->Dementia').
        n_trials (int): Number of Optuna HPO trials. Default 30.
        seed (int): Random seed. Default RANDOM_SEED.
 
    Returns:
        tuple:
            best_c_cv (float): Best Harrell C from HPO cross-validation.
            final_model (CoxPHFitter): Final model fitted on all data.
            scaler (StandardScaler): Fitted scaler — must be applied to any new
                data before passing to calc_cox_ph_c.
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
        penalizer = trial.suggest_float('penalizer', 0.1, 50.0, log=True)
        l1_ratio  = trial.suggest_float('l1_ratio', 0.0, 1.0)
        fold_cs = []
        for tr, va in skf.split(X_scaled, y_event):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    fitter = _fit(X_scaled.iloc[tr], y_event[tr], y_duration[tr],
                                  penalizer, l1_ratio)
                    val_df = X_scaled.iloc[va].copy()
                    val_df['_duration'] = y_duration[va]
                    val_df['_event'] = y_event[va].astype(int)
                    c = fitter.score(val_df, scoring_method='concordance_index')
                except Exception:
                    c = 0.0
            fold_cs.append(c)
        return np.mean(fold_cs)
 
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    print(f'  [{label}] CoxPH best CV C: {study.best_value:.4f} | params: {best}')
 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        final_model = _fit(X_scaled, y_event, y_duration,
                           best['penalizer'], best['l1_ratio'])
 
    return study.best_value, final_model, scaler
 
 
def calc_cox_ph_c(model, scaler, X, y_event, y_duration):
    """
    Evaluate a fitted CoxPHFitter model using the adj_antolini IPCW C-td,
    and return the full survival curve matrix for use in the ensemble.
 
    Survival curves are predicted on a time grid of observed event times,
    producing a (n_times × n_subjects) DataFrame matching the output format
    of calc_gbsa_c and calc_deepsurv_c.
 
    Args:
        model (CoxPHFitter): Fitted CoxPHFitter returned by run_cox_ph.
        scaler (StandardScaler): Fitted scaler returned by run_cox_ph.
        X (pd.DataFrame): Feature matrix (NaN-free, unscaled).
        y_event (np.ndarray): Binary event indicators (1=event, 0=censored).
        y_duration (np.ndarray): Time to event or censoring in years.
 
    Returns:
        tuple:
            c_td (float): adj_antolini IPCW C-td on the provided data.
            surv (pd.DataFrame): Predicted survival functions, shape
                (n_timepoints, n_subjects). Compatible with weighted_ensemble_td
                and plot_individual_survival_curves.
    """
    X_scaled = pd.DataFrame(scaler.transform(X.values), columns=X.columns)
    time_grid = np.sort(np.unique(y_duration[y_event == 1])).astype(np.float64)
 
    surv = model.predict_survival_function(X_scaled.reset_index(drop=True),
                                            times=time_grid)
    surv.index = time_grid
    surv.columns = range(len(X_scaled))
    surv = surv.clip(lower=0.0, upper=1.0)
 
    surv_arr = surv.values.astype(np.float64)
    surv_idx = np.searchsorted(time_grid, y_duration)
    surv_idx = np.clip(surv_idx, 0, len(time_grid) - 1).astype(np.int64)
 
    c_td = concordance_td(y_duration.astype(np.float64), y_event.astype(np.int32),
                           surv_arr, surv_idx, method='adj_antolini', ipcw=True)
    print(f'  CoxPH final C-td: {c_td:.4f}')
    return c_td, surv

def weighted_ensemble(risk_score_dict, weights_dict, y_event, y_duration, label, n_trials=50):
    '''
    Combine risk scores from multiple models using Optuna-optimized weights.
    risk_score_dict: {model_name: array of risk scores}
    weights_dict:    {model_name: initial weight (e.g., CV C-index)} (unused for optimization)
    '''
    model_names = list(risk_score_dict.keys())

    # Normalize each model's scores to [0,1] once
    normed = {}
    for name, scores in risk_score_dict.items():
        s_min, s_max = scores.min(), scores.max()
        normed[name] = (scores - s_min) / (s_max - s_min + 1e-9)

    def objective(trial):
        # Sample a weight for each model, then normalize to sum to 1
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

    # Reconstruct best ensemble
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
    '''
    Train one base model per domain, stack their OOF risk scores,
    fit a meta-learner (RidgeCV) on the stacked scores.
    base_model_fn(X_tr, y_ev, y_dur) -> fitted model with .predict(X) method
    '''
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

    # Meta-learner on OOF stacked scores
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    meta.fit(oof_matrix, -np.log1p(y_duration))  # predict risk
    meta_pred = meta.predict(oof_matrix)
    c = concordance_index(y_duration, -meta_pred, y_event)
    print(f'  [{label}] Domain ensemble C-index: {c:.4f}')
    print(f'  Meta-learner coefficients (domain weights): '
          f'{dict(zip(domain_names, meta.coef_))}')
    return c


def weighted_ensemble_td(risk_score_dict, y_event, y_duration, label, n_trials=50):
    '''
    Combine survival function predictions from multiple models using Optuna-optimized weights,
    evaluated with Antolini's time-dependent concordance index (adj_antolini, IPCW-weighted).

    risk_score_dict: {model_name: DataFrame with columns=patient_ids, rows=time_points}
                     Each cell contains S(t|x) — survival probability for a patient at a time.
                     Tables may have different time indices; they will be aligned to a common
                     union index by forward/backward filling within each column.
    y_event:         np.ndarray of event indicators (1=event, 0=censored)
    y_duration:      np.ndarray of observed times
    label:           Label for logging
    n_trials:        Number of Optuna optimization trials

    Returns: (best_c_index, ensemble_survival_df, best_weights)
    '''
    model_names = list(risk_score_dict.keys())

    # Validate all tables share the same patient columns
    ref_cols = risk_score_dict[model_names[0]].columns
    for name in model_names[1:]:
        assert (risk_score_dict[name].columns == ref_cols).all(), \
            f"Patient ID mismatch: {name} vs {model_names[0]}"

    # Build union time index across all models
    union_index = ref_cols  # reuse variable name would be confusing; build properly:
    union_index = risk_score_dict[model_names[0]].index
    for name in model_names[1:]:
        union_index = union_index.union(risk_score_dict[name].index)
    union_index = union_index.sort_values()

    # Reindex each model's survival table to the union time index.
    # Use nearest-time filling strictly within each column (no cross-column fill).
    # method='nearest' picks the closest existing time point for any new row.
    # limit=None allows filling across arbitrarily large gaps, which is appropriate
    # for survival functions that change slowly between observed time points.
    normed = {}
    for name, df in risk_score_dict.items():
        reindexed = (
            df.reindex(union_index)
              .interpolate(method='index', axis=0, limit_area='inside')  # interpolate interior gaps
              .ffill(axis=0)   # extend flat after last observed time (survival stays constant)
              .bfill(axis=0)   # fill any leading NaNs before first observed time
              .clip(lower=0.0, upper=1.0)
        )
        normed[name] = reindexed

    # Pre-compute surv_idx once outside the objective
    time_points = union_index.to_numpy()
    surv_idx = np.searchsorted(time_points, y_duration, side='right') - 1
    surv_idx = np.clip(surv_idx, 0, len(time_points) - 1).astype(np.int64)

    def _blend_surv(weights):
        return sum(w * normed[name] for w, name in zip(weights, model_names))

    def _cindex(surv_df):
        return concordance_td(
            durations=y_duration,
            events=y_event,
            surv=surv_df.values,
            surv_idx=surv_idx,
            method='adj_antolini',
            ipcw=True,
        )

    def objective(trial):
        raw_weights = [trial.suggest_float(f'w_{name}', 0.0, 1.0) for name in model_names]
        total = sum(raw_weights) + 1e-9
        norm_weights = [w / total for w in raw_weights]
        return _cindex(_blend_surv(norm_weights))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Reconstruct best ensemble
    best = study.best_params
    raw_weights = [best[f'w_{name}'] for name in model_names]
    total = sum(raw_weights) + 1e-9
    best_weights = {name: w / total for name, w in zip(model_names, raw_weights)}

    ensemble_surv_df = _blend_surv(list(best_weights.values()))
    c = _cindex(ensemble_surv_df)

    print(f'  [{label}] Optimized ensemble C-index: {c:.4f} | weights: {best_weights}')
    return c, ensemble_surv_df, best_weights
