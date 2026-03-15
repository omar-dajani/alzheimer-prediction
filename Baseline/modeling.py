from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import optuna
from tqdm.notebook import tqdm

# LightGBM imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from lifelines.utils import concordance_index
import lightgbm as lgb

# Deepsurv imports
import torch
import torchtuples as tt
from pycox.models import CoxPH as PycoxCoxPH
from pycox.evaluation import EvalSurv
from sklearn.preprocessing import StandardScaler

# Ensemble imports
from sklearn.linear_model import RidgeCV


CHECKPOINT_DIR = Path.cwd() / 'checkpoints'
RANDOM_SEED = 42
N_FOLDS = 5
HORIZONS = [3, 5]  # years for fixed-h


def save_checkpoint(name, obj):
    path = CHECKPOINT_DIR / f'{name}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  Checkpointed: {name} -> {path}')


def load_checkpoint(name):
    path = CHECKPOINT_DIR / f'{name}.pkl'
    if path.exists():
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f'  Loaded checkpoint: {name}')
        return obj
    return None


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
    '''
    Build binary labels for fixed-horizon prediction.
    Excludes censored subjects who were followed for < horizon years
    (their outcome is unknown).
    '''
    label  = np.full(len(y_event), -1, dtype=int)
    is_ev  = y_event == 1
    is_cen = y_event == 0
    label[is_ev  & (y_duration <= horizon_yr)] = 1
    label[is_ev  & (y_duration >  horizon_yr)] = 0
    label[is_cen & (y_duration >= horizon_yr)] = 0
    include = label != -1
    return label[include], include


def horizon_aucs(X_imp, y_event, y_duration, train_predict_fn, horizons=HORIZONS):
    '''
    AUC at each fixed horizon using stratified CV.
    train_predict_fn(X_tr, y_bin_tr, X_va) -> probs
    '''
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

def lgb_survival_cv(X_imp, y_event, y_duration, feature_names, label,
                     n_trials=30, seed=RANDOM_SEED):
    risk_target = -np.log1p(y_duration)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    def objective(trial):
        params = dict(
            objective='regression_l1',
            device='gpu',
            learning_rate=trial.suggest_float('lr', 0.02, 0.1, log=True),
            num_leaves=trial.suggest_int('num_leaves', 31, 127),
            min_child_samples=trial.suggest_int('min_child_samples', 10, 50),
            feature_fraction=trial.suggest_float('feature_fraction', 0.5, 1.0),
            bagging_fraction=trial.suggest_float('bagging_fraction', 0.6, 1.0),
            bagging_freq=5, lambda_l1=0.1, lambda_l2=0.1,
            verbose=-1, random_state=seed, n_estimators=500,
        )
        cs = []
        for tr, va in skf.split(X_imp, y_event):
            w = np.where(y_event[tr] == 1, 3.0, 1.0)
            dtrain = lgb.Dataset(X_imp.iloc[tr], label=risk_target[tr],
                                 feature_name=feature_names, weight=w)
            dval   = lgb.Dataset(X_imp.iloc[va], label=risk_target[va],
                                 feature_name=feature_names, reference=dtrain)
            m = lgb.train(params, dtrain, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50, verbose=False),
                                     lgb.log_evaluation(-1)])
            pred = m.predict(X_imp.iloc[va])
            c = concordance_index(y_duration[va], -pred, y_event[va])
            cs.append(c)
        return np.mean(cs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study.best_params
    print(f'  [{label}] LGB best C: {study.best_value:.4f} | params: {best}')

    # Refit final model
    w_all = np.where(y_event == 1, 3.0, 1.0)
    final_params = dict(objective='regression_l1', verbose=-1, random_state=seed,
                        n_estimators=500, **best)
    dtrain_all = lgb.Dataset(X_imp, label=risk_target,
                             feature_name=feature_names, weight=w_all)
    final_model = lgb.train(final_params, dtrain_all)
    imp = pd.Series(final_model.feature_importance(importance_type='gain'),
                    index=feature_names).sort_values(ascending=False)
    pred = final_model.predict(X_imp)
    c = concordance_index(y_duration, -pred, y_event)
    return c, imp, final_model


def run_deepsurv(X_imp, y_event, y_duration, label,
                  n_trials=20, seed=RANDOM_SEED):
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
    X_scaled = scaler.transform(X.values).astype(np.float32)
    surv = model.predict_surv_df(X_scaled)
    ev = EvalSurv(surv,
                  y_duration.astype(np.float64),
                  y_event.astype(bool))
    c_td = ev.concordance_td()
    print(f'  DeepSurv final C-td: {c_td:.4f}')
    return c_td, surv


def weighted_ensemble(risk_score_dict, weights_dict, y_event, y_duration, label):
    '''
    Combine risk scores from multiple models using C-index weights.
    risk_score_dict: {model_name: array of risk scores}
    weights_dict:    {model_name: weight (e.g., CV C-index)}
    '''
    total_weight = sum(weights_dict.values())
    ensemble_score = np.zeros(len(y_event))
    for name, scores in risk_score_dict.items():
        w = weights_dict.get(name, 1.0) / total_weight
        # Normalize each model's scores to [0,1] before averaging
        s_min, s_max = scores.min(), scores.max()
        norm = (scores - s_min) / (s_max - s_min + 1e-9)
        ensemble_score += w * norm
    c = concordance_index(y_duration, -ensemble_score, y_event)
    print(f'  [{label}] Weighted ensemble C-index: {c:.4f}')
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



def lgb_factory(X_tr, y_ev, y_dur):
    '''
    Define LightGBM base model factory
    '''
    risk = -np.log1p(y_dur)
    w = np.where(y_ev == 1, 3.0, 1.0)
    params = dict(objective='regression_l1', n_estimators=200, learning_rate=0.05,
                  num_leaves=31, verbose=-1)
    ds = lgb.Dataset(X_tr, label=risk, weight=w)
    m = lgb.train(params, ds)
    return m.predict  # returns callable
