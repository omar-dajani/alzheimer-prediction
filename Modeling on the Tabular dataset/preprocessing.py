from pathlib import Path
import pickle
import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import seaborn as sns
from neuroCombat import neuroCombat
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm.notebook import tqdm
import time
from config import RANDOM_SEED, N_FOLDS, HORIZONS, FIG_DIR, CHECKPOINT_DIR, OUT_DIR, MRI_HARMONIZE_COLS, BASE_DIR


def classify_reverters(df_all, from_state='MCI', to_state='CN'):
    """
    Classify MCI patients who reverted to CN into four trajectory groups
    (progressor, transient_noise, sustained_recovery, bouncer) based on
    their longitudinal diagnosis sequence.
    """
    mci_rids = df_all[df_all['DX_bl'] == from_state]['RID'].unique()
    groups = {'transient_noise': set(), 'sustained_recovery': set(),
              'bouncer': set(), 'progressor': set()}

    for rid in mci_rids:
        subj = df_all[df_all['RID'] == rid].sort_values('Years_bl')
        seq = subj['DX'].dropna().tolist()
        if to_state not in seq:
            continue
        # Find first reversion
        saw_from, first_idx = False, None
        for i, dx in enumerate(seq):
            if dx == from_state: saw_from = True
            elif dx == to_state and saw_from: first_idx = i; break
        if first_idx is None:
            continue
        after = seq[first_idx:]
        if 'Dementia' in seq:
            groups['progressor'].add(rid)
        elif len(after) >= 2 and after[0] == to_state and from_state in after[1:]:
            groups['transient_noise'].add(rid)
        elif seq[-1] == to_state and sum(1 for d in reversed(seq)
                                         if d == to_state or (d != to_state and False)) >= 3:
            trailing = sum(1 for d in reversed(seq)
                           if d == to_state or (_ := None) is None and d != to_state and False)
            # count trailing CN
            trailing_cn = 0
            for d in reversed(seq):
                if d == to_state: trailing_cn += 1
                else: break
            groups['sustained_recovery' if trailing_cn >= 3 else 'bouncer'].add(rid)
        else:
            groups['bouncer'].add(rid)

    return groups


def build_survival_labels(df_all, df_baseline, from_dx, to_dx,
                           exclusion_rids=None):
    '''
    Build survival (event, duration, cutoff) labels.
    exclusion_rids: set of RIDs to skip (reverters, etc.)
    '''
    if exclusion_rids is None:
        exclusion_rids = set()
    rids = df_baseline[df_baseline['DX_bl'] == from_dx]['RID'].unique()
    rids = [r for r in rids if r not in exclusion_rids]
    records = []
    for rid in rids:
        subj = df_all[df_all['RID'] == rid].sort_values('Years_bl')
        target_rows = subj[(subj['VISCODE'] != 'bl') & (subj['DX'] == to_dx)]
        if len(target_rows) > 0:
            event_time = target_rows['Years_bl'].min()
            event = 1
        else:
            event_time = subj['Years_bl'].max()
            event = 0
        if event_time <= 0:
            continue  # skip degenerate rows
        records.append({'RID': rid, 'event': event,
                         'duration': event_time, 'cutoff': event_time})
    return pd.DataFrame(records).set_index('RID')


def run_combat(df_baseline):
    """
    Batch variable       : FLDSTRENG (1.5T vs 3T)
    Protected covariates : DX_bl, AGE, PTGENDER
    NaN handling         : fill with per-feature median before ComBat,
                           restore NaN after so we don't fabricate data
    Writeback            : iloc + integer positions, no index alignment issues
    """
    has_mri   = df_baseline[MRI_HARMONIZE_COLS[0]].notna()
    has_field = df_baseline['FLDSTRENG'].notna()
    positions = np.where((has_mri & has_field).values)[0]

    print(f"  Subjects for ComBat : {len(positions)}")
    print(f"  Excluded            : {len(df_baseline) - len(positions)}")

    # Save originals across full dataframe before touching anything
    for col in MRI_HARMONIZE_COLS:
        df_baseline[f'{col}_raw'] = df_baseline[col].copy()

    df_sub  = df_baseline.iloc[positions].copy()
    data_df = df_sub[MRI_HARMONIZE_COLS].copy()

    # Track where NaNs are so we can restore them after ComBat
    nan_mask = data_df.isna()
    print(f"  NaN count in MRI matrix : {nan_mask.sum().sum()} (filled with median for ComBat)")

    # Fill NaN with per-feature median -- ComBat cannot handle NaN
    for col in MRI_HARMONIZE_COLS:
        data_df[col] = data_df[col].fillna(data_df[col].median())

    data_matrix = data_df.T.values.astype(float)
    print(f"  NaN after fill          : {np.isnan(data_matrix).sum()} (should be 0)")

    dx_dummies = pd.get_dummies(df_sub['DX_bl'], drop_first=True).astype(float)
    covariates = pd.DataFrame({
        'batch' : df_sub['FLDSTRENG'].map({'1.5 Tesla MRI': 1, '3 Tesla MRI': 2}).values,
        'AGE'   : df_sub['AGE'].fillna(df_sub['AGE'].median()).values,
        'GENDER': (df_sub['PTGENDER'] == 'Male').astype(int).values,
    })
    for col in dx_dummies.columns:
        covariates[str(col)] = dx_dummies[col].values

    print("  Running neuroCombat...")
    combat_result   = neuroCombat(dat=data_matrix, covars=covariates, batch_col='batch')
    data_harmonized = combat_result['data']
    print(f"  Output shape  : {data_harmonized.shape}")
    print(f"  Output sample : {data_harmonized[0, :3].round(1)}")

    # Write back with iloc -- restore original NaN positions
    for i, col in enumerate(MRI_HARMONIZE_COLS):
        harmonized_vals = data_harmonized[i, :].copy()
        harmonized_vals[nan_mask[col].values] = np.nan
        df_baseline.iloc[positions, df_baseline.columns.get_loc(col)] = harmonized_vals

    n_nonnull = df_baseline[MRI_HARMONIZE_COLS[0]].notna().sum()
    print(f"  Non-null Hippocampus after writeback : {n_nonnull} (should be {len(positions)})")
    return df_baseline

    
def plot_before_after(df_baseline, feature='Hippocampus_ICV'):
    """
    Plot 2×2 histograms comparing ICV-normalized MRI feature distributions
    before and after ComBat harmonization, stratified by ADNI phase and
    field strength.

    Requires that ComBat was applied and the pre-harmonization values were
    stored in a '<feature_base>_raw' column. Silently skips if ComBat was
    not run.

    Args:
        df_baseline (pd.DataFrame): Baseline DataFrame containing both raw
            and harmonized feature columns, plus 'COLPROT', 'FLDSTRENG',
            and 'ICV'.
        feature (str): ICV-normalized column name to visualize, e.g.
            'Hippocampus_ICV' or 'Entorhinal_ICV'. Default 'Hippocampus_ICV'.

    Returns:
        None. Saves a 2×2 figure to FIG_DIR as
        'combat_before_after_<feature_base>.png'.
    """
    base    = feature.replace('_ICV', '')
    raw_col = f'{base}_raw_ICV'

    if f'{base}_raw' not in df_baseline.columns:
        print(f"No _raw column -- ComBat was skipped.")
        return

    mask = df_baseline[f'{base}_raw'].notna()
    df_baseline.loc[mask, raw_col] = (
        df_baseline.loc[mask, f'{base}_raw'] / df_baseline.loc[mask, 'ICV']
    )

    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))

    for row, (strat_col, strat_label) in enumerate([
        ('COLPROT',   'ADNI Phase'),
        ('FLDSTRENG', 'Field Strength'),
    ]):
        for col_idx, (plot_col, title) in enumerate([
            (raw_col, f'BEFORE -- by {strat_label}'),
            (feature,  f'AFTER  -- by {strat_label}'),
        ]):
            ax = axes[row, col_idx]
            sub = df_baseline.dropna(subset=[plot_col, strat_col])
            for i, (name, grp) in enumerate(sub.groupby(strat_col)):
                ax.hist(grp[plot_col], bins=40, alpha=0.55,
                        color=palette[i % 4], label=str(name))
            ax.set(xlabel=feature, title=title)
            ax.legend(fontsize=8)

    plt.suptitle(f'ComBat: {feature}  |  scanner variance removed, biology preserved', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'combat_before_after_{base}.png', dpi=150, bbox_inches='tight')
    plt.show()


def harmonization_report(df_baseline):
    """
    Print a statistical validation report for ComBat MRI harmonization,
    comparing mean Hippocampus/ICV values before and after correction.

    Reports include:
    - Per-ADNI-phase mean before/after with delta
    - Per-field-strength mean before/after
    - 1.5T vs 3T gap size and percentage reduction
    - Kruskal-Wallis test across phases before and after

    Note: Residual significance after ComBat is expected and reflects real
    biological differences between ADNI cohorts (ADNI1 was sicker than ADNI3),
    not scanner artefacts.

    Args:
        df_baseline (pd.DataFrame): Baseline DataFrame. Must contain
            'Hippocampus_raw' column (set during ComBat); silently exits
            if ComBat was not applied.

    Returns:
        None. Prints report to stdout.
    """
    if 'Hippocampus_raw' not in df_baseline.columns:
        print("ComBat was not applied -- nothing to validate.")
        return

    mask = df_baseline['Hippocampus_raw'].notna()
    df_baseline.loc[mask, 'Hippocampus_raw_ICV'] = (
        df_baseline.loc[mask, 'Hippocampus_raw'] / df_baseline.loc[mask, 'ICV']
    )

    print("=" * 55)
    print("HARMONIZATION REPORT: Mean Hippocampus/ICV by phase")
    print("=" * 55)
    comp = pd.DataFrame({
        'BEFORE': df_baseline.groupby('COLPROT')['Hippocampus_raw_ICV'].mean(),
        'AFTER' : df_baseline.groupby('COLPROT')['Hippocampus_ICV'].mean(),
    }).round(6)
    comp['delta'] = (comp['AFTER'] - comp['BEFORE']).round(6)
    print(comp.to_string())

    print("\nMean Hippocampus/ICV by field strength:")
    comp2 = pd.DataFrame({
        'BEFORE': df_baseline.groupby('FLDSTRENG')['Hippocampus_raw_ICV'].mean(),
        'AFTER' : df_baseline.groupby('FLDSTRENG')['Hippocampus_ICV'].mean(),
    }).round(6)
    print(comp2.to_string())

    if '1.5 Tesla MRI' in comp2.index and '3 Tesla MRI' in comp2.index:
        gap_before = abs(comp2.loc['1.5 Tesla MRI', 'BEFORE'] - comp2.loc['3 Tesla MRI', 'BEFORE'])
        gap_after  = abs(comp2.loc['1.5 Tesla MRI', 'AFTER']  - comp2.loc['3 Tesla MRI', 'AFTER'])
        pct = (1 - gap_after / gap_before) * 100 if gap_before > 0 else float('nan')
        print(f"\n  1.5T vs 3T gap BEFORE : {gap_before:.6f}")
        print(f"  1.5T vs 3T gap AFTER  : {gap_after:.6f}")
        print(f"  Gap reduction         : {pct:.1f}%")

    print("\nKruskal-Wallis across ADNI phases:")
    for col, label in [('Hippocampus_raw_ICV', 'BEFORE'), ('Hippocampus_ICV', 'AFTER')]:
        groups = [
            g[col].dropna()
            for _, g in df_baseline.dropna(subset=[col]).groupby('COLPROT')
            if len(g) > 5
        ]
        if len(groups) >= 2:
            stat, p = stats.kruskal(*groups)
            note = '  <- residual bio variance (expected)' if label == 'AFTER' and p < 0.05 else ''
            print(f"  {label}: H={stat:.1f}, p={p:.4f}{note}")

    print("""
NOTE: Residual significance across ADNI phases after ComBat is expected.
ADNI1 = LMCI/AD heavy (sicker); ADNI3 = EMCI/SMC heavy (healthier).
That mean difference is real biology and should remain.
What matters is the 1.5T vs 3T gap reduction -- target 30-70%.
""")


def compute_slopes_cutoff(df_all, surv_labels, features, min_visits=2):
    '''
    Per-subject OLS slopes strictly from pre-cutoff visits.
    Returns DataFrame: slope_<feat>, slope_velocity_<feat>

    NOTE: n_visits_used, pre_conversion_span_yr, visit_regularity are computed
    here for diagnostics ONLY and are NOT returned as model features — they
    encode event timing and would cause data leakage.
    '''
    results = {}
    for rid, row in surv_labels.iterrows():
        cutoff, is_event = row['cutoff'], row['event'] == 1
        subj = df_all[df_all['RID'] == rid].sort_values('Years_bl')
        valid = (subj[subj['Years_bl'] <  cutoff] if is_event
                 else subj[subj['Years_bl'] <= cutoff])

        res = {}  # NO meta cols added here

        for feat in features:
            dat = valid[['Years_bl', feat]].dropna()
            if len(dat) >= min_visits and dat['Years_bl'].nunique() > 1:
                m, b, _, _, _ = stats.linregress(dat['Years_bl'], dat[feat])
                res[f'slope_{feat}'] = m
                # Slope velocity: split into first-half / second-half
                if len(dat) >= 4:
                    mid = dat['Years_bl'].median()
                    first  = dat[dat['Years_bl'] <= mid]
                    second = dat[dat['Years_bl'] >  mid]
                    if len(first) >= 2 and len(second) >= 2:
                        m1, *_ = stats.linregress(first['Years_bl'],  first[feat])
                        m2, *_ = stats.linregress(second['Years_bl'], second[feat])
                        res[f'slope_velocity_{feat}'] = m2 - m1
                    else:
                        res[f'slope_velocity_{feat}'] = np.nan
                else:
                    res[f'slope_velocity_{feat}'] = np.nan
            else:
                res[f'slope_{feat}'] = np.nan
                res[f'slope_velocity_{feat}'] = np.nan

        results[rid] = res

    out = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'RID'})
    out['RID'] = out['RID'].astype(int)
    return out

    
def longitudinal_fill(df_all, features, window_yr=1.0):
    """
    Tier-1 imputation: fill missing values using the nearest longitudinal
    observation within a ±window_yr time window for each subject.

    Args:
        df_all (pd.DataFrame): Full longitudinal DataFrame with 'RID' and 'Years_bl'.
        features (list[str]): Column names to impute.
        window_yr (float): Maximum time gap (years) to borrow a value from. Default 1.0.

    Returns:
        pd.DataFrame: Copy of df_all with NaNs filled where a nearby observation exists.
    """
    df_out = df_all.copy()
    for rid, grp in df_all.groupby('RID'):
        grp = grp.sort_values('Years_bl')
        idx = grp.index
        times = grp['Years_bl'].values
        for feat in features:
            if feat not in df_out.columns:
                continue
            vals = df_out.loc[idx, feat].values.copy()
            for i in range(len(vals)):
                if pd.isna(vals[i]):
                    diffs = np.abs(times - times[i])
                    diffs[i] = np.inf  # exclude self
                    j = np.argmin(diffs)
                    if diffs[j] <= window_yr and not pd.isna(vals[j]):
                        vals[i] = vals[j]
            df_out.loc[idx, feat] = vals
    return df_out

def mice_impute(X_df, max_iter=10, seed=RANDOM_SEED):
    """
    Tier-2 imputation: apply MICE (Multiple Imputation by Chained Equations)
    via sklearn's IterativeImputer to fill any remaining NaNs after longitudinal fill.

    Args:
        X_df (pd.DataFrame): Feature matrix with remaining NaN values.
        max_iter (int): Maximum MICE iterations. Default 10.
        seed (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Fully imputed DataFrame with same shape, columns, and index as X_df.
    """
    imp = IterativeImputer(max_iter=max_iter, random_state=seed)
    arr = imp.fit_transform(X_df)
    return pd.DataFrame(arr, columns=X_df.columns, index=X_df.index)

def assemble_cohort(df_baseline, surv_labels, slopes_df, core_features, slope_all_cols):
    """
    Join baseline features, survival labels, and longitudinal slopes into a
    single modeling-ready DataFrame for one cohort.

    Args:
        df_baseline (pd.DataFrame): Baseline visit data with 'RID' column.
        surv_labels (pd.DataFrame): Survival labels indexed by RID with 'event' and 'duration'.
        slopes_df (pd.DataFrame): Per-subject slope features with 'RID' column.
        core_features (list[str]): Baseline feature columns to include.
        slope_all_cols (list[str]): Slope feature column names to merge in.

    Returns:
        tuple: (X, y_event, y_duration, rids)
            X (pd.DataFrame): Feature matrix.
            y_event (np.ndarray): Binary event indicators.
            y_duration (np.ndarray): Time-to-event or censoring in years.
            rids (list): Subject IDs in row order.
    """
    merged = (
        df_baseline.set_index('RID')
        .join(surv_labels[['event','duration']], how='inner')
        .join(slopes_df.set_index('RID')[
            [c for c in slope_all_cols if c in slopes_df.columns]
        ], how='left')
    )
    merged = merged[merged['duration'] > 0]
    all_feat = [f for f in core_features + slope_all_cols if f in merged.columns]
    all_feat = list(dict.fromkeys(all_feat))  # deduplicate preserving order
    return (
        merged[all_feat].copy(),
        merged['event'].values.astype(int),
        merged['duration'].values.astype(float),
        merged.index.tolist(),
    )

def add_slope_concordance(X):
    """
    Add a binary concordance feature indicating simultaneous cognitive and
    structural decline — both MMSE slope and Hippocampus slope are negative.

    Subjects showing decline on both axes are considered to have concordant
    biomarker progression, which is a stronger signal for conversion risk.

    Args:
        X (pd.DataFrame): Feature matrix containing 'slope_MMSE' and
            'slope_Hippocampus' columns (if available).

    Returns:
        pd.DataFrame: Copy of X with an added 'slope_concordance' column
            (1.0 = both declining, 0.0 = otherwise).
    """
    X = X.copy()
    cog_decline = (X['slope_MMSE'] < 0).astype(float) if 'slope_MMSE' in X.columns else 0
    mri_decline = (X['slope_Hippocampus'] < 0).astype(float) if 'slope_Hippocampus' in X.columns else 0
    X['slope_concordance'] = cog_decline * mri_decline
    return X

def get_domain_features(feature_names):
    """
    Partition the full feature list into domain-specific subsets for
    modality-separated modeling experiments.

    Demographics, APOE4, missingness flags, and protocol dummies are
    appended to every non-combined domain as a shared base set.

    Args:
        feature_names (list[str]): Full list of feature column names
            for a given cohort (e.g. feature_names_mci).

    Returns:
        dict: Four keys mapping to feature lists:
            'imaging'  — ICV-normalized MRI volumes + base features.
            'csf_pet'  — CSF biomarkers, PET, amyloid/tau composites + base.
            'cognitive'— Cognitive scores, composites, Ecog + base.
            'combined' — All features (no filtering).
    """
    domains = {
        'imaging': [f for f in feature_names if any(x in f for x in
            ['Hippocampus','Entorhinal','Ventricles','Fusiform',
             'MidTemp','WholeBrain','hippo','midtemp'])],
        'csf_pet': [f for f in feature_names if any(x in f for x in
            ['ABETA','TAU','PTAU','FDG','AV45','FBB',
             'amyloid','neurodegeneration','ATN'])],
        'cognitive': [f for f in feature_names if any(x in f for x in
            ['MMSE','CDRSB','ADAS','LDELTOTAL','RAVLT','FAQ','MOCA',
             'mPACC','cog_','severity','Ecog','ecog'])],
        'combined': feature_names,  # all features
    }
    # Always add demographics + APOE4 to every domain
    base = ['AGE','PTGENDER_num','PTEDUCAT','APOE4','n_modalities',
            *[f for f in feature_names if f.startswith('miss_')],
            *[f for f in feature_names if f.startswith('prot_') or f.startswith('field_')]]
    for d in ['imaging','csf_pet','cognitive']:
        combined = list(dict.fromkeys(base + domains[d]))
        domains[d] = [f for f in combined if f in feature_names]
    return domains


def test_tqdm():
    """
    Smoke test to verify tqdm progress bars render correctly in the current
    environment (Colab notebook vs local terminal).

    Runs a short dummy loop with a tqdm progress bar. No side effects.

    Returns:
        None. Prints a confirmation message on success.
    """
    import time
    for i in tqdm(range(20), desc="Testing bar"):
        time.sleep(0.2)
