
import numpy as np
import pandas as pd
import numba
import lifelines

@numba.jit(nopython=True)
def _is_comparable(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & (d_i | d_j))

@numba.jit(nopython=True)
def _is_comparable_antolini(t_i, t_j, d_i, d_j):
    return ((t_i < t_j) & d_i) | ((t_i == t_j) & d_i & (d_j == 0))

@numba.jit(nopython=True)
def _is_concordant(s_i, s_j, t_i, t_j, d_i, d_j):
    conc = 0.
    if t_i < t_j:
        conc = (s_i < s_j) + (s_i == s_j) * 0.5
    elif t_i == t_j: 
        if d_i & d_j:
            conc = 1. - (s_i != s_j) * 0.5
        elif d_i:
            conc = (s_i < s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
        elif d_j:
            conc = (s_i > s_j) + (s_i == s_j) * 0.5  # different from RSF paper.
    return conc * _is_comparable(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True)
def _is_concordant_antolini(s_i, s_j, t_i, t_j, d_i, d_j):
    return (s_i < s_j) & _is_comparable_antolini(t_i, t_j, d_i, d_j)

@numba.jit(nopython=True, parallel=True)
def _sum_comparable(t, d, is_comparable_func, weights=None):
    n = t.shape[0]
    count = 0.
    for i in numba.prange(n):
        for j in range(n):
            if j != i:
                w = weights[i] if weights is not None else 1.
                count += w * is_comparable_func(t[i], t[j], d[i], d[j])
    return count

@numba.jit(nopython=True, parallel=True)
def _sum_concordant_disc(s, t, d, s_idx, is_concordant_func, weights=None):
    n = len(t)
    count = 0.
    for i in numba.prange(n):
        idx = s_idx[i]
        for j in range(n):
            if j != i:
                w = weights[i] if weights is not None else 1.
                count += w * is_concordant_func(s[idx, i], s[idx, j], t[i], t[j], d[i], d[j])
    return count

def concordance_td(durations, events, surv, surv_idx, method='adj_antolini', ipcw=False):
    if isinstance(surv, pd.DataFrame):
        surv = surv.values
    surv = np.ascontiguousarray(surv)
    assert durations.shape[0] == surv.shape[1] == surv_idx.shape[0] == events.shape[0]
    assert type(durations) is type(events) is type(surv) is type(surv_idx) is np.ndarray
    if events.dtype in ('float', 'float32'):
        events = events.astype('int32')

    weights = None
    if ipcw:
        from lifelines import KaplanMeierFitter
        kmf = KaplanMeierFitter().fit(durations, event_observed=(1 - events))
        g_t = np.clip(kmf.survival_function_at_times(durations).values, 1e-8, None)
        weights = 1. / g_t ** 2

    if method == 'adj_antolini':
        is_concordant = _is_concordant
        is_comparable = _is_comparable
    elif method == 'antolini':
        is_concordant = _is_concordant_antolini
        is_comparable = _is_comparable_antolini
    else:
        return ValueError(f"Need 'method' to be e.g. 'antolini', got '{method}'.")

    return (_sum_concordant_disc(surv, durations, events, surv_idx, is_concordant, weights) /
            _sum_comparable(durations, events, is_comparable, weights))
