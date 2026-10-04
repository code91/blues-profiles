#!/usr/bin/env python3
"""
09_granger_causality.py

Granger causality analysis for decision-making profiles.

For each artist (pooled across solos):
1. Time series: density(t), dissonance(t), anticipation(t) at phrase level
2. Stationarity test (ADF)
3. Lag selection (AIC/BIC)
4. Granger test both directions for two pairs:
   - Density ↔ Dissonance
   - Density ↔ Anticipation
5. Compute "Gravity" scores

Output:
- data/analysis/granger_results.csv
- data/analysis/granger_stats.json
- data/figures/granger_gravity_scores.png
"""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR

warnings.filterwarnings('ignore')

# Configuration
TIMESERIES_DIR = Path("data/timeseries")
ANALYSIS_DIR = Path("data/analysis")
FIGURES_DIR = Path("data/figures")

MAX_LAG = 4       # upper bound for the sensitivity sweep
GRANGER_LAG = 1   # fixed a priori; granger_lag_sweep() reports the alternatives
CONTROL_FOR_TEMPO = True  # Whether to residualize metrics against tempo

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def residualize_against_tempo(df, columns, tempo_col='tempo'):
    """
    Residualize columns against tempo using OLS regression.

    For each column, fits: column ~ tempo
    Returns residuals, which represent the column's variance
    unexplained by tempo.

    This removes tempo-related variance before Granger testing,
    ensuring any detected causality isn't confounded by tempo.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns or tempo_col not in df.columns:
            continue

        # Get non-null data
        mask = df[col].notna() & df[tempo_col].notna()
        if mask.sum() < 10:
            continue

        y = df.loc[mask, col].values
        X = df.loc[mask, tempo_col].values.reshape(-1, 1)

        # Add constant for intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y, rcond=None)
            fitted = X_with_const @ coeffs
            resid = y - fitted

            # Store residuals back
            df.loc[mask, f'{col}_resid'] = resid
        except:
            df[f'{col}_resid'] = df[col]

    return df


def compute_dissonance_from_iv(iv):
    """Dissonance = ic1 + 0.5*ic2 + 0.8*ic6"""
    ic1, ic2, ic3, ic4, ic5, ic6 = iv
    return ic1 + 0.5 * ic2 + 0.8 * ic6


# Blues IVs catalog (from script 9)
BLUES_IVS = {
    # TRICHORDS
    (0, 1, 1, 0, 1, 0): 3, (0, 2, 0, 1, 0, 0): 3, (2, 1, 0, 0, 0, 0): 3,
    (1, 0, 0, 0, 1, 1): 3, (0, 0, 2, 0, 0, 1): 3, (0, 1, 0, 0, 2, 0): 3,
    # TETRACHORDS
    (0, 2, 1, 1, 2, 0): 4, (0, 1, 2, 1, 2, 0): 4, (0, 2, 1, 0, 3, 0): 4,
    (1, 0, 2, 2, 1, 0): 4, (1, 1, 1, 1, 2, 0): 4,
    # PENTACHORDS
    (0, 3, 2, 1, 4, 0): 5, (2, 2, 2, 1, 2, 1): 5, (2, 2, 2, 2, 2, 0): 5, (1, 3, 2, 1, 4, 0): 5,
    # HEXATONIC
    (2, 3, 3, 2, 4, 1): 6,
}


def load_timeseries_with_all_metrics():
    """Load time series data and add anticipation and bluesiness."""
    # Load base time series
    ts_df = pd.read_csv(TIMESERIES_DIR / 'timeseries_data.csv')

    # Load anticipation
    antic_path = ANALYSIS_DIR / 'anticipation_metrics.csv'
    if antic_path.exists():
        antic_df = pd.read_csv(antic_path)
        ts_df = ts_df.merge(
            antic_df[['melid', 'phrase_id', 'loading_balance']],
            on=['melid', 'phrase_id'],
            how='left'
        )
        ts_df['anticipation'] = ts_df['loading_balance'].fillna(0)
    else:
        ts_df['anticipation'] = 0

    # Load phrase IVs for bluesiness
    phrases_dir = Path("data/phrases")
    iv_lookup = {}
    for filepath in phrases_dir.glob("*.json"):
        if filepath.name == "summary.csv":
            continue
        with open(filepath) as f:
            data = json.load(f)
        for phrase in data['phrases']:
            iv_lookup[(data['melid'], phrase['phrase_id'])] = tuple(phrase['iv'])

    # Add bluesiness metrics
    ts_df['iv_tuple'] = ts_df.apply(lambda row: iv_lookup.get((row['melid'], row['phrase_id']), None), axis=1)
    ts_df['is_blues'] = ts_df['iv_tuple'].apply(lambda iv: 1 if iv in BLUES_IVS else 0)
    # Binary, deliberately. The BLUES_IVS values are the cardinality of each
    # catalogue set, so the weighted variable used previously was
    # cardinality x 1[iv in catalogue] -- it carried the pitch-class count
    # inside it, and density predicting it was partly density predicting
    # density. The cardinality is kept separately for description only.
    ts_df['bluesiness'] = ts_df['is_blues']
    ts_df['blues_cardinality'] = ts_df['iv_tuple'].apply(lambda iv: BLUES_IVS.get(iv, 0))

    return ts_df


def adf_test(series, name=''):
    """Augmented Dickey-Fuller test for stationarity."""
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'series': name,
        'adf_statistic': float(result[0]),
        'p_value': float(result[1]),
        'lags_used': int(result[2]),
        'n_obs': int(result[3]),
        'critical_1%': float(result[4]['1%']),
        'critical_5%': float(result[4]['5%']),
        'critical_10%': float(result[4]['10%']),
        'stationary': result[1] < 0.05
    }


def select_lag(data, maxlag=MAX_LAG):
    """Select optimal lag using VAR and AIC."""
    try:
        model = VAR(data)
        results = model.select_order(maxlags=maxlag)
        return results.aic
    except:
        return 1


def _design(data, cause_col, effect_col, lag, group_col='melid'):
    """Lagged design for a Granger test, built WITHIN each solo.

    statsmodels' grangercausalitytests() takes one array and lags it blindly,
    which across a concatenated artist means the last phrases of one solo are
    used to predict the first phrases of the next. Those transitions are not
    musical continuations and must not be in the design. Building the lags per
    group and stacking is the only way to exclude them, and it is why this test
    is implemented here rather than called from statsmodels.

    Returns (y, X_restricted, X_unrestricted) or None when too little data.
    """
    ys, xr, xu = [], [], []
    for _, grp in data.groupby(group_col, sort=False):
        g = grp[[effect_col, cause_col]].dropna()
        if len(g) < lag + 2:
            continue
        eff = g[effect_col].to_numpy(dtype=float)
        cau = g[cause_col].to_numpy(dtype=float)
        for t in range(lag, len(g)):
            ys.append(eff[t])
            own = [eff[t - k] for k in range(1, lag + 1)]
            oth = [cau[t - k] for k in range(1, lag + 1)]
            xr.append([1.0] + own)
            xu.append([1.0] + own + oth)
    if len(ys) < 2 * lag + 5:
        return None
    return np.array(ys), np.array(xr), np.array(xu)


def _ols_rss(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def granger_test(data, cause_col, effect_col, lag=GRANGER_LAG, group_col='melid'):
    """Does cause_col Granger-cause effect_col, at a FIXED lag?

    The lag is fixed a priori rather than chosen per test. The previous version
    searched lags 1..4 and kept whichever gave the smallest p-value, which is a
    selection over four tests and inflates significance on top of the multiple
    comparisons already corrected for downstream. Sensitivity across lags is
    reported separately by granger_lag_sweep().
    """
    built = _design(data, cause_col, effect_col, lag, group_col)
    if built is None:
        return {'cause': cause_col, 'effect': effect_col, 'f_statistic': np.nan,
                'p_value': np.nan, 'lag': lag, 'n_obs': 0,
                'significant': False, 'error': 'Insufficient data'}
    y, Xr, Xu = built
    try:
        rss_r, rss_u = _ols_rss(y, Xr), _ols_rss(y, Xu)
        df_num = Xu.shape[1] - Xr.shape[1]
        df_den = len(y) - Xu.shape[1]
        if df_den <= 0 or rss_u <= 0:
            raise ValueError('degenerate fit')
        f = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        p = float(1.0 - scipy_stats.f.cdf(f, df_num, df_den))
        return {'cause': cause_col, 'effect': effect_col, 'f_statistic': float(f),
                'p_value': p, 'lag': lag, 'n_obs': int(len(y)),
                'significant': bool(p < 0.05), 'error': None}
    except Exception as e:
        return {'cause': cause_col, 'effect': effect_col, 'f_statistic': np.nan,
                'p_value': np.nan, 'lag': lag, 'n_obs': int(len(y)),
                'significant': False, 'error': str(e)}


def granger_lag_sweep(data, cause_col, effect_col, lags=(1, 2, 3, 4), group_col='melid'):
    """The same test at every lag, so the reader can see what the choice costs."""
    return {lag: granger_test(data, cause_col, effect_col, lag, group_col)
            for lag in lags}


def compute_gravity(result_forward, result_backward):
    """
    Compute gravity score from bidirectional Granger tests.

    Gravity = direction and strength of causal relationship
    Positive = forward causation (proactive)
    Negative = backward causation (reactive)

    Score based on F-statistic difference, weighted by significance.
    """
    f_fwd = result_forward['f_statistic'] if not np.isnan(result_forward['f_statistic']) else 0
    f_bwd = result_backward['f_statistic'] if not np.isnan(result_backward['f_statistic']) else 0

    p_fwd = result_forward['p_value'] if not np.isnan(result_forward['p_value']) else 1
    p_bwd = result_backward['p_value'] if not np.isnan(result_backward['p_value']) else 1

    # Weight by significance (transform p-value to 0-1 weight)
    weight_fwd = 1 - p_fwd if p_fwd < 0.1 else 0
    weight_bwd = 1 - p_bwd if p_bwd < 0.1 else 0

    # Gravity score: positive = forward dominates, negative = backward dominates
    if weight_fwd + weight_bwd == 0:
        gravity = 0
    else:
        gravity = (f_fwd * weight_fwd - f_bwd * weight_bwd) / max(f_fwd + f_bwd, 1)

    # Classify direction
    if result_forward['significant'] and not result_backward['significant']:
        direction = 'PROACTIVE'
    elif result_backward['significant'] and not result_forward['significant']:
        direction = 'REACTIVE'
    elif result_forward['significant'] and result_backward['significant']:
        direction = 'BIDIRECTIONAL'
    else:
        direction = 'NONE'

    return {
        'gravity_score': float(gravity),
        'direction': direction,
        'forward_significant': result_forward['significant'],
        'backward_significant': result_backward['significant']
    }


def analyze_artist(df, performer):
    """Run full Granger analysis for one artist."""
    artist_df = df[df['performer'] == performer].copy()
    artist_df = artist_df.sort_values(['melid', 'phrase_id']).reset_index(drop=True)

    results = {'performer': performer, 'n_phrases': len(artist_df)}

    # Residualize against tempo if enabled
    if CONTROL_FOR_TEMPO and 'tempo' in artist_df.columns:
        metrics_to_residualize = ['density', 'dissonance_ratio', 'entropy',
                                  'anticipation', 'n_notes', 'bluesiness']
        artist_df = residualize_against_tempo(artist_df, metrics_to_residualize)

        # Use residualized columns for Granger tests
        density_col = 'density_resid' if 'density_resid' in artist_df.columns else 'density'
        # Raw dissonance is a weighted sub-sum of the six components that make up
        # density (r = 0.991 on this corpus, VIF ~ 59), so Granger between the two
        # is close to self-prediction. The ratio removes the shared size component
        # and brings the correlation down to 0.088. Same correction the companion
        # Parker study made under review.
        dissonance_col = 'dissonance_ratio_resid' if 'dissonance_ratio_resid' in artist_df.columns else 'dissonance_ratio'
        entropy_col = 'entropy_resid' if 'entropy_resid' in artist_df.columns else 'entropy'
        anticipation_col = 'anticipation_resid' if 'anticipation_resid' in artist_df.columns else 'anticipation'
        n_notes_col = 'n_notes_resid' if 'n_notes_resid' in artist_df.columns else 'n_notes'
        bluesiness_col = 'bluesiness_resid' if 'bluesiness_resid' in artist_df.columns else 'bluesiness'

        results['tempo_controlled'] = True
    else:
        density_col = 'density'
        dissonance_col = 'dissonance_ratio'
        entropy_col = 'entropy'
        anticipation_col = 'anticipation'
        n_notes_col = 'n_notes'
        bluesiness_col = 'bluesiness'
        results['tempo_controlled'] = False

    # Stationarity tests (on original metrics for interpretability)
    # Phrase length is tested too; bluesiness is a 0/1 indicator, for which a unit
    # root is not a meaningful hypothesis, so it is reported as bounded instead.
    for col in ['density', 'dissonance_ratio', 'entropy', 'anticipation', 'n_notes']:
        adf = adf_test(artist_df[col], col)
        results[f'adf_{col}'] = adf

    # === Density ↔ Dissonance ===
    # Forward: density → dissonance (proactive)
    fwd_cd = granger_test(artist_df, density_col, dissonance_col)
    # Backward: dissonance → density (reactive)
    bwd_cd = granger_test(artist_df, dissonance_col, density_col)

    gravity_cd = compute_gravity(fwd_cd, bwd_cd)

    results['density_dissonance'] = {
        'forward': fwd_cd,  # density → dissonance
        'backward': bwd_cd,  # dissonance → density
        'gravity': gravity_cd
    }

    # === Density ↔ Anticipation ===
    # Forward: density → anticipation
    fwd_ca = granger_test(artist_df, density_col, anticipation_col)
    # Backward: anticipation → density
    bwd_ca = granger_test(artist_df, anticipation_col, density_col)

    gravity_ca = compute_gravity(fwd_ca, bwd_ca)

    results['density_anticipation'] = {
        'forward': fwd_ca,  # density → anticipation
        'backward': bwd_ca,  # anticipation → density
        'gravity': gravity_ca
    }

    # === Phrase Length ↔ Density ===
    # Forward: phrase_length → density (does length predict next density?)
    fwd_lc = granger_test(artist_df, n_notes_col, density_col)
    # Backward: density → phrase_length (does density predict next length?)
    bwd_lc = granger_test(artist_df, density_col, n_notes_col)

    gravity_lc = compute_gravity(fwd_lc, bwd_lc)

    results['length_density'] = {
        'forward': fwd_lc,  # length → density
        'backward': bwd_lc,  # density → length
        'gravity': gravity_lc
    }

    # === Density → Bluesiness ===
    fwd_cb = granger_test(artist_df, density_col, bluesiness_col)
    bwd_cb = granger_test(artist_df, bluesiness_col, density_col)
    gravity_cb = compute_gravity(fwd_cb, bwd_cb)

    results['density_bluesiness'] = {
        'forward': fwd_cb,  # density → bluesiness
        'backward': bwd_cb,  # bluesiness → density
        'gravity': gravity_cb
    }

    # Sensitivity: the same three headline pairs at every lag, so the fixed
    # choice of lag 1 can be checked rather than taken on trust.
    results['lag_sweep'] = {
        'density_bluesiness': granger_lag_sweep(artist_df, density_col, bluesiness_col),
        'length_bluesiness': granger_lag_sweep(artist_df, n_notes_col, bluesiness_col),
        'density_anticipation': granger_lag_sweep(artist_df, density_col, anticipation_col),
    }

    # === Interval entropy ↔ Bluesiness ===
    # Entropy is the density measure density is not: correlation with density
    # is 0.545, so this pair is not the near-tautology the old one was.
    fwd_eb = granger_test(artist_df, entropy_col, bluesiness_col)
    bwd_eb = granger_test(artist_df, bluesiness_col, entropy_col)
    gravity_eb = compute_gravity(fwd_eb, bwd_eb)

    results['entropy_bluesiness'] = {
        'forward': fwd_eb,
        'backward': bwd_eb,
        'gravity': gravity_eb
    }

    # === Dissonance → Bluesiness ===
    fwd_db = granger_test(artist_df, dissonance_col, bluesiness_col)
    bwd_db = granger_test(artist_df, bluesiness_col, dissonance_col)
    gravity_db = compute_gravity(fwd_db, bwd_db)

    results['dissonance_bluesiness'] = {
        'forward': fwd_db,  # dissonance → bluesiness
        'backward': bwd_db,  # bluesiness → dissonance
        'gravity': gravity_db
    }

    # === Phrase Length → Bluesiness ===
    fwd_lb = granger_test(artist_df, n_notes_col, bluesiness_col)
    bwd_lb = granger_test(artist_df, bluesiness_col, n_notes_col)
    gravity_lb = compute_gravity(fwd_lb, bwd_lb)

    results['length_bluesiness'] = {
        'forward': fwd_lb,  # length → bluesiness
        'backward': bwd_lb,  # bluesiness → length
        'gravity': gravity_lb
    }

    return results


def plot_gravity_scores(all_results):
    """Plot gravity scores for all pairs."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Extract data
    performers = []
    gravity_cd = []
    gravity_ca = []
    gravity_lc = []
    gravity_cb = []
    gravity_db = []
    gravity_lb = []
    direction_cd = []
    direction_ca = []
    direction_lc = []
    direction_cb = []
    direction_db = []
    direction_lb = []

    for r in all_results:
        performers.append(r['performer'])
        gravity_cd.append(r['density_dissonance']['gravity']['gravity_score'])
        gravity_ca.append(r['density_anticipation']['gravity']['gravity_score'])
        gravity_lc.append(r['length_density']['gravity']['gravity_score'])
        gravity_cb.append(r['density_bluesiness']['gravity']['gravity_score'])
        gravity_db.append(r['dissonance_bluesiness']['gravity']['gravity_score'])
        gravity_lb.append(r['length_bluesiness']['gravity']['gravity_score'])
        direction_cd.append(r['density_dissonance']['gravity']['direction'])
        direction_ca.append(r['density_anticipation']['gravity']['direction'])
        direction_lc.append(r['length_density']['gravity']['direction'])
        direction_cb.append(r['density_bluesiness']['gravity']['direction'])
        direction_db.append(r['dissonance_bluesiness']['gravity']['direction'])
        direction_lb.append(r['length_bluesiness']['gravity']['direction'])

    # Colors by direction
    color_map = {
        'PROACTIVE': '#4575b4',
        'REACTIVE': '#d73027',
        'BIDIRECTIONAL': '#fdae61',
        'NONE': '#999999'
    }

    y_pos = np.arange(len(performers))

    # Plot configs
    plots = [
        (gravity_cd, direction_cd, 'Density ↔ Dissonance', '← Reactive | Proactive →'),
        (gravity_ca, direction_ca, 'Density ↔ Anticipation', '← Anticipation drives | Density drives →'),
        (gravity_lc, direction_lc, 'Length ↔ Density', '← Density drives | Length drives →'),
        (gravity_cb, direction_cb, 'Density → Bluesiness', '← Blues drives | Density drives →'),
        (gravity_db, direction_db, 'Dissonance → Bluesiness', '← Blues drives | Dissonance drives →'),
        (gravity_lb, direction_lb, 'Length → Bluesiness', '← Blues drives | Length drives →'),
    ]

    for ax, (gravity, direction, title, xlabel) in zip(axes, plots):
        sorted_idx = np.argsort(gravity)
        colors = [color_map[direction[i]] for i in sorted_idx]
        ax.barh(y_pos, [gravity[i] for i in sorted_idx], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([performers[i] for i in sorted_idx], fontsize=8)
        ax.axvline(x=0, color='black', linewidth=1)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['PROACTIVE'], label='Forward causal'),
        Patch(facecolor=color_map['REACTIVE'], label='Backward causal'),
        Patch(facecolor=color_map['BIDIRECTIONAL'], label='Bidirectional'),
        Patch(facecolor=color_map['NONE'], label='No causality')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)

    plt.suptitle('Granger Causality: Gravity Scores by Artist', fontsize=13, y=1.06)
    plt.tight_layout()

    return fig


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading time series data...")
    df = load_timeseries_with_all_metrics()
    print(f"Total phrases: {len(df)}")
    print(f"Artists: {df['performer'].nunique()}")

    # Run analysis for each artist
    print("\n=== GRANGER CAUSALITY ANALYSIS ===")
    all_results = []

    for performer in sorted(df['performer'].unique()):
        print(f"\nAnalyzing: {performer}")
        results = analyze_artist(df, performer)
        all_results.append(results)

        # Print summary
        cd = results['density_dissonance']['gravity']
        ca = results['density_anticipation']['gravity']
        lc = results['length_density']['gravity']
        cb = results['density_bluesiness']['gravity']
        db = results['dissonance_bluesiness']['gravity']
        lb = results['length_bluesiness']['gravity']
        eb = results['entropy_bluesiness']['gravity']
        print(f"  Density↔Dissonance:   {cd['direction']:12} (gravity={cd['gravity_score']:+.3f})")
        print(f"  Density↔Anticipation: {ca['direction']:12} (gravity={ca['gravity_score']:+.3f})")
        print(f"  Length↔Density:       {lc['direction']:12} (gravity={lc['gravity_score']:+.3f})")
        print(f"  Density→Bluesiness:   {cb['direction']:12} (gravity={cb['gravity_score']:+.3f})")
        print(f"  Dissonance→Bluesiness:   {db['direction']:12} (gravity={db['gravity_score']:+.3f})")
        print(f"  Length→Bluesiness:       {lb['direction']:12} (gravity={lb['gravity_score']:+.3f})")
        print(f"  Entropy↔Bluesiness:   {eb['direction']:12} (gravity={eb['gravity_score']:+.3f})")

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Artist':20} {'C↔D':8} {'C↔A':8} {'L↔C':8} {'C→B':8} {'D→B':8} {'L→B':8}")
    print("-" * 75)

    summary_rows = []
    for r in all_results:
        cd = r['density_dissonance']['gravity']
        ca = r['density_anticipation']['gravity']
        lc = r['length_density']['gravity']
        cb = r['density_bluesiness']['gravity']
        db = r['dissonance_bluesiness']['gravity']
        lb = r['length_bluesiness']['gravity']
        eb = r['entropy_bluesiness']['gravity']

        print(f"{r['performer']:20} {cd['direction'][:7]:8} {ca['direction'][:7]:8} {lc['direction'][:7]:8} "
              f"{cb['direction'][:7]:8} {db['direction'][:7]:8} {lb['direction'][:7]:8}")

        summary_rows.append({
            'performer': r['performer'],
            'n_phrases': r['n_phrases'],
            'cd_direction': cd['direction'],
            'cd_gravity': cd['gravity_score'],
            'cd_forward_sig': cd['forward_significant'],
            'cd_backward_sig': cd['backward_significant'],
            'ca_direction': ca['direction'],
            'ca_gravity': ca['gravity_score'],
            'ca_forward_sig': ca['forward_significant'],
            'ca_backward_sig': ca['backward_significant'],
            'lc_direction': lc['direction'],
            'lc_gravity': lc['gravity_score'],
            'lc_forward_sig': lc['forward_significant'],
            'lc_backward_sig': lc['backward_significant'],
            'cb_direction': cb['direction'],
            'cb_gravity': cb['gravity_score'],
            'cb_forward_sig': cb['forward_significant'],
            'cb_backward_sig': cb['backward_significant'],
            'eb_direction': eb['direction'],
            'eb_gravity': eb['gravity_score'],
            'eb_forward_sig': eb['forward_significant'],
            'eb_backward_sig': eb['backward_significant'],
            'db_direction': db['direction'],
            'db_gravity': db['gravity_score'],
            'db_forward_sig': db['forward_significant'],
            'db_backward_sig': db['backward_significant'],
            'lb_direction': lb['direction'],
            'lb_gravity': lb['gravity_score'],
            'lb_forward_sig': lb['forward_significant'],
            'lb_backward_sig': lb['backward_significant']
        })

    # Save results
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(ANALYSIS_DIR / 'granger_results.csv', index=False)

    with open(ANALYSIS_DIR / 'granger_stats.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    print("\n=== GENERATING FIGURES ===")
    fig = plot_gravity_scores(all_results)
    plt.savefig(FIGURES_DIR / 'granger_gravity_scores.png')
    plt.savefig(FIGURES_DIR / 'granger_gravity_scores.pdf')
    plt.close()
    print("  granger_gravity_scores.png")

    print("\n" + "="*60)
    print("GRANGER CAUSALITY ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  {ANALYSIS_DIR / 'granger_results.csv'}")
    print(f"  {ANALYSIS_DIR / 'granger_stats.json'}")
    print(f"  {FIGURES_DIR / 'granger_gravity_scores.png'}")


if __name__ == "__main__":
    main()
