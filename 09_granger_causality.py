#!/usr/bin/env python3
"""
09_granger_causality.py

Granger causality analysis for decision-making profiles.

For each artist (pooled across solos):
1. Time series: complexity(t), dissonance(t), anticipation(t) at phrase level
2. Stationarity test (ADF)
3. Lag selection (AIC/BIC)
4. Granger test both directions for two pairs:
   - Complexity ↔ Dissonance
   - Complexity ↔ Anticipation
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

MAX_LAG = 4  # Maximum lag to test
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
    ts_df['bluesiness'] = ts_df['iv_tuple'].apply(lambda iv: BLUES_IVS.get(iv, 0))

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


def granger_test(data, cause_col, effect_col, maxlag=MAX_LAG):
    """
    Run Granger causality test.

    Tests if cause_col Granger-causes effect_col.
    Returns F-statistic and p-value for optimal lag.
    """
    # Prepare data: [effect, cause] order for statsmodels
    test_data = data[[effect_col, cause_col]].dropna()

    if len(test_data) < maxlag + 5:
        return {
            'cause': cause_col,
            'effect': effect_col,
            'f_statistic': np.nan,
            'p_value': np.nan,
            'optimal_lag': np.nan,
            'n_obs': len(test_data),
            'significant': False,
            'error': 'Insufficient data'
        }

    try:
        results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

        # Find best lag by minimum p-value
        best_lag = 1
        best_p = 1.0
        best_f = 0.0

        for lag in range(1, maxlag + 1):
            if lag in results:
                # Get F-test results (ssr based F test)
                f_stat = results[lag][0]['ssr_ftest'][0]
                p_val = results[lag][0]['ssr_ftest'][1]

                if p_val < best_p:
                    best_p = p_val
                    best_f = f_stat
                    best_lag = lag

        return {
            'cause': cause_col,
            'effect': effect_col,
            'f_statistic': float(best_f),
            'p_value': float(best_p),
            'optimal_lag': int(best_lag),
            'n_obs': len(test_data),
            'significant': best_p < 0.05,
            'error': None
        }
    except Exception as e:
        return {
            'cause': cause_col,
            'effect': effect_col,
            'f_statistic': np.nan,
            'p_value': np.nan,
            'optimal_lag': np.nan,
            'n_obs': len(test_data),
            'significant': False,
            'error': str(e)
        }


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
        metrics_to_residualize = ['complexity', 'dissonance', 'anticipation', 'n_notes', 'bluesiness']
        artist_df = residualize_against_tempo(artist_df, metrics_to_residualize)

        # Use residualized columns for Granger tests
        complexity_col = 'complexity_resid' if 'complexity_resid' in artist_df.columns else 'complexity'
        dissonance_col = 'dissonance_resid' if 'dissonance_resid' in artist_df.columns else 'dissonance'
        anticipation_col = 'anticipation_resid' if 'anticipation_resid' in artist_df.columns else 'anticipation'
        n_notes_col = 'n_notes_resid' if 'n_notes_resid' in artist_df.columns else 'n_notes'
        bluesiness_col = 'bluesiness_resid' if 'bluesiness_resid' in artist_df.columns else 'bluesiness'

        results['tempo_controlled'] = True
    else:
        complexity_col = 'complexity'
        dissonance_col = 'dissonance'
        anticipation_col = 'anticipation'
        n_notes_col = 'n_notes'
        bluesiness_col = 'bluesiness'
        results['tempo_controlled'] = False

    # Stationarity tests (on original metrics for interpretability)
    for col in ['complexity', 'dissonance', 'anticipation']:
        adf = adf_test(artist_df[col], col)
        results[f'adf_{col}'] = adf

    # === Complexity ↔ Dissonance ===
    # Forward: complexity → dissonance (proactive)
    fwd_cd = granger_test(artist_df, complexity_col, dissonance_col)
    # Backward: dissonance → complexity (reactive)
    bwd_cd = granger_test(artist_df, dissonance_col, complexity_col)

    gravity_cd = compute_gravity(fwd_cd, bwd_cd)

    results['complexity_dissonance'] = {
        'forward': fwd_cd,  # complexity → dissonance
        'backward': bwd_cd,  # dissonance → complexity
        'gravity': gravity_cd
    }

    # === Complexity ↔ Anticipation ===
    # Forward: complexity → anticipation
    fwd_ca = granger_test(artist_df, complexity_col, anticipation_col)
    # Backward: anticipation → complexity
    bwd_ca = granger_test(artist_df, anticipation_col, complexity_col)

    gravity_ca = compute_gravity(fwd_ca, bwd_ca)

    results['complexity_anticipation'] = {
        'forward': fwd_ca,  # complexity → anticipation
        'backward': bwd_ca,  # anticipation → complexity
        'gravity': gravity_ca
    }

    # === Phrase Length ↔ Complexity ===
    # Forward: phrase_length → complexity (does length predict next complexity?)
    fwd_lc = granger_test(artist_df, n_notes_col, complexity_col)
    # Backward: complexity → phrase_length (does complexity predict next length?)
    bwd_lc = granger_test(artist_df, complexity_col, n_notes_col)

    gravity_lc = compute_gravity(fwd_lc, bwd_lc)

    results['length_complexity'] = {
        'forward': fwd_lc,  # length → complexity
        'backward': bwd_lc,  # complexity → length
        'gravity': gravity_lc
    }

    # === Complexity → Bluesiness ===
    fwd_cb = granger_test(artist_df, complexity_col, bluesiness_col)
    bwd_cb = granger_test(artist_df, bluesiness_col, complexity_col)
    gravity_cb = compute_gravity(fwd_cb, bwd_cb)

    results['complexity_bluesiness'] = {
        'forward': fwd_cb,  # complexity → bluesiness
        'backward': bwd_cb,  # bluesiness → complexity
        'gravity': gravity_cb
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
        gravity_cd.append(r['complexity_dissonance']['gravity']['gravity_score'])
        gravity_ca.append(r['complexity_anticipation']['gravity']['gravity_score'])
        gravity_lc.append(r['length_complexity']['gravity']['gravity_score'])
        gravity_cb.append(r['complexity_bluesiness']['gravity']['gravity_score'])
        gravity_db.append(r['dissonance_bluesiness']['gravity']['gravity_score'])
        gravity_lb.append(r['length_bluesiness']['gravity']['gravity_score'])
        direction_cd.append(r['complexity_dissonance']['gravity']['direction'])
        direction_ca.append(r['complexity_anticipation']['gravity']['direction'])
        direction_lc.append(r['length_complexity']['gravity']['direction'])
        direction_cb.append(r['complexity_bluesiness']['gravity']['direction'])
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
        (gravity_cd, direction_cd, 'Complexity ↔ Dissonance', '← Reactive | Proactive →'),
        (gravity_ca, direction_ca, 'Complexity ↔ Anticipation', '← Anticipation drives | Complexity drives →'),
        (gravity_lc, direction_lc, 'Length ↔ Complexity', '← Complexity drives | Length drives →'),
        (gravity_cb, direction_cb, 'Complexity → Bluesiness', '← Blues drives | Complexity drives →'),
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
        cd = results['complexity_dissonance']['gravity']
        ca = results['complexity_anticipation']['gravity']
        lc = results['length_complexity']['gravity']
        cb = results['complexity_bluesiness']['gravity']
        db = results['dissonance_bluesiness']['gravity']
        lb = results['length_bluesiness']['gravity']
        print(f"  Complexity↔Dissonance:   {cd['direction']:12} (gravity={cd['gravity_score']:+.3f})")
        print(f"  Complexity↔Anticipation: {ca['direction']:12} (gravity={ca['gravity_score']:+.3f})")
        print(f"  Length↔Complexity:       {lc['direction']:12} (gravity={lc['gravity_score']:+.3f})")
        print(f"  Complexity→Bluesiness:   {cb['direction']:12} (gravity={cb['gravity_score']:+.3f})")
        print(f"  Dissonance→Bluesiness:   {db['direction']:12} (gravity={db['gravity_score']:+.3f})")
        print(f"  Length→Bluesiness:       {lb['direction']:12} (gravity={lb['gravity_score']:+.3f})")

    # Summary table
    print("\n=== SUMMARY TABLE ===")
    print(f"{'Artist':20} {'C↔D':8} {'C↔A':8} {'L↔C':8} {'C→B':8} {'D→B':8} {'L→B':8}")
    print("-" * 75)

    summary_rows = []
    for r in all_results:
        cd = r['complexity_dissonance']['gravity']
        ca = r['complexity_anticipation']['gravity']
        lc = r['length_complexity']['gravity']
        cb = r['complexity_bluesiness']['gravity']
        db = r['dissonance_bluesiness']['gravity']
        lb = r['length_bluesiness']['gravity']

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
