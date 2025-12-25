#!/usr/bin/env python3
"""
05_anticipation_analysis.py

Analyzes anticipation patterns within phrases using nested IV structure.

1. IV Transitions
   - Consecutive sub-IV pairs within phrases
   - Most frequent transitions per artist
   - Transition vocabulary (unique transitions / total)

2. Anticipation Metrics
   - Cosine similarity: first segment IV → phrase IV
   - Early vs late loading: sim(first, phrase) vs sim(last, phrase)
   - Cumulative convergence: how quickly does running IV approach phrase IV?

3. Statistical Tests
   - Kruskal-Wallis on anticipation scores across artists
   - Correlation: anticipation vs tempo, phrase length
   - Artist profiles: anticipators vs reactors

Output:
- data/analysis/iv_transitions_by_artist.csv
- data/analysis/anticipation_metrics.csv
- data/analysis/anticipation_stats.json
- data/figures/fig9_*.png
"""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cosine

# Configuration
PHRASES_DIR = Path("data/phrases")
METADATA_PATH = Path("data/corpus_metadata.csv")
ANALYSIS_DIR = Path("data/analysis")
FIGURES_DIR = Path("data/figures")

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1, v2 = np.array(v1), np.array(v2)
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def iv_to_str(iv):
    """Convert IV list to string for counting."""
    return str(iv)


def load_all_phrase_data():
    """Load all phrase data with chord segments."""
    all_data = []
    
    for filepath in PHRASES_DIR.glob("*.json"):
        if filepath.name == "summary.csv":
            continue
        with open(filepath) as f:
            data = json.load(f)
        
        for phrase in data['phrases']:
            if len(phrase['chord_segments']) >= 1:
                all_data.append({
                    'melid': data['melid'],
                    'performer': data['performer'],
                    'title': data['title'],
                    'phrase_id': phrase['phrase_id'],
                    'phrase_iv': phrase['iv'],
                    'n_notes': phrase['n_notes'],
                    'n_segments': len(phrase['chord_segments']),
                    'chord_segments': phrase['chord_segments']
                })
    
    return all_data


def extract_transitions(phrase_data):
    """Extract all IV transitions from phrases with 2+ segments."""
    transitions = []
    
    for phrase in phrase_data:
        if phrase['n_segments'] < 2:
            continue
        
        segments = phrase['chord_segments']
        for i in range(len(segments) - 1):
            iv_from = tuple(segments[i]['iv'])
            iv_to = tuple(segments[i + 1]['iv'])
            
            transitions.append({
                'performer': phrase['performer'],
                'melid': phrase['melid'],
                'phrase_id': phrase['phrase_id'],
                'iv_from': iv_from,
                'iv_to': iv_to,
                'iv_from_str': str(list(iv_from)),
                'iv_to_str': str(list(iv_to)),
                'transition': f"{list(iv_from)} → {list(iv_to)}",
                'chord_from': segments[i]['chord'],
                'chord_to': segments[i + 1]['chord']
            })
    
    return pd.DataFrame(transitions)


def compute_anticipation_metrics(phrase_data, metadata):
    """Compute anticipation metrics for each phrase."""
    
    # Load tempo info
    tempo_lookup = metadata.set_index('melid')['avgtempo'].to_dict()
    
    results = []
    
    for phrase in phrase_data:
        if phrase['n_segments'] < 2:
            continue
        
        segments = phrase['chord_segments']
        phrase_iv = np.array(phrase['phrase_iv'])
        
        # First and last segment IVs
        first_iv = np.array(segments[0]['iv'])
        last_iv = np.array(segments[-1]['iv'])
        
        # Anticipation: similarity of first segment to full phrase
        sim_first = cosine_similarity(first_iv, phrase_iv)
        sim_last = cosine_similarity(last_iv, phrase_iv)
        
        # Early vs late loading
        # Positive = early loading (anticipation), Negative = late loading (reaction)
        loading_balance = sim_first - sim_last
        
        # Cumulative convergence: track running IV through segments
        running_iv = np.zeros(6)
        convergence_trajectory = []
        
        for seg in segments:
            running_iv = running_iv + np.array(seg['iv'])
            sim_to_phrase = cosine_similarity(running_iv, phrase_iv)
            convergence_trajectory.append(sim_to_phrase)
        
        # How quickly does it converge? (area under curve, normalized)
        # Higher = faster convergence = more anticipation
        if len(convergence_trajectory) > 1:
            convergence_speed = np.trapezoid(convergence_trajectory) / len(convergence_trajectory)
        else:
            convergence_speed = convergence_trajectory[0] if convergence_trajectory else 0

        # Midpoint similarity (for phrases with 3+ segments)
        if len(segments) >= 3:
            mid_idx = len(segments) // 2
            mid_iv = np.array(segments[mid_idx]['iv'])
            sim_mid = cosine_similarity(mid_iv, phrase_iv)
        else:
            sim_mid = np.nan

        results.append({
            'performer': phrase['performer'],
            'melid': phrase['melid'],
            'title': phrase['title'],
            'phrase_id': phrase['phrase_id'],
            'n_segments': phrase['n_segments'],
            'n_notes': phrase['n_notes'],
            'tempo': tempo_lookup.get(phrase['melid'], np.nan),
            'sim_first_to_phrase': sim_first,
            'sim_last_to_phrase': sim_last,
            'sim_mid_to_phrase': sim_mid,
            'loading_balance': loading_balance,
            'convergence_speed': convergence_speed,
            'convergence_trajectory': convergence_trajectory
        })

    return pd.DataFrame(results)


def analyze_transitions_by_artist(transitions_df):
    """Analyze transition patterns per artist."""
    results = []
    top_transitions = []

    for performer in transitions_df['performer'].unique():
        artist_df = transitions_df[transitions_df['performer'] == performer]

        # Count unique transitions
        total = len(artist_df)
        unique = artist_df['transition'].nunique()
        ttr = unique / total if total > 0 else 0

        # Most common transitions
        top_5 = artist_df['transition'].value_counts().head(5)

        for rank, (transition, count) in enumerate(top_5.items(), 1):
            top_transitions.append({
                'performer': performer,
                'rank': rank,
                'transition': transition,
                'count': count,
                'proportion': count / total
            })

        results.append({
            'performer': performer,
            'total_transitions': total,
            'unique_transitions': unique,
            'transition_ttr': ttr
        })

    return pd.DataFrame(results), pd.DataFrame(top_transitions)


def statistical_tests(anticipation_df):
    """Run statistical tests on anticipation metrics."""

    results = {}

    # 1. Kruskal-Wallis on loading_balance across artists
    groups = [group['loading_balance'].dropna().values
              for _, group in anticipation_df.groupby('performer')]
    groups = [g for g in groups if len(g) > 0]

    h_stat, p_value = stats.kruskal(*groups)

    n = len(anticipation_df.dropna(subset=['loading_balance']))
    k = anticipation_df['performer'].nunique()
    epsilon_squared = (h_stat - k + 1) / (n - k) if n > k else 0

    results['kruskal_wallis_loading_balance'] = {
        'test': 'Kruskal-Wallis H-test',
        'metric': 'loading_balance (early - late similarity)',
        'h_statistic': float(h_stat),
        'p_value': float(p_value),
        'effect_size': float(epsilon_squared),
        'effect_size_name': 'epsilon_squared',
        'significant': bool(p_value < 0.05),
        'interpretation': 'Tests whether anticipation style differs across artists'
    }

    # 2. Correlation: anticipation vs tempo
    valid = anticipation_df.dropna(subset=['loading_balance', 'tempo'])
    if len(valid) > 10:
        r, p = stats.spearmanr(valid['tempo'], valid['loading_balance'])
        results['correlation_tempo_anticipation'] = {
            'test': 'Spearman correlation',
            'variables': ['tempo', 'loading_balance'],
            'r': float(r),
            'p_value': float(p),
            'significant': bool(p < 0.05),
            'interpretation': 'Positive r = faster tempos have more anticipation'
        }

    # 3. Correlation: anticipation vs phrase length
    valid = anticipation_df.dropna(subset=['loading_balance', 'n_notes'])
    if len(valid) > 10:
        r, p = stats.spearmanr(valid['n_notes'], valid['loading_balance'])
        results['correlation_phrase_length_anticipation'] = {
            'test': 'Spearman correlation',
            'variables': ['n_notes', 'loading_balance'],
            'r': float(r),
            'p_value': float(p),
            'significant': bool(p < 0.05),
            'interpretation': 'Positive r = longer phrases have more anticipation'
        }

    # 4. Artist profiles: mean anticipation scores
    artist_profiles = anticipation_df.groupby('performer').agg({
        'loading_balance': ['mean', 'std', 'count'],
        'sim_first_to_phrase': 'mean',
        'sim_last_to_phrase': 'mean',
        'convergence_speed': 'mean'
    }).round(4)

    artist_profiles.columns = ['_'.join(col).strip() for col in artist_profiles.columns]
    artist_profiles = artist_profiles.reset_index()

    # Classify artists
    mean_loading = anticipation_df['loading_balance'].mean()
    artist_profiles['style'] = artist_profiles['loading_balance_mean'].apply(
        lambda x: 'ANTICIPATOR' if x > mean_loading + 0.05 else ('REACTOR' if x < mean_loading - 0.05 else 'BALANCED')
    )

    results['artist_profiles'] = artist_profiles.to_dict(orient='records')

    return results


def fig9_anticipation_boxplots(anticipation_df):
    """Boxplots of loading balance by artist."""
    print("Generating Figure 9: Anticipation boxplots...")

    # Order by mean loading balance
    order = anticipation_df.groupby('performer')['loading_balance'].mean().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=anticipation_df, x='performer', y='loading_balance',
                order=order, hue='performer', palette='RdYlBu_r', legend=False, ax=ax)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Balanced')
    ax.set_xlabel('Artist')
    ax.set_ylabel('Loading Balance\n(First−Last Similarity to Phrase)')
    ax.set_title('Anticipation Style by Artist\n(Positive = Early Loading / Anticipator, Negative = Late Loading / Reactor)')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig9_anticipation_boxplots.png')
    plt.savefig(FIGURES_DIR / 'fig9_anticipation_boxplots.pdf')
    plt.close()


def fig10_anticipation_scatter(anticipation_df):
    """Scatter plot: first vs last segment similarity."""
    print("Generating Figure 10: First vs Last similarity scatter...")

    fig, ax = plt.subplots(figsize=(8, 8))

    performers = anticipation_df['performer'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(performers)))

    for performer, color in zip(performers, colors):
        mask = anticipation_df['performer'] == performer
        ax.scatter(anticipation_df.loc[mask, 'sim_first_to_phrase'],
                   anticipation_df.loc[mask, 'sim_last_to_phrase'],
                   c=[color], label=performer, alpha=0.6, s=30)

    # Diagonal line (balanced)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Balanced')

    ax.set_xlabel('First Segment → Phrase Similarity')
    ax.set_ylabel('Last Segment → Phrase Similarity')
    ax.set_title('Anticipation vs Reaction\n(Above diagonal = Reactor, Below = Anticipator)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig10_anticipation_scatter.png')
    plt.savefig(FIGURES_DIR / 'fig10_anticipation_scatter.pdf')
    plt.close()


def fig11_artist_anticipation_profile(anticipation_df):
    """Bar chart of mean loading balance per artist."""
    print("Generating Figure 11: Artist anticipation profiles...")

    artist_means = anticipation_df.groupby('performer')['loading_balance'].agg(['mean', 'std', 'count'])
    artist_means['se'] = artist_means['std'] / np.sqrt(artist_means['count'])
    artist_means = artist_means.sort_values('mean')

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#d73027' if x < -0.02 else '#4575b4' if x > 0.02 else '#ffffbf'
              for x in artist_means['mean']]

    bars = ax.barh(artist_means.index, artist_means['mean'],
                   xerr=artist_means['se'] * 1.96,  # 95% CI
                   color=colors, edgecolor='black', linewidth=0.5, capsize=3)

    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Mean Loading Balance (± 95% CI)\n← Reactor | Anticipator →')
    ax.set_title('Artist Anticipation Profiles')

    # Add annotations
    for bar, (performer, row) in zip(bars, artist_means.iterrows()):
        x = row['mean']
        label = 'R' if x < -0.02 else 'A' if x > 0.02 else 'B'
        ax.text(x + 0.01 if x >= 0 else x - 0.01, bar.get_y() + bar.get_height()/2,
                label, va='center', ha='left' if x >= 0 else 'right', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig11_artist_anticipation_profile.png')
    plt.savefig(FIGURES_DIR / 'fig11_artist_anticipation_profile.pdf')
    plt.close()


def fig12_top_transitions(top_transitions_df):
    """Top transitions visualization."""
    print("Generating Figure 12: Top transitions by artist...")

    # Get rank 1 transitions for each artist
    top1 = top_transitions_df[top_transitions_df['rank'] == 1].copy()
    top1 = top1.sort_values('count', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(top1))
    bars = ax.barh(y_pos, top1['count'], color='steelblue', edgecolor='black', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['performer']}" for _, row in top1.iterrows()])

    # Add transition labels
    for bar, (_, row) in zip(bars, top1.iterrows()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{row['transition'][:50]}...", va='center', fontsize=7)

    ax.set_xlabel('Count')
    ax.set_title('Most Frequent IV Transition per Artist')
    ax.set_xlim(0, top1['count'].max() * 1.8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig12_top_transitions.png')
    plt.savefig(FIGURES_DIR / 'fig12_top_transitions.pdf')
    plt.close()


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading phrase data...")
    phrase_data = load_all_phrase_data()
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(phrase_data)} phrases")

    # === TRANSITIONS ===
    print("\n=== IV TRANSITIONS ===")
    transitions_df = extract_transitions(phrase_data)
    print(f"Total transitions: {len(transitions_df)}")

    trans_by_artist, top_trans = analyze_transitions_by_artist(transitions_df)
    trans_by_artist = trans_by_artist.sort_values('total_transitions', ascending=False)

    print("\nTransition vocabulary by artist:")
    print(trans_by_artist.to_string(index=False))

    trans_by_artist.to_csv(ANALYSIS_DIR / 'transition_vocabulary_by_artist.csv', index=False)
    top_trans.to_csv(ANALYSIS_DIR / 'top_transitions_by_artist.csv', index=False)

    # === ANTICIPATION METRICS ===
    print("\n=== ANTICIPATION METRICS ===")
    anticipation_df = compute_anticipation_metrics(phrase_data, metadata)
    print(f"Phrases with 2+ segments: {len(anticipation_df)}")

    # Summary by artist
    artist_summary = anticipation_df.groupby('performer').agg({
        'loading_balance': ['mean', 'std'],
        'sim_first_to_phrase': 'mean',
        'convergence_speed': 'mean'
    }).round(3)
    artist_summary.columns = ['_'.join(col) for col in artist_summary.columns]
    artist_summary = artist_summary.sort_values('loading_balance_mean', ascending=False)

    print("\nAnticipation summary by artist:")
    print(artist_summary.to_string())

    # Save (without trajectory column for CSV)
    anticipation_save = anticipation_df.drop(columns=['convergence_trajectory'])
    anticipation_save.to_csv(ANALYSIS_DIR / 'anticipation_metrics.csv', index=False)

    # === STATISTICAL TESTS ===
    print("\n=== STATISTICAL TESTS ===")
    stat_results = statistical_tests(anticipation_df)

    kw = stat_results['kruskal_wallis_loading_balance']
    print(f"\nKruskal-Wallis (loading balance across artists):")
    print(f"  H = {kw['h_statistic']:.2f}, p = {kw['p_value']:.4f}, ε² = {kw['effect_size']:.3f}")
    print(f"  Significant: {kw['significant']}")

    if 'correlation_tempo_anticipation' in stat_results:
        corr = stat_results['correlation_tempo_anticipation']
        print(f"\nTempo vs Anticipation:")
        print(f"  r = {corr['r']:.3f}, p = {corr['p_value']:.4f}")

    if 'correlation_phrase_length_anticipation' in stat_results:
        corr = stat_results['correlation_phrase_length_anticipation']
        print(f"\nPhrase Length vs Anticipation:")
        print(f"  r = {corr['r']:.3f}, p = {corr['p_value']:.4f}")

    # Artist classification
    print("\nArtist Classification:")
    for profile in sorted(stat_results['artist_profiles'], key=lambda x: x['loading_balance_mean'], reverse=True):
        print(f"  {profile['performer']:20} {profile['loading_balance_mean']:+.3f} ({profile['style']})")

    with open(ANALYSIS_DIR / 'anticipation_stats.json', 'w') as f:
        json.dump(stat_results, f, indent=2)

    # === FIGURES ===
    print("\n=== GENERATING FIGURES ===")
    fig9_anticipation_boxplots(anticipation_df)
    fig10_anticipation_scatter(anticipation_df)
    fig11_artist_anticipation_profile(anticipation_df)
    fig12_top_transitions(top_trans)

    # === SUMMARY ===
    print("\n" + "="*60)
    print("ANTICIPATION ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  {ANALYSIS_DIR / 'transition_vocabulary_by_artist.csv'}")
    print(f"  {ANALYSIS_DIR / 'top_transitions_by_artist.csv'}")
    print(f"  {ANALYSIS_DIR / 'anticipation_metrics.csv'}")
    print(f"  {ANALYSIS_DIR / 'anticipation_stats.json'}")
    print(f"\nFigures:")
    for f in sorted(FIGURES_DIR.glob("fig9*.png")) + sorted(FIGURES_DIR.glob("fig1[0-2]*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
