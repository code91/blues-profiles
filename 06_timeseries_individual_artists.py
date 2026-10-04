#!/usr/bin/env python3
"""
06_timeseries_individual_artists.py

Generates time series plots for each artist showing density and dissonance
across all their blues phrases (pooled across solos).

Metrics:
- Density = sum(IV) = ic1 + ic2 + ic3 + ic4 + ic5 + ic6 = C(cardinality, 2)
- Interval entropy = Shannon entropy over the six interval classes
- Dissonance = ic1 + 0.5*ic2 + 0.8*ic6 (Rubini 2025, psychoacoustic weighting)

Output:
- data/figures/timeseries_{artist}.png
- data/timeseries/timeseries_data.csv
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
PHRASES_DIR = Path("data/phrases")
METADATA_PATH = Path("data/corpus_metadata.csv")
FIGURES_DIR = Path("data/figures")
TIMESERIES_DIR = Path("data/timeseries")

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def compute_density(iv):
    """Pitch-class density = sum of all IV components.

    NOT a complexity measure, and it was called one until review made the point.
    The sum of an interval vector over an n-note pitch-class set is exactly
    C(n, 2) = n(n-1)/2, so this quantity is a bijection with the set size and
    carries no information beyond it. Verified on this corpus: the identity
    holds for 1170 of 1170 phrases. It is kept because density per phrase is a
    real variable (how much of the chromatic a phrase touches), but it is named
    for what it measures.

    Use compute_interval_entropy() where harmonic complexity is meant.
    """
    return sum(iv)


def compute_interval_entropy(iv):
    """Shannon entropy over the six interval classes, in bits.

    This is the complexity measure that density is not: it asks how evenly a
    phrase's interval content is spread, independently of how many notes it
    uses. A diminished seventh concentrates in two interval classes and scores
    low (0.92 bits) despite twice a triad's density; a whole-tone collection
    scores 1.52, below a major triad's 1.59, despite five times the density.
    Measured correlation with density on this corpus is 0.545, so the two are
    substantially independent.

    Maximum is log2(6) = 2.585, approached by a chromatic phrase.
    """
    total = sum(iv)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in iv if c > 0)


def compute_dissonance(iv):
    """
    Dissonance = ic1 + 0.5*ic2 + 0.8*ic6
    
    Weighted sum emphasizing dissonant intervals:
    - Minor seconds (ic1): 1.0 (maximal critical band overlap)
    - Major seconds (ic2): 0.5 (moderate interference)
    - Tritones (ic6): 0.8 (tonal instability)
    
    Reference: Rubini 2025, based on Plomp & Levelt 1965, Sethares 1993
    """
    ic1, ic2, ic3, ic4, ic5, ic6 = iv
    return ic1 + 0.5 * ic2 + 0.8 * ic6


def compute_dissonance_ratio(iv):
    """Dissonance per unit of density, which is the comparable measure.

    Raw dissonance is a weighted SUB-SUM of the same six components that make up
    density, so the two correlate at r = 0.991 on this corpus and a model using
    both carries a variance inflation factor near 59. Dividing by density removes
    the shared size component and drops that correlation to 0.088.

    The same correction was asked for, and made, in review of the companion
    Parker study.
    """
    total = sum(iv)
    if total <= 0:
        return 0.0
    return compute_dissonance(iv) / total


def load_timeseries_data():
    """Load all phrase data and compute time series metrics."""
    all_data = []
    
    # Load metadata for ordering
    metadata = pd.read_csv(METADATA_PATH)
    
    for filepath in PHRASES_DIR.glob("*.json"):
        if filepath.name == "summary.csv":
            continue
        with open(filepath) as f:
            data = json.load(f)
        
        melid = data['melid']
        performer = data['performer']
        title = data['title']
        
        # Get tempo for this solo
        tempo = metadata[metadata['melid'] == melid]['avgtempo'].values
        tempo = tempo[0] if len(tempo) > 0 else np.nan
        
        for phrase in data['phrases']:
            iv = phrase['iv']
            
            all_data.append({
                'melid': melid,
                'performer': performer,
                'title': title,
                'phrase_id': phrase['phrase_id'],
                'chorus_id': phrase['chorus_id'],
                'n_notes': phrase['n_notes'],
                'tempo': tempo,
                'iv': iv,
                'density': compute_density(iv),
                'dissonance': compute_dissonance(iv),
                'dissonance_ratio': compute_dissonance_ratio(iv),
                'entropy': compute_interval_entropy(iv),
                'cardinality': len(phrase['pcs'])
            })
    
    df = pd.DataFrame(all_data)
    
    # Sort by performer, melid, phrase_id to get temporal order
    df = df.sort_values(['performer', 'melid', 'phrase_id']).reset_index(drop=True)
    
    # Add within-artist phrase index
    df['artist_phrase_idx'] = df.groupby('performer').cumcount()
    
    return df


def plot_artist_timeseries(df, performer, ax=None):
    """Plot time series for a single artist."""
    artist_df = df[df['performer'] == performer].copy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    x = artist_df['artist_phrase_idx'].values
    density = artist_df['density'].values
    dissonance = artist_df['dissonance'].values
    
    # Plot both metrics
    ax.plot(x, density, color='steelblue', linewidth=1, alpha=0.7, label='Density')
    ax.plot(x, dissonance, color='crimson', linewidth=1, alpha=0.7, label='Dissonance')
    
    # Add rolling mean (window=5)
    if len(x) >= 5:
        window = min(5, len(x) // 3) if len(x) > 3 else 1
        density_smooth = pd.Series(density).rolling(window, center=True).mean()
        dissonance_smooth = pd.Series(dissonance).rolling(window, center=True).mean()
        ax.plot(x, density_smooth, color='darkblue', linewidth=2, label='Density (smoothed)')
        ax.plot(x, dissonance_smooth, color='darkred', linewidth=2, label='Dissonance (smoothed)')
    
    # Mark solo boundaries
    solo_changes = artist_df[artist_df['phrase_id'] == 1]['artist_phrase_idx'].values
    for sc in solo_changes[1:]:  # Skip first
        ax.axvline(x=sc, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel('Phrase Index')
    ax.set_ylabel('Value')
    ax.set_title(f'{performer} — Blues Time Series ({len(artist_df)} phrases across {artist_df["melid"].nunique()} solos)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, len(x) - 1)
    
    return ax


def plot_all_artists_grid(df):
    """Plot all artists in a grid."""
    performers = sorted(df['performer'].unique())
    n_artists = len(performers)
    
    # Calculate grid dimensions
    ncols = 3
    nrows = (n_artists + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))
    axes = axes.flatten()
    
    for idx, performer in enumerate(performers):
        plot_artist_timeseries(df, performer, axes[idx])
        axes[idx].set_title(performer, fontsize=11)
    
    # Hide empty subplots
    for idx in range(len(performers), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading and computing time series data...")
    df = load_timeseries_data()
    print(f"Total phrases: {len(df)}")
    print(f"Artists: {df['performer'].nunique()}")
    
    # Save time series data
    df_save = df.drop(columns=['iv'])  # IV is a list, save separately
    df_save.to_csv(TIMESERIES_DIR / 'timeseries_data.csv', index=False)
    print(f"\nSaved: {TIMESERIES_DIR / 'timeseries_data.csv'}")
    
    # Summary stats
    print("\n=== TIME SERIES SUMMARY ===")
    summary = df.groupby('performer').agg({
        'density': ['mean', 'std'],
        'dissonance': ['mean', 'std'],
        'artist_phrase_idx': 'max'
    }).round(2)
    summary.columns = ['density_mean', 'density_std', 'dissonance_mean', 'dissonance_std', 'n_phrases']
    summary['n_phrases'] = summary['n_phrases'] + 1
    summary = summary.sort_values('n_phrases', ascending=False)
    print(summary.to_string())
    
    # Plot individual artist time series
    print("\n=== GENERATING INDIVIDUAL PLOTS ===")
    for performer in df['performer'].unique():
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_artist_timeseries(df, performer, ax)
        
        # Clean filename
        filename = performer.replace(' ', '_').replace('.', '')
        plt.savefig(FIGURES_DIR / f'timeseries_{filename}.png')
        plt.savefig(FIGURES_DIR / f'timeseries_{filename}.pdf')
        plt.close()
        print(f"  {performer}")
    
    # Plot grid of all artists
    print("\n=== GENERATING GRID PLOT ===")
    fig = plot_all_artists_grid(df)
    plt.savefig(FIGURES_DIR / 'timeseries_all_artists.png')
    plt.savefig(FIGURES_DIR / 'timeseries_all_artists.pdf')
    plt.close()
    print("  timeseries_all_artists.png")
    
    print("\n" + "="*60)
    print("TIME SERIES VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\nOutput: {FIGURES_DIR}/timeseries_*.png")


if __name__ == "__main__":
    main()
