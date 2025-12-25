#!/usr/bin/env python3
"""
06_timeseries_individual_artists.py

Generates time series plots for each artist showing complexity and dissonance
across all their blues phrases (pooled across solos).

Metrics:
- Complexity = sum(IV) = ic1 + ic2 + ic3 + ic4 + ic5 + ic6
- Dissonance = ic1 + 0.5*ic2 + 0.8*ic6 (Rubini 2025, psychoacoustic weighting)

Output:
- data/figures/timeseries_{artist}.png
- data/timeseries/timeseries_data.csv
"""

import json
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


def compute_complexity(iv):
    """Complexity = sum of all IV components."""
    return sum(iv)


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
                'complexity': compute_complexity(iv),
                'dissonance': compute_dissonance(iv),
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
    complexity = artist_df['complexity'].values
    dissonance = artist_df['dissonance'].values
    
    # Plot both metrics
    ax.plot(x, complexity, color='steelblue', linewidth=1, alpha=0.7, label='Complexity')
    ax.plot(x, dissonance, color='crimson', linewidth=1, alpha=0.7, label='Dissonance')
    
    # Add rolling mean (window=5)
    if len(x) >= 5:
        window = min(5, len(x) // 3) if len(x) > 3 else 1
        complexity_smooth = pd.Series(complexity).rolling(window, center=True).mean()
        dissonance_smooth = pd.Series(dissonance).rolling(window, center=True).mean()
        ax.plot(x, complexity_smooth, color='darkblue', linewidth=2, label='Complexity (smoothed)')
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
        'complexity': ['mean', 'std'],
        'dissonance': ['mean', 'std'],
        'artist_phrase_idx': 'max'
    }).round(2)
    summary.columns = ['complexity_mean', 'complexity_std', 'dissonance_mean', 'dissonance_std', 'n_phrases']
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
