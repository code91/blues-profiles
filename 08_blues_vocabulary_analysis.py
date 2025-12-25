#!/usr/bin/env python3
"""
08_blues_vocabulary_analysis.py

Analyzes blues scale deployments across artists.

Metrics:
1. Blues IV frequency - count exact matches from blues IV list
2. Blues quotient - % of phrases using blues IVs
3. Blues commitment level - weighted by cardinality (hex=6, pent=5, etc.)
4. Blues vocabulary breadth - how many different blues IVs each artist uses
5. Blues density over time - early vs late in solos
6. Blues IV transitions - common blues IV → blues IV moves
7. Blues vs non-blues alternation - oscillation patterns

Output:
- data/analysis/blues_vocabulary.csv
- data/analysis/blues_transitions.csv
- data/analysis/blues_stats.json
- data/figures/blues_*.png
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configuration
PHRASES_DIR = Path("data/phrases")
ANALYSIS_DIR = Path("data/analysis")
FIGURES_DIR = Path("data/figures")

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Blues IVs catalog
BLUES_IVS = {
    # TRICHORDS (3-note)
    (0, 1, 1, 0, 1, 0): {'name': 'blues_cell', 'cardinality': 3, 'description': '{1, b3, 4} or {1, b3, b7} or {b3, 5, b7}'},
    (0, 2, 0, 1, 0, 0): {'name': 'upper_structure', 'cardinality': 3, 'description': '{b3, 4, 5} or {4, 5, b7}'},
    (2, 1, 0, 0, 0, 0): {'name': 'blue_note_enclosure', 'cardinality': 3, 'description': '{4, b5, 5}'},
    (1, 0, 0, 0, 1, 1): {'name': 'tritone_resolution', 'cardinality': 3, 'description': '{1, b5, 5}'},
    (0, 0, 2, 0, 0, 1): {'name': 'dim_fragment', 'cardinality': 3, 'description': '{1, b3, b5}'},
    (0, 1, 0, 0, 2, 0): {'name': 'cadential_approach', 'cardinality': 3, 'description': '{1, 4, b7} or {b3, 4, b7}'},
    
    # TETRACHORDS (4-note)
    (0, 2, 1, 1, 2, 0): {'name': '4-23_blues', 'cardinality': 4, 'description': '{1, b3, 4, 5} or {b3, 4, 5, b7}'},
    (0, 1, 2, 1, 2, 0): {'name': '4-26_min7', 'cardinality': 4, 'description': '{1, b3, 5, b7}'},
    (0, 2, 1, 0, 3, 0): {'name': 'no5_tension', 'cardinality': 4, 'description': '{1, b3, 4, b7}'},
    (1, 0, 2, 2, 1, 0): {'name': '4-19_maj_min_clash', 'cardinality': 4, 'description': '{1, b3, 3, 5}'},
    (1, 1, 1, 1, 2, 0): {'name': 'chromatic_cluster', 'cardinality': 4, 'description': '{1, b3, 4, b5}'},
    
    # PENTACHORDS (5-note)
    (0, 3, 2, 1, 4, 0): {'name': 'minor_pentatonic', 'cardinality': 5, 'description': '{1, b3, 4, 5, b7}'},
    (2, 2, 2, 1, 2, 1): {'name': 'pent_blue_note', 'cardinality': 5, 'description': '{1, b3, 4, b5, 5}'},
    (2, 2, 2, 2, 2, 0): {'name': '5-27_upper', 'cardinality': 5, 'description': '{b3, 4, b5, 5, b7}'},
    (1, 3, 2, 1, 4, 0): {'name': 'pent_maj3', 'cardinality': 5, 'description': '{1, b3, 3, 4, 5, b7}'},
    
    # HEXATONIC (6-note)
    (2, 3, 3, 2, 4, 1): {'name': 'full_blues_scale', 'cardinality': 6, 'description': '{1, b3, 4, b5, 5, b7}'},
}


def load_phrase_data():
    """Load all phrase data with IVs."""
    all_data = []
    
    for filepath in PHRASES_DIR.glob("*.json"):
        if filepath.name == "summary.csv":
            continue
        with open(filepath) as f:
            data = json.load(f)
        
        melid = data['melid']
        performer = data['performer']
        title = data['title']
        
        for phrase in data['phrases']:
            iv_tuple = tuple(phrase['iv'])
            is_blues = iv_tuple in BLUES_IVS
            blues_info = BLUES_IVS.get(iv_tuple, None)
            
            all_data.append({
                'melid': melid,
                'performer': performer,
                'title': title,
                'phrase_id': phrase['phrase_id'],
                'chorus_id': phrase['chorus_id'],
                'iv': iv_tuple,
                'iv_str': str(phrase['iv']),
                'is_blues': is_blues,
                'blues_name': blues_info['name'] if blues_info else None,
                'blues_cardinality': blues_info['cardinality'] if blues_info else 0,
                'n_notes': phrase['n_notes']
            })
    
    return pd.DataFrame(all_data)


def compute_blues_metrics(df):
    """Compute per-artist blues metrics."""
    results = []
    
    for performer in df['performer'].unique():
        artist_df = df[df['performer'] == performer]
        
        total_phrases = len(artist_df)
        blues_phrases = artist_df['is_blues'].sum()
        
        # 1. Blues IV frequency
        blues_frequency = blues_phrases
        
        # 2. Blues quotient (%)
        blues_quotient = (blues_phrases / total_phrases * 100) if total_phrases > 0 else 0
        
        # 3. Blues commitment level (weighted by cardinality)
        blues_commitment = artist_df['blues_cardinality'].sum()
        blues_commitment_normalized = blues_commitment / total_phrases if total_phrases > 0 else 0
        
        # 4. Blues vocabulary breadth
        blues_ivs_used = artist_df[artist_df['is_blues']]['blues_name'].nunique()
        
        # 5. Blues density over time (first half vs second half)
        n_solos = artist_df['melid'].nunique()
        
        early_blues = 0
        late_blues = 0
        for melid in artist_df['melid'].unique():
            solo_df = artist_df[artist_df['melid'] == melid].sort_values('phrase_id')
            n = len(solo_df)
            mid = n // 2
            early_blues += solo_df.iloc[:mid]['is_blues'].sum()
            late_blues += solo_df.iloc[mid:]['is_blues'].sum()
        
        # Ratio: >1 means more blues early, <1 means more blues late
        blues_early_late_ratio = early_blues / late_blues if late_blues > 0 else (1.0 if early_blues == 0 else float('inf'))
        
        # 6. Blues vs non-blues alternation (count switches)
        alternations = 0
        for melid in artist_df['melid'].unique():
            solo_df = artist_df[artist_df['melid'] == melid].sort_values('phrase_id')
            blues_seq = solo_df['is_blues'].values
            for i in range(1, len(blues_seq)):
                if blues_seq[i] != blues_seq[i-1]:
                    alternations += 1
        
        alternation_rate = alternations / total_phrases if total_phrases > 1 else 0
        
        results.append({
            'performer': performer,
            'total_phrases': total_phrases,
            'blues_frequency': blues_frequency,
            'blues_quotient': blues_quotient,
            'blues_commitment': blues_commitment,
            'blues_commitment_normalized': blues_commitment_normalized,
            'blues_vocabulary_breadth': blues_ivs_used,
            'early_blues_count': early_blues,
            'late_blues_count': late_blues,
            'blues_early_late_ratio': blues_early_late_ratio,
            'alternations': alternations,
            'alternation_rate': alternation_rate
        })
    
    return pd.DataFrame(results)


def compute_blues_transitions(df):
    """Compute blues IV → blues IV transition frequencies."""
    transitions = []
    
    for performer in df['performer'].unique():
        artist_df = df[df['performer'] == performer]
        
        for melid in artist_df['melid'].unique():
            solo_df = artist_df[artist_df['melid'] == melid].sort_values('phrase_id')
            
            for i in range(len(solo_df) - 1):
                current = solo_df.iloc[i]
                next_phrase = solo_df.iloc[i + 1]
                
                # Only track blues → blues transitions
                if current['is_blues'] and next_phrase['is_blues']:
                    transitions.append({
                        'performer': performer,
                        'from_iv': current['blues_name'],
                        'to_iv': next_phrase['blues_name'],
                        'from_cardinality': current['blues_cardinality'],
                        'to_cardinality': next_phrase['blues_cardinality']
                    })
    
    return pd.DataFrame(transitions)


def compute_top_blues_ivs(df):
    """Get most common blues IVs per artist."""
    results = []
    
    for performer in df['performer'].unique():
        artist_df = df[(df['performer'] == performer) & (df['is_blues'])]
        
        if len(artist_df) == 0:
            continue
        
        counts = artist_df['blues_name'].value_counts()
        
        for rank, (name, count) in enumerate(counts.head(5).items(), 1):
            info = [v for k, v in BLUES_IVS.items() if v['name'] == name][0]
            results.append({
                'performer': performer,
                'rank': rank,
                'blues_name': name,
                'count': count,
                'cardinality': info['cardinality'],
                'description': info['description']
            })
    
    return pd.DataFrame(results)


def plot_blues_quotient(metrics_df):
    """Bar chart of blues quotient by artist."""
    print("Generating: blues_quotient.png")
    
    df = metrics_df.sort_values('blues_quotient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(df)))
    
    ax.barh(df['performer'], df['blues_quotient'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Blues Quotient (%)')
    ax.set_title('Blues IV Usage by Artist\n(% of phrases using catalogued blues IVs)')
    
    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['blues_quotient'] + 0.5, i, f"{row['blues_quotient']:.1f}%", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'blues_quotient.png')
    plt.savefig(FIGURES_DIR / 'blues_quotient.pdf')
    plt.close()


def plot_blues_vocabulary_breadth(metrics_df):
    """Bar chart of blues vocabulary breadth."""
    print("Generating: blues_vocabulary_breadth.png")
    
    df = metrics_df.sort_values('blues_vocabulary_breadth', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Greens(np.linspace(0.3, 0.9, len(df)))
    
    ax.barh(df['performer'], df['blues_vocabulary_breadth'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Unique Blues IVs Used')
    ax.set_title(f'Blues Vocabulary Breadth by Artist\n(out of {len(BLUES_IVS)} catalogued blues IVs)')
    
    # Add value labels
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row['blues_vocabulary_breadth'] + 0.1, i, f"{int(row['blues_vocabulary_breadth'])}", va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'blues_vocabulary_breadth.png')
    plt.savefig(FIGURES_DIR / 'blues_vocabulary_breadth.pdf')
    plt.close()


def plot_blues_timing(metrics_df):
    """Scatter plot of early vs late blues usage."""
    print("Generating: blues_timing.png")
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(metrics_df['early_blues_count'], metrics_df['late_blues_count'], 
               s=100, alpha=0.7, c='steelblue', edgecolors='black', linewidth=0.5)
    
    # Add labels
    for idx, row in metrics_df.iterrows():
        ax.annotate(row['performer'], (row['early_blues_count'], row['late_blues_count']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Diagonal line (equal usage)
    max_val = max(metrics_df['early_blues_count'].max(), metrics_df['late_blues_count'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal early/late')
    
    ax.set_xlabel('Blues IVs in First Half of Solos')
    ax.set_ylabel('Blues IVs in Second Half of Solos')
    ax.set_title('Blues Deployment Timing\n(Above diagonal = more late, Below = more early)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'blues_timing.png')
    plt.savefig(FIGURES_DIR / 'blues_timing.pdf')
    plt.close()


def plot_blues_alternation(metrics_df):
    """Bar chart of alternation rate."""
    print("Generating: blues_alternation.png")
    
    df = metrics_df.sort_values('alternation_rate', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(df)))
    
    ax.barh(df['performer'], df['alternation_rate'], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Alternation Rate (switches per phrase)')
    ax.set_title('Blues vs Non-Blues Alternation by Artist\n(Higher = more switching between blues/non-blues)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'blues_alternation.png')
    plt.savefig(FIGURES_DIR / 'blues_alternation.pdf')
    plt.close()


def plot_blues_commitment(metrics_df):
    """Scatter: quotient vs commitment (weighted by cardinality)."""
    print("Generating: blues_commitment.png")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(metrics_df['blues_quotient'], metrics_df['blues_commitment_normalized'],
               s=100, alpha=0.7, c='darkgreen', edgecolors='black', linewidth=0.5)
    
    for idx, row in metrics_df.iterrows():
        ax.annotate(row['performer'], (row['blues_quotient'], row['blues_commitment_normalized']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Blues Quotient (% of phrases)')
    ax.set_ylabel('Blues Commitment (cardinality-weighted per phrase)')
    ax.set_title('Blues Quotient vs Commitment\n(X = frequency, Y = depth)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'blues_commitment.png')
    plt.savefig(FIGURES_DIR / 'blues_commitment.pdf')
    plt.close()


def plot_top_blues_grid(top_ivs_df, metrics_df):
    """Grid showing top blues IVs per artist."""
    print("Generating: blues_top_ivs_grid.png")
    
    performers = metrics_df.sort_values('blues_quotient', ascending=False)['performer'].tolist()
    performers = [p for p in performers if p in top_ivs_df['performer'].values]
    
    if len(performers) == 0:
        print("  No blues IVs found, skipping grid")
        return
    
    n_artists = len(performers)
    ncols = 3
    nrows = (n_artists + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()
    
    for idx, performer in enumerate(performers):
        ax = axes[idx]
        artist_top = top_ivs_df[top_ivs_df['performer'] == performer].head(5)
        
        if len(artist_top) == 0:
            ax.set_visible(False)
            continue
        
        colors = plt.cm.Blues(artist_top['cardinality'] / 6)
        ax.barh(artist_top['blues_name'], artist_top['count'], color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xlabel('Count')
        ax.set_title(performer)
        ax.invert_yaxis()
    
    for idx in range(len(performers), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Top Blues IVs by Artist (color = cardinality)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'blues_top_ivs_grid.png')
    plt.savefig(FIGURES_DIR / 'blues_top_ivs_grid.pdf')
    plt.close()


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading phrase data...")
    df = load_phrase_data()
    print(f"Total phrases: {len(df)}")
    print(f"Blues phrases: {df['is_blues'].sum()} ({df['is_blues'].mean()*100:.1f}%)")
    
    # Compute metrics
    print("\n=== COMPUTING BLUES METRICS ===")
    metrics_df = compute_blues_metrics(df)
    metrics_df = metrics_df.sort_values('blues_quotient', ascending=False)
    
    print("\nBlues Vocabulary Summary:")
    print(metrics_df[['performer', 'total_phrases', 'blues_frequency', 'blues_quotient', 
                      'blues_vocabulary_breadth', 'alternation_rate']].to_string(index=False))
    
    # Top blues IVs per artist
    print("\n=== TOP BLUES IVs BY ARTIST ===")
    top_ivs_df = compute_top_blues_ivs(df)
    
    for performer in metrics_df['performer'].head(5):
        artist_top = top_ivs_df[top_ivs_df['performer'] == performer].head(3)
        if len(artist_top) > 0:
            print(f"\n{performer}:")
            for _, row in artist_top.iterrows():
                print(f"  {row['blues_name']}: {row['count']} ({row['description']})")
    
    # Blues transitions
    print("\n=== BLUES TRANSITIONS ===")
    transitions_df = compute_blues_transitions(df)
    print(f"Total blues→blues transitions: {len(transitions_df)}")
    
    if len(transitions_df) > 0:
        top_transitions = transitions_df.groupby(['from_iv', 'to_iv']).size().sort_values(ascending=False).head(10)
        print("\nTop 10 blues IV transitions:")
        for (from_iv, to_iv), count in top_transitions.items():
            print(f"  {from_iv} → {to_iv}: {count}")
    
    # Save results
    metrics_df.to_csv(ANALYSIS_DIR / 'blues_vocabulary.csv', index=False)
    top_ivs_df.to_csv(ANALYSIS_DIR / 'blues_top_ivs.csv', index=False)
    if len(transitions_df) > 0:
        transitions_df.to_csv(ANALYSIS_DIR / 'blues_transitions.csv', index=False)
    
    # Stats JSON
    stats = {
        'total_phrases': int(len(df)),
        'blues_phrases': int(df['is_blues'].sum()),
        'blues_percentage': float(df['is_blues'].mean() * 100),
        'blues_ivs_catalog_size': len(BLUES_IVS),
        'blues_ivs_found': int(df[df['is_blues']]['blues_name'].nunique()),
        'artists': metrics_df.to_dict(orient='records')
    }
    
    with open(ANALYSIS_DIR / 'blues_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Generate figures
    print("\n=== GENERATING FIGURES ===")
    plot_blues_quotient(metrics_df)
    plot_blues_vocabulary_breadth(metrics_df)
    plot_blues_timing(metrics_df)
    plot_blues_alternation(metrics_df)
    plot_blues_commitment(metrics_df)
    plot_top_blues_grid(top_ivs_df, metrics_df)
    
    print("\n" + "="*60)
    print("BLUES VOCABULARY ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  {ANALYSIS_DIR / 'blues_vocabulary.csv'}")
    print(f"  {ANALYSIS_DIR / 'blues_top_ivs.csv'}")
    print(f"  {ANALYSIS_DIR / 'blues_transitions.csv'}")
    print(f"  {ANALYSIS_DIR / 'blues_stats.json'}")
    print(f"\nFigures:")
    for f in sorted(FIGURES_DIR.glob("blues_*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
