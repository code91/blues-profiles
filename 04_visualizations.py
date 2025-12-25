#!/usr/bin/env python3
"""
04_visualizations.py

Generates publication-quality figures for the blues profile analysis.

Figures:
1. TTR bar chart - vocabulary exploration ranked by artist
2. Jaccard heatmap - similarity matrix with clustering dendrogram
3. Phrase length boxplots - by artist, with significance indicators
4. Artist dendrogram - hierarchical clustering on IV centroids
5. IV space (PCA) - 6D → 2D, phrases colored by artist
6. Top IV heatmap - mean IV profile per artist
7. Shorter vs. others - sparse IV signature comparison

Output:
- data/figures/*.png and *.pdf
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration
ANALYSIS_DIR = Path("data/analysis")
PHRASES_DIR = Path("data/phrases")
FIGURES_DIR = Path("data/figures")

# Style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Color palette - distinct colors for 13 artists
ARTIST_COLORS = {
    'Charlie Parker': '#e41a1c',
    'Dizzy Gillespie': '#377eb8', 
    'Don Byas': '#4daf4a',
    'Louis Armstrong': '#984ea3',
    'J.J. Johnson': '#ff7f00',
    'Miles Davis': '#1f78b4',
    'John Coltrane': '#a6cee3',
    'Sonny Rollins': '#b2df8a',
    'Eric Dolphy': '#fb9a99',
    'Wayne Shorter': '#e31a1c',
    'Freddie Hubbard': '#fdbf6f',
    'Kenny Dorham': '#cab2d6',
    'Steve Coleman': '#6a3d9a',
    'Branford Marsalis': '#b15928'
}

# Era groupings
ERA_COLORS = {
    'TRADITIONAL': '#8dd3c7',
    'SWING': '#ffffb3',
    'BEBOP': '#bebada',
    'COOL': '#fb8072',
    'HARDBOP': '#80b1d3',
    'POSTBOP': '#fdb462',
    'FUSION': '#b3de69'
}


def load_all_phrases():
    """Load all phrase data from JSON files."""
    all_data = []
    
    for filepath in PHRASES_DIR.glob("*.json"):
        if filepath.name == "summary.csv":
            continue
        with open(filepath) as f:
            data = json.load(f)
        
        for phrase in data['phrases']:
            all_data.append({
                'melid': data['melid'],
                'performer': data['performer'],
                'title': data['title'],
                'phrase_id': phrase['phrase_id'],
                'n_notes': phrase['n_notes'],
                'n_chord_segments': len(phrase['chord_segments']),
                'iv': tuple(phrase['iv']),
                'iv_list': phrase['iv'],
                'forte': phrase['forte'],
                'cardinality': len(phrase['pcs'])
            })
    
    return pd.DataFrame(all_data)


def fig1_ttr_bar_chart():
    """Type-Token Ratio bar chart ranked by artist."""
    print("Generating Figure 1: TTR bar chart...")
    
    vocab_df = pd.read_csv(ANALYSIS_DIR / 'vocabulary_by_artist.csv')
    vocab_df = vocab_df.sort_values('type_token_ratio', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = [ARTIST_COLORS.get(p, '#666666') for p in vocab_df['performer']]
    
    bars = ax.barh(vocab_df['performer'], vocab_df['type_token_ratio'], color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Type-Token Ratio (Unique IVs / Total Phrases)')
    ax.set_title('Vocabulary Exploration by Artist')
    ax.set_xlim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars, vocab_df['type_token_ratio']):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', fontsize=9)
    
    # Add reference lines
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='50% repetition')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_ttr_bar_chart.png')
    plt.savefig(FIGURES_DIR / 'fig1_ttr_bar_chart.pdf')
    plt.close()


def fig2_jaccard_heatmap():
    """Jaccard similarity heatmap - lower triangle, clustered order."""
    print("Generating Figure 2: Jaccard heatmap...")

    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    jaccard_df = pd.read_csv(ANALYSIS_DIR / 'jaccard_similarity_matrix.csv', index_col=0)

    # Get clustered order using ward linkage
    distances = pdist(jaccard_df.values)
    Z = linkage(distances, method='ward')
    order = leaves_list(Z)

    # Reorder matrix
    ordered_labels = [jaccard_df.index[i] for i in order]
    jaccard_ordered = jaccard_df.loc[ordered_labels, ordered_labels]

    # Mask upper triangle
    mask = np.triu(np.ones_like(jaccard_ordered, dtype=bool))

    # Plot lower triangle only
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(jaccard_ordered,
                mask=mask,
                cmap='RdYlBu_r',
                annot=True,
                fmt='.2f',
                annot_kws={'size': 9},
                linewidths=0.5,
                square=True,
                cbar_kws={'label': 'Jaccard Similarity', 'shrink': 0.5, 'aspect': 20},
                ax=ax)

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('IV Vocabulary Overlap (Jaccard Similarity)', fontsize=12, pad=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_jaccard_heatmap.png')
    plt.savefig(FIGURES_DIR / 'fig2_jaccard_heatmap.pdf')
    plt.close()


def fig3_phrase_length_boxplots():
    """Phrase length boxplots by artist with significance indicators."""
    print("Generating Figure 3: Phrase length boxplots...")

    df = load_all_phrases()

    # Order by median phrase length
    order = df.groupby('performer')['n_notes'].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create boxplot (no outlier circles)
    bp = ax.boxplot([df[df['performer'] == p]['n_notes'].values for p in order],
                    tick_labels=order,
                    patch_artist=True,
                    vert=True,
                    showfliers=False)

    # Color boxes
    for patch, performer in zip(bp['boxes'], order):
        patch.set_facecolor(ARTIST_COLORS.get(performer, '#666666'))
        patch.set_alpha(0.7)

    ax.set_ylabel('Phrase Length (notes)')
    ax.set_title('Phrase Length Distribution by Artist\n(Kruskal-Wallis H=109.3, p<0.001)')

    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_phrase_length_boxplots.png')
    plt.savefig(FIGURES_DIR / 'fig3_phrase_length_boxplots.pdf')
    plt.close()


def fig4_artist_dendrogram():
    """Hierarchical clustering dendrogram on IV centroids."""
    print("Generating Figure 4: Artist dendrogram...")

    with open(ANALYSIS_DIR / 'cluster_analysis.json') as f:
        cluster_data = json.load(f)

    performers = cluster_data['performers']
    profiles = np.array([cluster_data['mean_iv_profiles'][p] for p in performers])
    styles = [cluster_data['style_labels'][p] for p in performers]

    # Standardize and compute linkage
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles)
    Z = linkage(profiles_scaled, method='ward')

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create dendrogram
    dend = dendrogram(Z,
                      labels=performers,
                      leaf_rotation=45,
                      leaf_font_size=10,
                      ax=ax)

    # Color labels by style
    xlbls = ax.get_xmajorticklabels()
    for lbl in xlbls:
        performer = lbl.get_text()
        style = cluster_data['style_labels'].get(performer, 'UNKNOWN')
        lbl.set_color(ERA_COLORS.get(style, 'black'))

    ax.set_ylabel('Distance (Ward)')
    ax.set_title('Artist Clustering by Mean IV Profile')

    # Add legend for styles
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=style)
                       for style, color in ERA_COLORS.items()
                       if style in styles]
    ax.legend(handles=legend_elements, loc='upper right', title='Style')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_artist_dendrogram.png')
    plt.savefig(FIGURES_DIR / 'fig4_artist_dendrogram.pdf')
    plt.close()


def fig5_iv_space_pca():
    """PCA of IV space with phrases colored by artist."""
    print("Generating Figure 5: IV space PCA...")

    df = load_all_phrases()

    # Extract IV vectors
    X = np.array(df['iv_list'].tolist())

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each artist
    for performer in df['performer'].unique():
        mask = df['performer'] == performer
        ax.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'],
                   c=ARTIST_COLORS.get(performer, '#666666'),
                   label=performer,
                   alpha=0.6,
                   s=30,
                   edgecolors='white',
                   linewidth=0.3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.set_title('Phrase IV Space (PCA)')

    # Add legend outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)

    # Add component loadings as arrows
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    iv_labels = ['m2/M7', 'M2/m7', 'm3/M6', 'M3/m6', 'P4/P5', 'TT']

    scale = 3  # Scale arrows for visibility
    for i, (loading, label) in enumerate(zip(loadings, iv_labels)):
        ax.annotate('', xy=(loading[0]*scale, loading[1]*scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        ax.text(loading[0]*scale*1.15, loading[1]*scale*1.15, label,
                color='red', fontsize=9, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_iv_space_pca.png')
    plt.savefig(FIGURES_DIR / 'fig5_iv_space_pca.pdf')
    plt.close()


def fig6_mean_iv_heatmap():
    """Mean IV profile heatmap per artist."""
    print("Generating Figure 6: Mean IV heatmap...")

    with open(ANALYSIS_DIR / 'cluster_analysis.json') as f:
        cluster_data = json.load(f)

    performers = cluster_data['performers']
    profiles = {p: cluster_data['mean_iv_profiles'][p] for p in performers}

    # Create DataFrame
    iv_labels = ['m2/M7', 'M2/m7', 'm3/M6', 'M3/m6', 'P4/P5', 'TT']
    profile_df = pd.DataFrame(profiles, index=iv_labels).T

    # Sort by total IV magnitude
    profile_df['total'] = profile_df.sum(axis=1)
    profile_df = profile_df.sort_values('total', ascending=False)
    profile_df = profile_df.drop('total', axis=1)

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(profile_df,
                cmap='YlOrRd',
                annot=True,
                fmt='.1f',
                linewidths=0.5,
                ax=ax,
                cbar_kws={'label': 'Mean Count'})

    ax.set_xlabel('Interval Class')
    ax.set_ylabel('Artist')
    ax.set_title('Mean Interval Vector Profile by Artist')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_mean_iv_heatmap.png')
    plt.savefig(FIGURES_DIR / 'fig6_mean_iv_heatmap.pdf')
    plt.close()


def fig7_shorter_signature():
    """Wayne Shorter's sparse IV signature vs others."""
    print("Generating Figure 7: Shorter's signature...")

    df = load_all_phrases()

    # Get top IV for each artist
    top_ivs = pd.read_csv(ANALYSIS_DIR / 'top_ivs_by_artist.csv')
    top1 = top_ivs[top_ivs['rank'] == 1]

    # Parse IV strings back to lists
    def parse_iv(iv_str):
        return [int(x) for x in iv_str.strip('[]').split(', ')]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()

    # Select representative artists
    artists_to_show = ['Wayne Shorter', 'Miles Davis', 'John Coltrane',
                       'Charlie Parker', 'Eric Dolphy', 'Louis Armstrong']

    iv_labels = ['m2/M7', 'M2/m7', 'm3/M6', 'M3/m6', 'P4/P5', 'TT']
    x = np.arange(6)

    for idx, (ax, artist) in enumerate(zip(axes, artists_to_show)):
        row = top1[top1['performer'] == artist].iloc[0]
        iv = parse_iv(row['iv'])

        color = ARTIST_COLORS.get(artist, '#666666')
        bars = ax.bar(x, iv, color=color, edgecolor='black', linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(iv_labels, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title(f"{artist}\n(n={int(row['count'])}, {row['proportion']*100:.1f}%)")
        ax.set_ylim(0, 14)

        # Highlight Shorter's sparseness
        if artist == 'Wayne Shorter':
            ax.set_facecolor('#ffffcc')
            ax.text(0.5, 0.95, 'SPARSE', transform=ax.transAxes,
                    ha='center', va='top', fontsize=11, color='red', weight='bold')

    fig.suptitle('Most Frequent IV by Artist: Shorter\'s Distinctive Sparse Signature',
                 fontsize=13, y=1.02)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig7_shorter_signature.png')
    plt.savefig(FIGURES_DIR / 'fig7_shorter_signature.pdf')
    plt.close()


def fig8_cardinality_distribution():
    """Bonus: Cardinality (pitch-class count) distribution by artist."""
    print("Generating Figure 8 (bonus): Cardinality distribution...")

    df = load_all_phrases()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Order by mean cardinality
    order = df.groupby('performer')['cardinality'].mean().sort_values().index.tolist()

    # Violin plot
    parts = ax.violinplot([df[df['performer'] == p]['cardinality'].values for p in order],
                          positions=range(len(order)),
                          showmeans=True,
                          showmedians=True)

    # Color violins
    for pc, performer in zip(parts['bodies'], order):
        pc.set_facecolor(ARTIST_COLORS.get(performer, '#666666'))
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha='right')
    ax.set_ylabel('Cardinality (Pitch Classes per Phrase)')
    ax.set_title('Pitch-Class Set Size Distribution by Artist')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig8_cardinality_distribution.png')
    plt.savefig(FIGURES_DIR / 'fig8_cardinality_distribution.pdf')
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating all figures...\n")

    fig1_ttr_bar_chart()
    fig2_jaccard_heatmap()
    fig3_phrase_length_boxplots()
    fig4_artist_dendrogram()
    fig5_iv_space_pca()
    fig6_mean_iv_heatmap()
    fig7_shorter_signature()
    fig8_cardinality_distribution()

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED")
    print("="*60)
    print(f"Output directory: {FIGURES_DIR}/")
    print("\nFiles created:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
