#!/usr/bin/env python3
"""
03_vocabulary_and_structure_analysis.py

Analyzes IV vocabulary and phrase structure across artists.

Descriptive (per artist):
- Type-token ratio (unique IVs / total phrases)
- Most frequent IVs
- Mean phrase length, mean chord segments

Inferential (cross-artist):
- Jaccard similarity matrix between artist IV vocabularies
- Kruskal-Wallis test on phrase lengths
- Cluster analysis by era/style
- MANOVA / permutation test on IV distributions
- Discriminant analysis (which IV dimensions separate artists)

Output:
- data/analysis/*.csv and *.json
"""

import json
import os
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings('ignore')

# Configuration
PHRASES_DIR = Path("data/phrases")
METADATA_PATH = Path("data/corpus_metadata.csv")
ANALYSIS_DIR = Path("data/analysis")

# Style/era mapping for clustering validation
STYLE_ERA = {
    'TRADITIONAL': 1,
    'SWING': 2,
    'BEBOP': 3,
    'COOL': 4,
    'HARDBOP': 5,
    'POSTBOP': 6,
    'FUSION': 7
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
                'chorus_id': phrase['chorus_id'],
                'n_notes': phrase['n_notes'],
                'n_chord_segments': len(phrase['chord_segments']),
                'iv': tuple(phrase['iv']),
                'iv_str': str(phrase['iv']),
                'forte': phrase['forte'],
                'cardinality': len(phrase['pcs']),
                'bar_span': phrase['end_bar'] - phrase['start_bar'] + 1
            })

    df = pd.DataFrame(all_data)

    # Merge tempo from metadata
    metadata = pd.read_csv(METADATA_PATH)
    df = df.merge(metadata[['melid', 'avgtempo']], on='melid', how='left')
    df.rename(columns={'avgtempo': 'tempo'}, inplace=True)

    return df


def compute_descriptive_stats(df):
    """Compute per-artist descriptive statistics."""

    results = []
    top_ivs_all = []

    for performer in df['performer'].unique():
        artist_df = df[df['performer'] == performer]

        # Type-token ratio
        total_phrases = len(artist_df)
        unique_ivs = artist_df['iv'].nunique()
        ttr = unique_ivs / total_phrases

        # Phrase structure
        mean_phrase_length = artist_df['n_notes'].mean()
        std_phrase_length = artist_df['n_notes'].std()
        mean_chord_segments = artist_df['n_chord_segments'].mean()
        std_chord_segments = artist_df['n_chord_segments'].std()
        mean_bar_span = artist_df['bar_span'].mean()
        mean_cardinality = artist_df['cardinality'].mean()

        # Most frequent IVs
        iv_counts = artist_df['iv_str'].value_counts()
        top_5 = iv_counts.head(5)

        for rank, (iv_str, count) in enumerate(top_5.items(), 1):
            top_ivs_all.append({
                'performer': performer,
                'rank': rank,
                'iv': iv_str,
                'count': count,
                'proportion': count / total_phrases
            })

        results.append({
            'performer': performer,
            'total_phrases': total_phrases,
            'unique_ivs': unique_ivs,
            'type_token_ratio': ttr,
            'mean_phrase_length': mean_phrase_length,
            'std_phrase_length': std_phrase_length,
            'mean_chord_segments': mean_chord_segments,
            'std_chord_segments': std_chord_segments,
            'mean_bar_span': mean_bar_span,
            'mean_cardinality': mean_cardinality
        })

    return pd.DataFrame(results), pd.DataFrame(top_ivs_all)


def compute_jaccard_similarity(df):
    """Compute pairwise Jaccard similarity of IV vocabularies."""

    performers = df['performer'].unique()
    n = len(performers)

    # Get IV sets per artist
    iv_sets = {}
    for performer in performers:
        artist_df = df[df['performer'] == performer]
        iv_sets[performer] = set(artist_df['iv'].unique())

    # Compute pairwise Jaccard
    similarity_matrix = np.zeros((n, n))

    for i, p1 in enumerate(performers):
        for j, p2 in enumerate(performers):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                intersection = len(iv_sets[p1] & iv_sets[p2])
                union = len(iv_sets[p1] | iv_sets[p2])
                similarity_matrix[i, j] = intersection / union if union > 0 else 0

    return pd.DataFrame(similarity_matrix, index=performers, columns=performers)


def kruskal_wallis_phrase_length(df):
    """Kruskal-Wallis test for phrase length differences across artists."""

    groups = [group['n_notes'].values for _, group in df.groupby('performer')]

    # Kruskal-Wallis H-test
    h_stat, p_value = stats.kruskal(*groups)

    # Effect size: epsilon-squared (η²H)
    n = len(df)
    k = df['performer'].nunique()
    epsilon_squared = (h_stat - k + 1) / (n - k)

    # Post-hoc: Dunn's test with Bonferroni correction
    # We'll compute pairwise Mann-Whitney tests
    performers = df['performer'].unique()
    n_comparisons = len(performers) * (len(performers) - 1) // 2

    pairwise_results = []
    for p1, p2 in combinations(performers, 2):
        g1 = df[df['performer'] == p1]['n_notes']
        g2 = df[df['performer'] == p2]['n_notes']

        u_stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')

        # Cliff's delta (effect size for Mann-Whitney)
        n1, n2 = len(g1), len(g2)
        cliffs_delta = (2 * u_stat / (n1 * n2)) - 1

        # Bonferroni correction
        p_adjusted = min(p * n_comparisons, 1.0)

        pairwise_results.append({
            'artist_1': p1,
            'artist_2': p2,
            'u_statistic': float(u_stat),
            'p_value': float(p),
            'p_adjusted': float(p_adjusted),
            'cliffs_delta': float(cliffs_delta),
            'significant': bool(p_adjusted < 0.05)
        })

    return {
        'test': 'Kruskal-Wallis H-test',
        'h_statistic': float(h_stat),
        'p_value': float(p_value),
        'df': k - 1,
        'n_groups': k,
        'n_total': n,
        'effect_size': float(epsilon_squared),
        'effect_size_name': 'epsilon_squared',
        'significant': bool(p_value < 0.05),
        'pairwise_comparisons': pairwise_results
    }


def ancova_phrase_length(df):
    """
    ANCOVA for phrase length differences across artists, controlling for tempo.

    Model: n_notes ~ C(performer) + tempo

    This tests whether artists differ in phrase length after controlling
    for the effect of tempo on phrase length.
    """
    # Prepare data - ensure no missing values
    analysis_df = df[['n_notes', 'performer', 'tempo']].dropna().copy()

    # Fit ANCOVA model
    model = ols('n_notes ~ C(performer) + tempo', data=analysis_df).fit()

    # Type II ANOVA table (tests performer effect controlling for tempo)
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Extract results
    performer_f = anova_table.loc['C(performer)', 'F']
    performer_p = anova_table.loc['C(performer)', 'PR(>F)']
    performer_ss = anova_table.loc['C(performer)', 'sum_sq']

    tempo_f = anova_table.loc['tempo', 'F']
    tempo_p = anova_table.loc['tempo', 'PR(>F)']
    tempo_ss = anova_table.loc['tempo', 'sum_sq']

    residual_ss = anova_table.loc['Residual', 'sum_sq']
    total_ss = performer_ss + tempo_ss + residual_ss

    # Effect sizes (partial eta-squared)
    eta_sq_performer = performer_ss / (performer_ss + residual_ss)
    eta_sq_tempo = tempo_ss / (tempo_ss + residual_ss)

    # Tempo coefficient (negative = faster tempo → shorter phrases)
    tempo_coef = model.params['tempo']
    tempo_se = model.bse['tempo']

    # Post-hoc: estimated marginal means (adjusted for tempo)
    # Get mean tempo for adjustment
    mean_tempo = analysis_df['tempo'].mean()
    performers = analysis_df['performer'].unique()

    # Compute adjusted means
    adjusted_means = {}
    for perf in performers:
        perf_data = analysis_df[analysis_df['performer'] == perf]
        raw_mean = perf_data['n_notes'].mean()
        tempo_diff = perf_data['tempo'].mean() - mean_tempo
        adjusted_mean = raw_mean - tempo_coef * tempo_diff
        adjusted_means[perf] = {
            'raw_mean': float(raw_mean),
            'adjusted_mean': float(adjusted_mean),
            'mean_tempo': float(perf_data['tempo'].mean()),
            'n': len(perf_data)
        }

    return {
        'test': 'ANCOVA (Type II)',
        'model': 'n_notes ~ C(performer) + tempo',
        'n_total': len(analysis_df),
        'n_groups': len(performers),
        'performer_effect': {
            'f_statistic': float(performer_f),
            'p_value': float(performer_p),
            'partial_eta_squared': float(eta_sq_performer),
            'significant': bool(performer_p < 0.05)
        },
        'tempo_covariate': {
            'f_statistic': float(tempo_f),
            'p_value': float(tempo_p),
            'partial_eta_squared': float(eta_sq_tempo),
            'coefficient': float(tempo_coef),
            'std_error': float(tempo_se),
            'significant': bool(tempo_p < 0.05),
            'interpretation': 'negative = faster tempo → shorter phrases'
        },
        'model_r_squared': float(model.rsquared),
        'model_adj_r_squared': float(model.rsquared_adj),
        'adjusted_means': adjusted_means,
        'grand_mean_tempo': float(mean_tempo)
    }


def cluster_analysis(df, metadata):
    """Hierarchical clustering of artists based on IV usage patterns."""

    performers = df['performer'].unique()

    # Build artist IV profiles (mean IV vector)
    profiles = []
    for performer in performers:
        artist_df = df[df['performer'] == performer]
        ivs = np.array([list(iv) for iv in artist_df['iv']])
        mean_iv = ivs.mean(axis=0)
        profiles.append(mean_iv)

    profiles = np.array(profiles)

    # Standardize
    scaler = StandardScaler()
    profiles_scaled = scaler.fit_transform(profiles)

    # Hierarchical clustering
    distances = pdist(profiles_scaled, metric='euclidean')
    linkage_matrix = linkage(distances, method='ward')

    # Cut into clusters (try k=3,4,5 and report)
    cluster_results = {}
    for k in [2, 3, 4]:
        clusters = fcluster(linkage_matrix, k, criterion='maxclust')
        cluster_results[f'k={k}'] = {performer: int(c) for performer, c in zip(performers, clusters)}

    # Get style info from metadata
    performer_styles = metadata.groupby('performer')['style'].first().to_dict()

    # Compute adjusted Rand index if we have style labels
    from sklearn.metrics import adjusted_rand_score

    style_labels = [STYLE_ERA.get(performer_styles.get(p, 'UNKNOWN'), 0) for p in performers]

    ari_results = {}
    for k_label, cluster_dict in cluster_results.items():
        cluster_labels = [cluster_dict[p] for p in performers]
        ari = adjusted_rand_score(style_labels, cluster_labels)
        ari_results[k_label] = float(ari)

    return {
        'performers': list(performers),
        'mean_iv_profiles': {p: [float(x) for x in profiles[i]] for i, p in enumerate(performers)},
        'cluster_assignments': cluster_results,
        'style_labels': {p: performer_styles.get(p, 'UNKNOWN') for p in performers},
        'adjusted_rand_index_vs_style': ari_results,
        'linkage_matrix': [[float(x) for x in row] for row in linkage_matrix],
        'interpretation': 'ARI close to 1 = clusters match style/era; close to 0 = no relationship'
    }


def permutation_manova(df, n_permutations=1000):
    """Permutation-based MANOVA for IV distributions across artists."""

    performers = df['performer'].unique()

    # Prepare data: IV vectors as features, performer as group
    X = np.array([list(iv) for iv in df['iv']])
    y = df['performer'].values

    # Observed statistic: sum of between-group variance (Pillai's trace approximation)
    def compute_statistic(X, y):
        groups = np.unique(y)
        grand_mean = X.mean(axis=0)

        # Between-group sum of squares
        ssb = 0
        for g in groups:
            group_mask = (y == g)
            group_mean = X[group_mask].mean(axis=0)
            n_g = group_mask.sum()
            ssb += n_g * np.sum((group_mean - grand_mean) ** 2)

        # Total sum of squares
        sst = np.sum((X - grand_mean) ** 2)

        # Ratio (like eta-squared)
        return ssb / sst if sst > 0 else 0

    observed_stat = compute_statistic(X, y)

    # Permutation test
    perm_stats = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_stats.append(compute_statistic(X, y_perm))

    perm_stats = np.array(perm_stats)
    p_value = np.mean(perm_stats >= observed_stat)

    # Effect size: eta-squared (observed_stat is already this)
    eta_squared = observed_stat

    return {
        'test': 'Permutation MANOVA (IV distributions)',
        'observed_statistic': float(observed_stat),
        'statistic_name': 'eta_squared (between-group variance ratio)',
        'n_permutations': n_permutations,
        'p_value': float(p_value),
        'effect_size': float(eta_squared),
        'effect_size_name': 'eta_squared',
        'significant': bool(p_value < 0.05),
        'interpretation': 'Tests whether IV distributions differ across artists'
    }


def discriminant_analysis(df):
    """Linear Discriminant Analysis to find which IV dimensions separate artists."""

    # Prepare data
    X = np.array([list(iv) for iv in df['iv']])
    y = df['performer'].values

    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    # Get coefficients for each discriminant function
    iv_labels = ['m2/M7', 'M2/m7', 'm3/M6', 'M3/m6', 'P4/P5', 'TT']  # IV dimension names

    # Scalings (coefficients)
    n_components = min(lda.scalings_.shape[1], 5)  # Top 5 or fewer

    discriminant_functions = []
    for i in range(n_components):
        coeffs = lda.scalings_[:, i]
        explained_var = lda.explained_variance_ratio_[i] if i < len(lda.explained_variance_ratio_) else 0

        # Rank dimensions by absolute coefficient
        dim_importance = sorted(
            zip(iv_labels, coeffs, np.abs(coeffs)),
            key=lambda x: x[2],
            reverse=True
        )

        discriminant_functions.append({
            'function': i + 1,
            'explained_variance_ratio': float(explained_var),
            'coefficients': {label: float(coef) for label, coef, _ in dim_importance},
            'top_dimensions': [label for label, _, _ in dim_importance[:3]]
        })

    # Overall classification accuracy (leave-one-out would be better, but this is indicative)
    accuracy = lda.score(X, y)

    # Which dimensions matter most overall?
    overall_importance = np.abs(lda.scalings_).mean(axis=1)
    dimension_ranking = sorted(
        zip(iv_labels, overall_importance),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        'test': 'Linear Discriminant Analysis',
        'n_artists': len(np.unique(y)),
        'n_phrases': len(y),
        'n_discriminant_functions': n_components,
        'training_accuracy': float(accuracy),
        'discriminant_functions': discriminant_functions,
        'overall_dimension_importance': {label: float(imp) for label, imp in dimension_ranking},
        'interpretation': 'Higher importance = dimension better separates artists'
    }


def main():
    # Setup
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading phrase data...")
    df = load_all_phrases()
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Loaded {len(df)} phrases from {df['performer'].nunique()} artists")

    # === DESCRIPTIVE STATS ===
    print("\n=== DESCRIPTIVE STATISTICS ===")

    vocab_df, top_ivs_df = compute_descriptive_stats(df)
    vocab_df = vocab_df.sort_values('total_phrases', ascending=False)

    print("\nVocabulary by artist:")
    print(vocab_df[['performer', 'total_phrases', 'unique_ivs', 'type_token_ratio',
                    'mean_phrase_length', 'mean_chord_segments']].to_string(index=False))

    vocab_df.to_csv(ANALYSIS_DIR / 'vocabulary_by_artist.csv', index=False)
    top_ivs_df.to_csv(ANALYSIS_DIR / 'top_ivs_by_artist.csv', index=False)

    # === JACCARD SIMILARITY ===
    print("\n=== JACCARD SIMILARITY MATRIX ===")

    jaccard_df = compute_jaccard_similarity(df)
    jaccard_df.to_csv(ANALYSIS_DIR / 'jaccard_similarity_matrix.csv')

    print("\nMost similar artist pairs:")
    # Get upper triangle pairs
    pairs = []
    for i, p1 in enumerate(jaccard_df.index):
        for j, p2 in enumerate(jaccard_df.columns):
            if i < j:
                pairs.append((p1, p2, jaccard_df.iloc[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    for p1, p2, sim in pairs[:5]:
        print(f"  {p1} <-> {p2}: {sim:.3f}")

    # === KRUSKAL-WALLIS ===
    print("\n=== KRUSKAL-WALLIS TEST (Phrase Lengths) ===")

    kw_results = kruskal_wallis_phrase_length(df)
    print(f"H = {kw_results['h_statistic']:.2f}, p = {kw_results['p_value']:.4f}, "
          f"ε² = {kw_results['effect_size']:.3f}")
    print(f"Significant: {kw_results['significant']}")

    with open(ANALYSIS_DIR / 'phrase_length_kruskal_wallis.json', 'w') as f:
        json.dump(kw_results, f, indent=2)

    # === ANCOVA (with tempo covariate) ===
    print("\n=== ANCOVA TEST (Phrase Lengths, controlling for Tempo) ===")

    ancova_results = ancova_phrase_length(df)
    pe = ancova_results['performer_effect']
    tc = ancova_results['tempo_covariate']
    print(f"Performer effect: F = {pe['f_statistic']:.2f}, p = {pe['p_value']:.4f}, "
          f"η²p = {pe['partial_eta_squared']:.3f}")
    print(f"Tempo covariate:  F = {tc['f_statistic']:.2f}, p = {tc['p_value']:.4f}, "
          f"η²p = {tc['partial_eta_squared']:.3f}")
    print(f"Tempo coefficient: {tc['coefficient']:.4f} ({tc['interpretation']})")
    print(f"Model R²: {ancova_results['model_r_squared']:.3f}")

    with open(ANALYSIS_DIR / 'phrase_length_ancova.json', 'w') as f:
        json.dump(ancova_results, f, indent=2)

    # === CLUSTER ANALYSIS ===
    print("\n=== CLUSTER ANALYSIS ===")

    cluster_results = cluster_analysis(df, metadata)

    print("\nCluster assignments (k=3):")
    k3_clusters = cluster_results['cluster_assignments']['k=3']
    for cluster_id in sorted(set(k3_clusters.values())):
        members = [p for p, c in k3_clusters.items() if c == cluster_id]
        styles = [cluster_results['style_labels'][p] for p in members]
        print(f"  Cluster {cluster_id}: {', '.join(members)}")
        print(f"    Styles: {', '.join(styles)}")

    print(f"\nAdjusted Rand Index vs Style:")
    for k, ari in cluster_results['adjusted_rand_index_vs_style'].items():
        print(f"  {k}: {ari:.3f}")

    with open(ANALYSIS_DIR / 'cluster_analysis.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json.dump(cluster_results, f, indent=2)

    # === PERMUTATION MANOVA ===
    print("\n=== PERMUTATION MANOVA (IV Distributions) ===")

    manova_results = permutation_manova(df, n_permutations=1000)
    print(f"η² = {manova_results['effect_size']:.4f}, p = {manova_results['p_value']:.4f}")
    print(f"Significant: {manova_results['significant']}")

    with open(ANALYSIS_DIR / 'permutation_manova.json', 'w') as f:
        json.dump(manova_results, f, indent=2)

    # === DISCRIMINANT ANALYSIS ===
    print("\n=== DISCRIMINANT ANALYSIS ===")

    lda_results = discriminant_analysis(df)
    print(f"Training accuracy: {lda_results['training_accuracy']:.3f}")
    print("\nIV dimension importance (for separating artists):")
    for dim, imp in lda_results['overall_dimension_importance'].items():
        print(f"  {dim}: {imp:.3f}")

    with open(ANALYSIS_DIR / 'discriminant_analysis.json', 'w') as f:
        json.dump(lda_results, f, indent=2)

    # === SUMMARY ===
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output directory: {ANALYSIS_DIR}/")
    print("\nFiles created:")
    for f in sorted(ANALYSIS_DIR.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
