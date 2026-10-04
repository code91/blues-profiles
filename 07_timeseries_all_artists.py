#!/usr/bin/env python3
"""
07_timeseries_all_artists.py

Generates aligned time series plots for cross-artist comparison.

Methodology:
1. Find minimum number of choruses common to all artists
2. For each chorus, resample phrases to fixed N positions (default: 4)
3. Average across tunes per artist at each position
4. Min-max scale (0-1) for comparison
5. Mark chorus boundaries
6. DTW analysis for trajectory similarity

Output:
- data/figures/timeseries_aligned_grid.png
- data/figures/timeseries_full_trajectory_grid.png
- data/figures/dtw_similarity_matrix.png
- data/analysis/dtw_results.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Configuration
PHRASES_DIR = Path("data/phrases")
ANALYSIS_DIR = Path("data/analysis")
FIGURES_DIR = Path("data/figures")

POSITIONS_PER_CHORUS = 4  # Resample each chorus to this many positions
MIN_CHORUSES = 2  # Minimum choruses to include (set manually for useful comparison)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def dtw_distance(s1, s2):
    """
    Compute Dynamic Time Warping distance between two sequences.
    Returns the DTW distance (lower = more similar).
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )

    return dtw_matrix[n, m]


def dtw_distance_normalized(s1, s2):
    """DTW distance normalized by path length."""
    n, m = len(s1), len(s2)
    return dtw_distance(s1, s2) / (n + m)


def compute_density(iv):
    """Pitch-class density = sum(IV) = C(cardinality, 2).

    Renamed from "density" after review: the sum of an interval vector is a
    bijection with pitch-class-set size and measures density, not density.
    See 06_timeseries_individual_artists.py for the full note.
    """
    return sum(iv)


def load_phrase_data_with_chorus():
    """Load phrase data with chorus info and compute metrics."""
    all_data = []

    # Load anticipation data
    antic_path = ANALYSIS_DIR / 'anticipation_metrics.csv'
    if antic_path.exists():
        antic_df = pd.read_csv(antic_path)
        antic_lookup = antic_df.set_index(['melid', 'phrase_id'])['loading_balance'].to_dict()
    else:
        antic_lookup = {}

    for filepath in PHRASES_DIR.glob("*.json"):
        if filepath.name == "summary.csv":
            continue
        with open(filepath) as f:
            data = json.load(f)

        melid = data['melid']
        performer = data['performer']
        title = data['title']

        for phrase in data['phrases']:
            iv = phrase['iv']
            density = compute_density(iv)
            anticipation = antic_lookup.get((melid, phrase['phrase_id']), np.nan)

            all_data.append({
                'melid': melid,
                'performer': performer,
                'title': title,
                'phrase_id': phrase['phrase_id'],
                'chorus_id': phrase['chorus_id'],
                'density': density,
                'anticipation': anticipation
            })

    return pd.DataFrame(all_data)


def get_min_choruses_per_artist(df):
    """Find minimum number of choruses each artist has across all their tunes."""
    # For each tune, get max chorus_id
    tune_choruses = df.groupby(['performer', 'melid'])['chorus_id'].max().reset_index()
    tune_choruses.columns = ['performer', 'melid', 'n_choruses']

    # For each artist, get minimum across their tunes
    artist_min = tune_choruses.groupby('performer')['n_choruses'].min()

    return artist_min.to_dict()


def resample_chorus(values, n_positions=POSITIONS_PER_CHORUS):
    """Resample a variable-length array to fixed n_positions using linear interpolation."""
    if len(values) == 0:
        return np.full(n_positions, np.nan)
    if len(values) == 1:
        return np.full(n_positions, values[0])

    # Original positions (0 to 1)
    x_orig = np.linspace(0, 1, len(values))
    # Target positions
    x_new = np.linspace(0, 1, n_positions)

    # Interpolate
    f = interpolate.interp1d(x_orig, values, kind='linear', fill_value='extrapolate')
    return f(x_new)


def build_aligned_timeseries(df, metric, n_choruses):
    """
    Build aligned time series for a metric.

    Returns dict: performer -> array of shape (n_choruses * POSITIONS_PER_CHORUS,)
    """
    results = {}

    for performer in df['performer'].unique():
        artist_df = df[df['performer'] == performer]
        tunes = artist_df['melid'].unique()

        # Collect resampled data from each tune
        tune_arrays = []

        for melid in tunes:
            tune_df = artist_df[artist_df['melid'] == melid]

            # Check if this tune has enough choruses
            max_chorus = tune_df['chorus_id'].max()
            if max_chorus < n_choruses:
                continue  # Skip tunes with fewer choruses than minimum

            tune_resampled = []

            for chorus in range(1, n_choruses + 1):
                chorus_df = tune_df[tune_df['chorus_id'] == chorus].sort_values('phrase_id')
                values = chorus_df[metric].values

                # Handle NaN
                values = np.nan_to_num(values, nan=0.0)

                resampled = resample_chorus(values, POSITIONS_PER_CHORUS)
                tune_resampled.extend(resampled)

            tune_arrays.append(tune_resampled)

        if tune_arrays:
            # Average across tunes
            tune_arrays = np.array(tune_arrays)
            results[performer] = np.nanmean(tune_arrays, axis=0)
        else:
            results[performer] = None

    return results


def min_max_scale(values):
    """Scale values to 0-1 range."""
    values = np.array(values)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax - vmin == 0:
        return np.zeros_like(values)
    return (values - vmin) / (vmax - vmin)


def plot_aligned_grid(density_data, anticipation_data, n_choruses, performers):
    """Plot aligned time series grid."""
    n_artists = len(performers)
    ncols = 3
    nrows = (n_artists + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))
    axes = axes.flatten()

    total_positions = n_choruses * POSITIONS_PER_CHORUS
    # X-axis from 0 to 1
    x = np.linspace(0, 1, total_positions)

    # Chorus boundaries (between 0 and 1)
    chorus_boundaries = [i / n_choruses for i in range(1, n_choruses)]

    lines_for_legend = None

    for idx, performer in enumerate(performers):
        ax = axes[idx]

        density = density_data.get(performer)
        anticipation = anticipation_data.get(performer)

        if density is None or anticipation is None:
            ax.set_visible(False)
            continue

        # Min-max scale
        density_scaled = min_max_scale(density)
        anticipation_scaled = min_max_scale(anticipation)

        # Plot
        line1, = ax.plot(x, density_scaled, color='steelblue', linewidth=2, label='Density')
        line2, = ax.plot(x, anticipation_scaled, color='forestgreen', linewidth=2, label='Anticipation')

        if lines_for_legend is None:
            lines_for_legend = [line1, line2]

        # Mark chorus boundaries
        for boundary in chorus_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

        # Add chorus labels
        for c in range(n_choruses):
            cx = (c + 0.5) / n_choruses
            ax.text(cx, -0.12, f'C{c+1}', ha='center', fontsize=8, color='gray',
                    transform=ax.get_xaxis_transform())

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Choruses')
        ax.set_ylabel('Scaled (0-1)')
        ax.set_title(performer)

    # Hide empty subplots
    for idx in range(len(performers), len(axes)):
        axes[idx].set_visible(False)

    # Single legend
    fig.legend(lines_for_legend, ['Density', 'Anticipation'],
               loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=11)

    plt.suptitle(f'Aligned Time Series: {n_choruses} Choruses\n(Averaged across tunes, min-max scaled)',
                 fontsize=13, y=1.06)
    plt.tight_layout()

    return fig


def build_full_timeseries(df, metric):
    """
    Build full time series for each artist using ALL their choruses.
    Returns dict: performer -> array of resampled values (variable length)
    """
    results = {}
    chorus_counts = {}

    for performer in df['performer'].unique():
        artist_df = df[df['performer'] == performer]
        tunes = artist_df['melid'].unique()

        # Find max choruses for this artist
        max_choruses = artist_df.groupby('melid')['chorus_id'].max().max()

        # Collect resampled data from each tune
        tune_arrays = []

        for melid in tunes:
            tune_df = artist_df[artist_df['melid'] == melid]
            tune_max_chorus = tune_df['chorus_id'].max()

            tune_resampled = []

            for chorus in range(1, tune_max_chorus + 1):
                chorus_df = tune_df[tune_df['chorus_id'] == chorus].sort_values('phrase_id')
                values = chorus_df[metric].values
                values = np.nan_to_num(values, nan=0.0)
                resampled = resample_chorus(values, POSITIONS_PER_CHORUS)
                tune_resampled.extend(resampled)

            # Pad shorter tunes with NaN to align
            while len(tune_resampled) < max_choruses * POSITIONS_PER_CHORUS:
                tune_resampled.append(np.nan)

            tune_arrays.append(tune_resampled[:max_choruses * POSITIONS_PER_CHORUS])

        if tune_arrays:
            tune_arrays = np.array(tune_arrays)
            # Average across tunes, ignoring NaN
            results[performer] = np.nanmean(tune_arrays, axis=0)
            chorus_counts[performer] = max_choruses

    return results, chorus_counts


def plot_full_trajectory_grid(density_data, anticipation_data, chorus_counts, performers):
    """Plot full trajectory grid showing all choruses per artist."""
    n_artists = len(performers)
    ncols = 3
    nrows = (n_artists + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.5 * nrows))
    axes = axes.flatten()

    lines_for_legend = None

    for idx, performer in enumerate(performers):
        ax = axes[idx]

        density = density_data.get(performer)
        anticipation = anticipation_data.get(performer)
        n_choruses = chorus_counts.get(performer, 1)

        if density is None or anticipation is None:
            ax.set_visible(False)
            continue

        # Remove NaN from end if present
        density = np.array(density)
        anticipation = np.array(anticipation)
        valid_mask = ~np.isnan(density)
        density = density[valid_mask]
        anticipation = anticipation[valid_mask]

        total_positions = len(density)
        x = np.linspace(0, 1, total_positions)

        # Min-max scale
        density_scaled = min_max_scale(density)
        anticipation_scaled = min_max_scale(anticipation)

        # Plot
        line1, = ax.plot(x, density_scaled, color='steelblue', linewidth=2, label='Density')
        line2, = ax.plot(x, anticipation_scaled, color='forestgreen', linewidth=2, label='Anticipation')

        if lines_for_legend is None:
            lines_for_legend = [line1, line2]

        # Chorus boundaries
        actual_choruses = total_positions // POSITIONS_PER_CHORUS
        for c in range(1, actual_choruses):
            boundary = c / actual_choruses
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(f'Choruses (n={actual_choruses})')
        ax.set_ylabel('Scaled (0-1)')
        ax.set_title(performer)

    for idx in range(len(performers), len(axes)):
        axes[idx].set_visible(False)

    fig.legend(lines_for_legend, ['Density', 'Anticipation'],
               loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=11)

    plt.suptitle('Full Trajectory Time Series\n(All choruses, averaged across tunes, min-max scaled)',
                 fontsize=13, y=1.06)
    plt.tight_layout()

    return fig


def compute_dtw_matrix(data_dict, performers):
    """Compute pairwise DTW distance matrix."""
    n = len(performers)
    dtw_matrix = np.zeros((n, n))

    for i, p1 in enumerate(performers):
        for j, p2 in enumerate(performers):
            if i < j:
                s1 = np.nan_to_num(data_dict[p1], nan=0.0)
                s2 = np.nan_to_num(data_dict[p2], nan=0.0)
                # Normalize before DTW
                s1 = min_max_scale(s1)
                s2 = min_max_scale(s2)
                dist = dtw_distance_normalized(s1, s2)
                dtw_matrix[i, j] = dist
                dtw_matrix[j, i] = dist

    return dtw_matrix


def compute_within_artist_dtw(density_data, anticipation_data, performers):
    """
    For each artist, compute DTW distance between their density and anticipation trajectories.
    Lower = more coupled/synchronized movement.
    """
    results = {}

    for performer in performers:
        c = density_data.get(performer)
        a = anticipation_data.get(performer)

        if c is None or a is None:
            continue

        c = np.nan_to_num(c, nan=0.0)
        a = np.nan_to_num(a, nan=0.0)

        # Normalize
        c_scaled = min_max_scale(c)
        a_scaled = min_max_scale(a)

        dist = dtw_distance_normalized(c_scaled, a_scaled)
        results[performer] = dist

    return results


def plot_dtw_heatmap(dtw_matrix, performers, metric_name):
    """Plot DTW distance matrix as heatmap with dendrogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(dtw_matrix, cmap='viridis_r')
    ax.set_xticks(range(len(performers)))
    ax.set_yticks(range(len(performers)))
    ax.set_xticklabels(performers, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(performers, fontsize=9)
    ax.set_title(f'DTW Distance Matrix: {metric_name}\n(Lower = more similar)')
    plt.colorbar(im, ax=ax, label='DTW Distance')

    # Dendrogram
    ax = axes[1]
    # Convert distance matrix to condensed form
    condensed = squareform(dtw_matrix)
    Z = linkage(condensed, method='ward')
    dendrogram(Z, labels=performers, ax=ax, leaf_rotation=45)
    ax.set_title(f'Hierarchical Clustering: {metric_name}')
    ax.set_ylabel('DTW Distance')

    plt.tight_layout()
    return fig


def plot_within_artist_dtw(within_dtw, performers):
    """Plot bar chart of within-artist DTW (density vs anticipation coupling)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by DTW distance
    sorted_performers = sorted(performers, key=lambda p: within_dtw.get(p, 0))
    values = [within_dtw.get(p, 0) for p in sorted_performers]

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_performers)))

    ax.barh(sorted_performers, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('DTW Distance (Density vs Anticipation)')
    ax.set_title('Density-Anticipation Coupling by Artist\n(Lower = more synchronized trajectories)')

    plt.tight_layout()
    return fig


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading phrase data...")
    df = load_phrase_data_with_chorus()
    print(f"Total phrases: {len(df)}")

    # Find minimum choruses per artist
    artist_min_choruses = get_min_choruses_per_artist(df)
    print("\nMinimum choruses per artist:")
    for artist, n in sorted(artist_min_choruses.items(), key=lambda x: x[1]):
        print(f"  {artist}: {n}")

    # Global minimum across all artists
    global_min = min(artist_min_choruses.values())
    print(f"\nGlobal minimum choruses: {global_min}")
    print(f"Using MIN_CHORUSES = {MIN_CHORUSES} for comparison")

    # Filter to artists who have at least MIN_CHORUSES
    eligible_artists = [a for a, n in artist_min_choruses.items() if n >= MIN_CHORUSES]
    print(f"Artists with >= {MIN_CHORUSES} choruses: {len(eligible_artists)}")

    # Filter dataframe
    df = df[df['performer'].isin(eligible_artists)]

    # Build aligned time series
    print(f"\nBuilding aligned time series ({MIN_CHORUSES} choruses × {POSITIONS_PER_CHORUS} positions)...")

    density_data = build_aligned_timeseries(df, 'density', MIN_CHORUSES)
    anticipation_data = build_aligned_timeseries(df, 'anticipation', MIN_CHORUSES)

    # Filter to artists with data
    performers = sorted([p for p in density_data.keys() if density_data[p] is not None])
    print(f"Artists with sufficient data: {len(performers)}")

    # Summary stats
    print("\n=== ALIGNED TIME SERIES SUMMARY ===")
    for performer in performers:
        c = density_data[performer]
        a = anticipation_data[performer]
        print(f"{performer:20} density: {np.mean(c):.1f} (±{np.std(c):.1f})  "
              f"anticipation: {np.mean(a):.2f} (±{np.std(a):.2f})")

    # Plot aligned grid
    print("\n=== GENERATING ALIGNED GRID ===")
    fig = plot_aligned_grid(density_data, anticipation_data, MIN_CHORUSES, performers)
    plt.savefig(FIGURES_DIR / 'timeseries_aligned_grid.png')
    plt.savefig(FIGURES_DIR / 'timeseries_aligned_grid.pdf')
    plt.close()
    print("  timeseries_aligned_grid.png")

    # === FULL TRAJECTORY ANALYSIS (all choruses) ===
    print("\n=== BUILDING FULL TRAJECTORIES ===")

    # Reload full dataframe (not filtered)
    df_full = load_phrase_data_with_chorus()

    full_density, chorus_counts = build_full_timeseries(df_full, 'density')
    full_anticipation, _ = build_full_timeseries(df_full, 'anticipation')

    all_performers = sorted([p for p in full_density.keys() if full_density[p] is not None])
    print(f"Artists with full trajectories: {len(all_performers)}")

    for performer in all_performers:
        print(f"  {performer}: {chorus_counts[performer]} choruses")

    # Plot full trajectory grid
    print("\n=== GENERATING FULL TRAJECTORY GRID ===")
    fig = plot_full_trajectory_grid(full_density, full_anticipation, chorus_counts, all_performers)
    plt.savefig(FIGURES_DIR / 'timeseries_full_trajectory_grid.png')
    plt.savefig(FIGURES_DIR / 'timeseries_full_trajectory_grid.pdf')
    plt.close()
    print("  timeseries_full_trajectory_grid.png")

    # === DTW ANALYSIS ===
    print("\n=== DTW ANALYSIS ===")

    # DTW matrix for density trajectories
    print("Computing DTW matrix for density...")
    dtw_density = compute_dtw_matrix(full_density, all_performers)

    print("Computing DTW matrix for anticipation...")
    dtw_anticipation = compute_dtw_matrix(full_anticipation, all_performers)

    # Within-artist DTW (density vs anticipation coupling)
    print("Computing within-artist DTW (density-anticipation coupling)...")
    within_dtw = compute_within_artist_dtw(full_density, full_anticipation, all_performers)

    # Print DTW summary
    print("\n=== DTW SIMILARITY SUMMARY ===")
    print("\nMost similar pairs (Density):")
    pairs = []
    for i, p1 in enumerate(all_performers):
        for j, p2 in enumerate(all_performers):
            if i < j:
                pairs.append((p1, p2, dtw_density[i, j]))
    pairs.sort(key=lambda x: x[2])
    for p1, p2, dist in pairs[:5]:
        print(f"  {p1} ↔ {p2}: {dist:.3f}")

    print("\nMost different pairs (Density):")
    for p1, p2, dist in pairs[-5:]:
        print(f"  {p1} ↔ {p2}: {dist:.3f}")

    print("\nDensity-Anticipation Coupling (within artist):")
    sorted_coupling = sorted(within_dtw.items(), key=lambda x: x[1])
    for performer, dist in sorted_coupling:
        coupling_level = "HIGH" if dist < 0.05 else "MEDIUM" if dist < 0.1 else "LOW"
        print(f"  {performer:20}: {dist:.3f} ({coupling_level})")

    # Save DTW results
    dtw_results = {
        'performers': all_performers,
        'chorus_counts': {k: int(v) for k, v in chorus_counts.items()},
        'dtw_density_matrix': dtw_density.tolist(),
        'dtw_anticipation_matrix': dtw_anticipation.tolist(),
        'within_artist_dtw': {k: float(v) for k, v in within_dtw.items()},
        'most_similar_density': [(p1, p2, float(d)) for p1, p2, d in pairs[:5]],
        'most_different_density': [(p1, p2, float(d)) for p1, p2, d in pairs[-5:]]
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / 'dtw_results.json', 'w') as f:
        json.dump(dtw_results, f, indent=2)

    # Plot DTW heatmaps
    print("\n=== GENERATING DTW FIGURES ===")

    fig = plot_dtw_heatmap(dtw_density, all_performers, 'Density')
    plt.savefig(FIGURES_DIR / 'dtw_density_heatmap.png')
    plt.savefig(FIGURES_DIR / 'dtw_density_heatmap.pdf')
    plt.close()
    print("  dtw_density_heatmap.png")

    fig = plot_dtw_heatmap(dtw_anticipation, all_performers, 'Anticipation')
    plt.savefig(FIGURES_DIR / 'dtw_anticipation_heatmap.png')
    plt.savefig(FIGURES_DIR / 'dtw_anticipation_heatmap.pdf')
    plt.close()
    print("  dtw_anticipation_heatmap.png")

    fig = plot_within_artist_dtw(within_dtw, all_performers)
    plt.savefig(FIGURES_DIR / 'dtw_within_artist_coupling.png')
    plt.savefig(FIGURES_DIR / 'dtw_within_artist_coupling.pdf')
    plt.close()
    print("  dtw_within_artist_coupling.png")

    print("\n" + "="*60)
    print("ALIGNED TIME SERIES + DTW ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  {FIGURES_DIR / 'timeseries_aligned_grid.png'}")
    print(f"  {FIGURES_DIR / 'timeseries_full_trajectory_grid.png'}")
    print(f"  {FIGURES_DIR / 'dtw_density_heatmap.png'}")
    print(f"  {FIGURES_DIR / 'dtw_anticipation_heatmap.png'}")
    print(f"  {FIGURES_DIR / 'dtw_within_artist_coupling.png'}")
    print(f"  {ANALYSIS_DIR / 'dtw_results.json'}")


if __name__ == "__main__":
    main()
