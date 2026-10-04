# Blues Connotation

**Cross-Artist Patterns in Improvisational Decision-Making Over the Blues Form**

This repository contains the analysis code and data processing pipeline for a computational musicology study examining how jazz improvisers navigate the 12-bar blues form. Using 48 solos from 13 artists in the Weimar Jazz Database (1925–1991), we construct "blues profiles" combining interval vector distributions, phrase-level metrics, anticipation strategies, Granger causality patterns, and blues vocabulary deployment.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18053636.svg)](https://doi.org/10.5281/zenodo.18053636)

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{rubini2025blues,
  author = {Rubini, Mike},
  title = {Blues Connotation: Cross-Artist Patterns in Improvisational Decision-Making},
  year = {2025},
  doi = {10.5281/zenodo.18053636}
}
```

This work builds on:

```bibtex
@article{rubini2025bird,
  author = {Rubini, Mike},
  title = {"Bird" Over Time: A Time Series Analysis of Charlie Parker's Harmonic Complexity},
  journal = {Empirical Musicology Review},
  year = {2025},
  doi = {10.5281/zenodo.18037822}
}
```

## Data Source

This study uses the **Weimar Jazz Database (WJazzD)** version 2.1:

> Pfleiderer, M., Frieler, K., Abeßer, J., Zaddach, W.-G., & Burkhart, B. (Eds.). (2017). *Inside the Jazzomat: New Perspectives for Jazz Research*. Schott Campus.

The database is freely available at: https://jazzomat.hfm-weimar.de

**Note:** The WJazzD SQLite file (`wjazzd.db`) is not included in this repository. Download it from the Jazzomat website and place it in the project root.

## Requirements

```
python >= 3.10
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
music21 (optional, for notation examples)
```

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

## Project Structure

```
blues-profiles/
├── wjazzd.db                    # Weimar Jazz Database (not included, download separately)
├── 1_extract_blues_corpus.py    # Extract blues solos from WJazzD
├── 2_segment_phrases.py         # Phrase segmentation with nested IVs
├── 3_vocabulary_analysis.py     # Vocabulary and structure analysis
├── 4_visualizations.py          # Generate publication figures
├── 5_anticipation_analysis.py   # Anticipation metrics
├── 6_timeseries_visualization.py
├── 7_aligned_timeseries.py      # DTW trajectory analysis
├── 8_granger_causality.py       # Granger causality testing
├── 9_blues_vocabulary.py        # Blues IV catalog analysis
├── data/
│   ├── phrases/                 # Extracted phrase JSON files
│   ├── analysis/                # Analysis outputs (CSV, JSON)
│   └── figures/                 # Generated figures (PNG, PDF)
└── paper/
    ├── blues_profiles_paper.typ # Typst manuscript
    └── figures/                 # Paper figures
```

## Usage

Run scripts in order:

```bash
# 1. Extract blues corpus from WJazzD
python 1_extract_blues_corpus.py

# 2. Segment phrases and compute interval vectors
python 2_segment_phrases.py

# 3. Vocabulary and structure analysis
python 3_vocabulary_analysis.py

# 4. Generate visualizations
python 4_visualizations.py

# 5-9. Additional analyses
python 5_anticipation_analysis.py
python 6_timeseries_visualization.py
python 7_aligned_timeseries.py
python 8_granger_causality.py
python 9_blues_vocabulary.py
```

## Key Findings

- **Vocabulary diversity** (TTR) ranges from 0.42 (Coltrane) to 0.93 (Don Byas)
- **Anticipation profiles** differentiate front-loaders (Miles, J.J. Johnson) from back-loaders (Parker, Dolphy)
- **Granger causality** patterns are artist-specific: Coltrane builds momentum, Marsalis and Armstrong show recovery patterns
- **DTW trajectory clustering** does not align with historical era, suggesting individual cognitive fingerprints
- **Blues vocabulary**: Wayne Shorter shows highest blues quotient (25.6%), Coltrane lowest (3.0%)

## Related Tools

- [musicxml-to-pcs](https://github.com/code91/musicxml-to-pcs) — Extract pitch-class sets) interval vectors, and Forte numbers from MusicXML

## License

This work is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), consistent with the Weimar Jazz Database usage terms.

## Author

**Mike Rubini**  
music@mikerubini.com  
https://music.mikerubini.com

## Acknowledgments

Thanks to the Jazzomat Research Project at Hochschule für Musik Franz Liszt Weimar for making the Weimar Jazz Database freely available for research.

## Note on metric names (revision, October 2026)

What earlier versions called **complexity** is now called **density**, and the
change is substantive rather than cosmetic. The sum of an interval vector over a
pitch-class set of size *n* is C(n, 2) = n(n-1)/2 regardless of which pitch
classes it contains, so the quantity is a bijection with set size and carries no
information beyond it. The identity holds for all 1,170 phrases in this corpus.

Where harmonic complexity is meant, the pipeline now computes **interval
entropy**, the Shannon entropy of the interval-class distribution. It correlates
with density at only r = 0.545 and separates cases density cannot: a diminished
seventh scores 0.92 bits against a major triad's 1.59 despite twice the density.

**Dissonance is used as a ratio to density.** Raw dissonance is a weighted
sub-sum of the same six components, so the two correlate at r = 0.991 across the
corpus (0.982 to 0.995 within every artist), implying a variance inflation factor
near 59. The ratio brings that to 0.088.

### Reproducibility fix

`09_granger_causality.py` passed `verbose=False` to statsmodels'
`grangercausalitytests()`. That parameter was removed in statsmodels 0.15, and
because the call was wrapped in a broad exception handler the failure surfaced as
NaN for every test and NONE for every direction rather than as an error, so the
script appeared to run and produced an empty result set. The call no longer
passes it and falls back for older versions, and statsmodels is now pinned in
`requirements.txt`. With the fix, the published Density-Anticipation and
Length-Density tables reproduce exactly.
