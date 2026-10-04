#!/usr/bin/env python3
"""
Two checks the earlier scripts get wrong or omit.

1. NESTING. Phrases are nested in solos, which are nested in artists. Scripts 03
   and 05 test artist differences by treating all 1,170 phrases as independent
   observations, which they are not: 184 Rollins phrases come from six
   performances, not 184. The test statistic is fine; the null distribution is
   not, because shuffling phrase labels destroys a structure the data has.

   The fix keeps the statistic and rebuilds the null by permuting artist labels
   across whole SOLOS, so every solo keeps its phrases together. Both the
   phrase-length comparison and the multivariate IV comparison are re-run that
   way, and reported beside the phrase-level p-values they replace.

2. BLUES VOCABULARY AGAINST THE HARMONIC NULL. Script 11 asks whether density,
   entropy and the dissonance ratio exceed what the chord-scale produces. It
   never asks the same of the blues catalogue itself, which is the measure the
   paper leans on hardest. A blues interval vector may simply be what these
   changes hand you. Same null as script 11: keep each phrase's pitch-class
   count, redraw which ones from the scales in force.

    python3 12_nesting_and_blues_null.py
"""
import collections
import csv
import glob
import json
import math
import random
import re
import statistics as st
from pathlib import Path

import numpy as np

DATA = Path('data')
ANALYSIS = DATA / 'analysis'
SEED = 20261004
N_PERM = 10000
N_NULL = 200

SCALES = {'maj': [0, 2, 4, 5, 7, 9, 11], 'dom': [0, 2, 4, 5, 7, 9, 10],
          'min': [0, 2, 3, 5, 7, 9, 10], 'halfd': [0, 1, 3, 5, 6, 8, 10],
          'dim': [0, 2, 3, 5, 6, 8, 9, 11]}
ROOTS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def chord_scale(sym):
    if not sym or sym == 'NC':
        return None
    m = re.match(r'^([A-G])([b#]?)(.*)$', sym)
    if not m:
        return None
    root = (ROOTS[m.group(1)] + (1 if m.group(2) == '#' else -1 if m.group(2) == 'b' else 0)) % 12
    suf = m.group(3).split('/')[0]
    kind = ('maj' if suf.startswith('j') or suf == '' else
            'halfd' if suf.startswith('m7b5') else
            'dim' if suf.startswith('o') else
            'min' if suf.startswith('-') else 'dom')
    return [(root + s) % 12 for s in SCALES[kind]]


def interval_vector(pcs):
    iv = [0] * 6
    p = sorted(set(pcs))
    for a in range(len(p)):
        for b in range(a + 1, len(p)):
            d = (p[b] - p[a]) % 12
            iv[min(d, 12 - d) - 1] += 1
    return iv


def load_phrases():
    out = []
    for f in sorted(glob.glob(str(DATA / 'phrases' / '*.json'))):
        d = json.load(open(f))
        for p in d['phrases']:
            if p.get('pcs') and len(set(p['pcs'])) > 1:
                out.append({'melid': str(d['melid']), 'performer': d['performer'],
                            'pcs': p['pcs'], 'iv': tuple(p['iv']),
                            'segments': p.get('chord_segments', [])})
    return out


# ------------------------------------------------------------------ 1. nesting
def kruskal_h(groups):
    """Kruskal-Wallis H on a list of lists, ties corrected."""
    allv = [v for g in groups for v in g]
    n = len(allv)
    order = sorted(range(n), key=lambda i: allv[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    pos, h = 0, 0.0
    for g in groups:
        rs = sum(ranks[pos:pos + len(g)])
        h += rs * rs / len(g)
        pos += len(g)
    h = 12.0 / (n * (n + 1)) * h - 3 * (n + 1)
    counts = collections.Counter(allv)
    tie = 1 - sum(c ** 3 - c for c in counts.values()) / (n ** 3 - n)
    return h / tie if tie else h


def eta_squared(vectors, labels):
    """Between-group share of total variance, the permutation MANOVA statistic."""
    X = np.asarray(vectors, dtype=float)
    grand = X.mean(axis=0)
    total = ((X - grand) ** 2).sum()
    between = 0.0
    for lab in set(labels):
        idx = [i for i, l in enumerate(labels) if l == lab]
        between += len(idx) * ((X[idx].mean(axis=0) - grand) ** 2).sum()
    return between / total if total else 0.0


def nested_tests(phrases):
    ts = {r['melid']: r for r in csv.DictReader(open(DATA / 'corpus_metadata.csv'))}
    rows = [r for r in csv.DictReader(open(DATA / 'timeseries' / 'timeseries_data.csv'))
            if r.get('n_notes')]
    solo_of = {}
    length, ivs, keys = [], [], []
    for r in rows:
        key = (r['melid'], r['phrase_id'])
        solo_of[key] = r['melid']
        length.append(float(r['n_notes']))
        keys.append(key)
    iv_by = {(p['melid'], str(i)): p['iv'] for p in phrases for i in [0]}
    # interval vectors keyed by (melid, phrase_id) from the phrase files
    ivmap = {}
    for f in glob.glob(str(DATA / 'phrases' / '*.json')):
        d = json.load(open(f))
        for p in d['phrases']:
            if p.get('iv'):
                ivmap[(str(d['melid']), str(p['phrase_id']))] = p['iv']
    use = [i for i, k in enumerate(keys) if k in ivmap]
    length = [length[i] for i in use]
    keys = [keys[i] for i in use]
    ivs = [ivmap[k] for k in keys]
    solos = [k[0] for k in keys]
    artist_of_solo = {m: ts[m]['performer'] for m in set(solos) if m in ts}
    solos = [s for s in solos]
    labels = [artist_of_solo.get(s) for s in solos]
    ok = [i for i, l in enumerate(labels) if l]
    length = [length[i] for i in ok]; ivs = [ivs[i] for i in ok]
    solos = [solos[i] for i in ok]; labels = [labels[i] for i in ok]

    def grouped(vals, labs):
        g = collections.defaultdict(list)
        for v, l in zip(vals, labs):
            g[l].append(v)
        return list(g.values())

    obs_h = kruskal_h(grouped(length, labels))
    obs_eta = eta_squared(ivs, labels)

    solo_ids = sorted(set(solos))
    solo_label = {s: artist_of_solo[s] for s in solo_ids}
    rng = random.Random(SEED)
    hit_h = hit_eta = 0
    # permute artist labels ACROSS SOLOS, so phrases stay with their solo
    for _ in range(N_PERM):
        names = [solo_label[s] for s in solo_ids]
        rng.shuffle(names)
        lookup = dict(zip(solo_ids, names))
        perm = [lookup[s] for s in solos]
        if kruskal_h(grouped(length, perm)) >= obs_h:
            hit_h += 1
        if eta_squared(ivs, perm) >= obs_eta:
            hit_eta += 1
    return {'n_phrases': len(length), 'n_solos': len(solo_ids),
            'n_artists': len(set(labels)),
            'phrase_length': {'H': obs_h, 'p_solo_permutation': (hit_h + 1) / (N_PERM + 1)},
            'iv_distribution': {'eta_squared': obs_eta,
                                'p_solo_permutation': (hit_eta + 1) / (N_PERM + 1)}}


# -------------------------------------------------------- 2. blues null
def blues_null(phrases, catalogue):
    rng = random.Random(SEED)
    obs = sum(1 for p in phrases if p['iv'] in catalogue) / len(phrases)
    by_artist_obs = collections.defaultdict(list)
    for p in phrases:
        by_artist_obs[p['performer']].append(1 if p['iv'] in catalogue else 0)

    runs, per_artist = [], collections.defaultdict(list)
    for _ in range(N_NULL):
        hits = 0
        a_hits = collections.defaultdict(int)
        a_n = collections.defaultdict(int)
        for p in phrases:
            k = len(set(p['pcs']))
            pool = sorted({pc for s in p['segments'] for pc in (chord_scale(s.get('chord')) or [])})
            if len(pool) < k:
                pool = sorted(set(pool) | set(range(12)))
            v = tuple(interval_vector(rng.sample(pool, min(k, len(pool)))))
            got = 1 if v in catalogue else 0
            hits += got
            a_hits[p['performer']] += got
            a_n[p['performer']] += 1
        runs.append(hits / len(phrases))
        for a in a_n:
            per_artist[a].append(a_hits[a] / a_n[a])

    beat = sum(1 for r in runs if r >= obs)
    out = {'observed': obs, 'null_mean': st.mean(runs), 'null_sd': st.pstdev(runs),
           'z': (obs - st.mean(runs)) / (st.pstdev(runs) or 1),
           'p_one_sided': (beat + 1) / (N_NULL + 1), 'artists': {}}
    for a, vals in per_artist.items():
        o = st.mean(by_artist_obs[a])
        out['artists'][a] = {'observed': o, 'null_mean': st.mean(vals),
                             'z': (o - st.mean(vals)) / (st.pstdev(vals) or 1)}
    return out



# ------------------------------------------------------- 3. TTR rarefaction
def ttr_rarefaction(phrases, n_draws=2000):
    """Type-token ratio at a common sample size.

    TTR is unique interval vectors over phrases, and it falls as the denominator
    grows: an artist with 29 phrases can reach 1.0, one with 184 would need 184
    distinct vectors to do the same. The raw spread from 0.42 to 0.93 therefore
    confounds vocabulary breadth with how much of each player the corpus holds.

    Rarefaction removes that: every artist is repeatedly subsampled to the
    smallest artist's phrase count and the ratio recomputed, so the denominator
    is identical for all of them.
    """
    rng = random.Random(SEED)
    by = collections.defaultdict(list)
    for p in phrases:
        by[p['performer']].append(p['iv'])
    k = min(len(v) for v in by.values())
    out = {'common_n': k, 'artists': {}}
    for a, ivs in by.items():
        raw = len(set(ivs)) / len(ivs)
        draws = [len(set(rng.sample(ivs, k))) / k for _ in range(n_draws)]
        draws.sort()
        out['artists'][a] = {
            'n_phrases': len(ivs), 'raw_ttr': raw,
            'rarefied_mean': st.mean(draws),
            'ci_low': draws[int(0.025 * n_draws)], 'ci_high': draws[int(0.975 * n_draws)]}
    return out


def main():
    phrases = load_phrases()
    src = open('09_granger_causality.py').read()
    ns = {}
    exec(src[src.index('BLUES_IVS = {'):src.index('}', src.index('BLUES_IVS = {')) + 1], ns)
    catalogue = set(ns['BLUES_IVS'])

    print('1. NESTING: phrases are not independent observations')
    nest = nested_tests(phrases)
    print('   %d phrases in %d solos by %d artists' % (nest['n_phrases'], nest['n_solos'], nest['n_artists']))
    print('   phrase length   H = %8.1f   p (solo-level permutation) = %.4f'
          % (nest['phrase_length']['H'], nest['phrase_length']['p_solo_permutation']))
    print('   IV distribution eta2 = %6.4f   p (solo-level permutation) = %.4f'
          % (nest['iv_distribution']['eta_squared'], nest['iv_distribution']['p_solo_permutation']))

    print('\n2. BLUES VOCABULARY against the chord-scale null')
    bn = blues_null(phrases, catalogue)
    print('   corpus blues rate observed %.4f   null %.4f +- %.4f   z = %+.2f   p = %.4f'
          % (bn['observed'], bn['null_mean'], bn['null_sd'], bn['z'], bn['p_one_sided']))
    print('   per artist (observed, null, z):')
    for a, v in sorted(bn['artists'].items(), key=lambda kv: -kv[1]['z']):
        print('     %-20s %.3f  %.3f  %+6.2f' % (a, v['observed'], v['null_mean'], v['z']))

    print('\n3. TYPE-TOKEN RATIO at a common sample size')
    rf = ttr_rarefaction(phrases)
    print('   every artist subsampled to %d phrases, 2000 draws' % rf['common_n'])
    print('   %-20s %5s %8s %8s  %s' % ('artist', 'n', 'raw', 'rarefied', '95% CI'))
    for a, v in sorted(rf['artists'].items(), key=lambda kv: -kv[1]['rarefied_mean']):
        print('     %-20s %4d %7.3f %8.3f  [%.3f, %.3f]'
              % (a, v['n_phrases'], v['raw_ttr'], v['rarefied_mean'], v['ci_low'], v['ci_high']))
    raw_order = [a for a, _ in sorted(rf['artists'].items(), key=lambda kv: -kv[1]['raw_ttr'])]
    rar_order = [a for a, _ in sorted(rf['artists'].items(), key=lambda kv: -kv[1]['rarefied_mean'])]
    moved = sum(1 for i, a in enumerate(raw_order) if rar_order.index(a) != i)
    print('   artists whose rank changes once the denominator is equalised: %d of %d'
          % (moved, len(raw_order)))

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS / 'nesting_and_blues_null.json', 'w') as fh:
        json.dump({'nesting': nest, 'blues_null': bn, 'ttr_rarefaction': rf}, fh, indent=2)
    print('\nSaved: %s' % (ANALYSIS / 'nesting_and_blues_null.json'))


if __name__ == '__main__':
    main()
