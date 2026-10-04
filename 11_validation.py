#!/usr/bin/env python3
"""
Two checks the study needs and does not have.

  1. Held-out artist identification. Script 03 reports Linear Discriminant
     Analysis accuracy on the data it was fitted to, which cannot distinguish a
     real signal from memorisation. This refits the model 48 times, each time
     holding out one whole solo, and scores only the held-out predictions. A
     label permutation gives the empirical chance level, which is above 1/13
     because the classes are unbalanced.

  2. A null baseline for the interval-vector metrics. Without one, "Coltrane's
     type-token ratio is 0.42" has no scale: low compared to what? Two nulls are
     generated, each preserving the number of distinct pitch classes the player
     actually used in every phrase and discarding only WHICH ones:

        uniform      pitch classes drawn from all twelve
        chord-scale  drawn from the scale implied by the chord being played,
                     so the null keeps the harmony and discards only the choice
                     of notes within it

     The second is the control Reviewer 1 asked for on the companion study.

    python3 11_validation.py
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

DATA = Path('data')
ANALYSIS = DATA / 'analysis'
SEED = 20261004
N_NULL = 200           # null corpora per model
N_CHANCE_PERM = 1000   # label permutations for the chance level

FEATURES = ['density', 'entropy', 'dissonance_ratio', 'n_notes', 'cardinality']

# Chord quality to scale, as pitch classes above the root. Blues repertoire, so
# an unrecognised quality falls back to mixolydian rather than being dropped.
SCALES = {
    'maj':   [0, 2, 4, 5, 7, 9, 11],      # j7
    'dom':   [0, 2, 4, 5, 7, 9, 10],      # 7, sus7, altered extensions
    'min':   [0, 2, 3, 5, 7, 9, 10],      # -7
    'halfd': [0, 1, 3, 5, 6, 8, 10],      # m7b5
    'dim':   [0, 2, 3, 5, 6, 8, 9, 11],   # o
}
ROOTS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def chord_scale(sym):
    """Pitch classes implied by a WJazzD chord symbol, or None for NC."""
    if not sym or sym == 'NC':
        return None
    m = re.match(r'^([A-G])([b#]?)(.*)$', sym)
    if not m:
        return None
    root = (ROOTS[m.group(1)] + (1 if m.group(2) == '#' else -1 if m.group(2) == 'b' else 0)) % 12
    suf = m.group(3).split('/')[0]
    if suf.startswith('j'):        kind = 'maj'
    elif suf.startswith('m7b5'):   kind = 'halfd'
    elif suf.startswith('o'):      kind = 'dim'
    elif suf.startswith('-'):      kind = 'min'
    elif suf == '':                kind = 'maj'
    else:                          kind = 'dom'
    return [(root + s) % 12 for s in SCALES[kind]]


def interval_vector(pcs):
    iv = [0] * 6
    pcs = sorted(set(pcs))
    for a in range(len(pcs)):
        for b in range(a + 1, len(pcs)):
            d = (pcs[b] - pcs[a]) % 12
            iv[min(d, 12 - d) - 1] += 1
    return iv


def metrics(iv):
    total = sum(iv)
    if total <= 0:
        return None
    ent = -sum((c / total) * math.log2(c / total) for c in iv if c > 0)
    diss = iv[0] + 0.5 * iv[1] + 0.8 * iv[5]
    return {'density': total, 'entropy': ent, 'dissonance_ratio': diss / total}


def load_phrases():
    out = []
    for f in sorted(glob.glob(str(DATA / 'phrases' / '*.json'))):
        d = json.load(open(f))
        for p in d['phrases']:
            if not p.get('pcs'):
                continue
            out.append({'melid': str(d['melid']), 'performer': d['performer'],
                        'pcs': p['pcs'], 'segments': p.get('chord_segments', [])})
    return out


# ---------------------------------------------------------------- 1. held out
def leave_one_solo_out():
    meta = {r['melid']: r for r in csv.DictReader(open(DATA / 'corpus_metadata.csv'))}
    rows = [r for r in csv.DictReader(open(DATA / 'timeseries' / 'timeseries_data.csv'))
            if r.get('density')]
    by = collections.defaultdict(list)
    for r in rows:
        by[r['melid']].append(r)

    X, y, solos = [], [], []
    for mel, rs in by.items():
        if len(rs) < 5 or mel not in meta:
            continue
        v = []
        for f in FEATURES:
            xs = [float(x[f]) for x in rs]
            v += [st.mean(xs), st.pstdev(xs)]
        X.append(v); y.append(meta[mel]['performer']); solos.append(mel)
    X = np.array(X); y = np.array(y)

    def run(make, labels):
        c = 0
        for i in range(len(X)):
            mask = np.ones(len(X), dtype=bool); mask[i] = False
            if len(set(labels[mask])) < 2:
                continue
            sc = StandardScaler().fit(X[mask])
            clf = make().fit(sc.transform(X[mask]), labels[mask])
            if clf.predict(sc.transform(X[i:i + 1]))[0] == labels[i]:
                c += 1
        return c / len(X)

    # Two classifiers, and the difference between them is the point. LDA
    # estimates a covariance structure over ten features for thirteen classes
    # from forty-seven training solos, which it cannot do; its held-out score
    # falls below chance. Nearest centroid fits one mean per artist and is the
    # appropriate model at this sample size.
    lda = run(LinearDiscriminantAnalysis, y)
    nc = run(NearestCentroid, y)

    # 1,000 permutations rather than 30: at 30 the smallest attainable p-value is
    # 1/31, which is the same order as the result being tested.
    rng = random.Random(SEED)
    chance = []
    for _ in range(N_CHANCE_PERM):
        yy = list(y); rng.shuffle(yy)
        chance.append(run(NearestCentroid, np.array(yy)))
    beat = sum(1 for c in chance if c >= nc)
    majority = max(collections.Counter(y).values()) / len(y)
    return {'n_solos': int(len(X)), 'n_artists': int(len(set(y))),
            'nearest_centroid_accuracy': nc, 'lda_accuracy': lda,
            'uniform_chance': 1 / len(set(y)), 'majority_class': majority,
            'permuted_mean': st.mean(chance), 'permuted_max': max(chance),
            'p': (beat + 1) / (len(chance) + 1)}


# ------------------------------------------------------------------- 2. nulls
def nulls():
    phrases = load_phrases()
    rng = random.Random(SEED)
    observed = collections.defaultdict(list)
    for p in phrases:
        m = metrics(interval_vector(p['pcs']))
        if m:
            for k, v in m.items():
                observed[k].append(v)

    def draw(p, mode):
        """Keep the phrase's pitch-class count, resample which ones."""
        k = len(set(p['pcs']))
        if mode == 'uniform':
            pool = list(range(12))
        else:
            pool = sorted({pc for s in p['segments']
                           for pc in (chord_scale(s.get('chord')) or [])})
            if len(pool) < k:
                pool = sorted(set(pool) | set(range(12)))
        return rng.sample(pool, min(k, len(pool)))

    out = {'observed': {k: st.mean(v) for k, v in observed.items()},
           'n_phrases': len(phrases)}
    for mode in ('uniform', 'chord-scale'):
        runs = collections.defaultdict(list)
        for _ in range(N_NULL):
            acc = collections.defaultdict(list)
            for p in phrases:
                m = metrics(interval_vector(draw(p, mode)))
                if m:
                    for k, v in m.items():
                        acc[k].append(v)
            for k, v in acc.items():
                runs[k].append(st.mean(v))
        res = {}
        for k, sims in runs.items():
            obs = out['observed'][k]
            beat = sum(1 for s in sims if s <= obs)
            res[k] = {'null_mean': st.mean(sims), 'null_sd': st.pstdev(sims),
                      'observed': obs,
                      'z': (obs - st.mean(sims)) / (st.pstdev(sims) or 1),
                      'p_two_sided': 2 * min(beat + 1, N_NULL - beat + 1) / (N_NULL + 1)}
        out[mode] = res
    return out


def main():
    print('1. LEAVE-ONE-SOLO-OUT ARTIST IDENTIFICATION')
    lo = leave_one_solo_out()
    print('   %d solos, %d artists' % (lo['n_solos'], lo['n_artists']))
    print('   nearest centroid, held out : %.1f%%' % (100 * lo['nearest_centroid_accuracy']))
    print('   LDA, held out              : %.1f%%  (over-parameterised)' % (100 * lo['lda_accuracy']))
    print('   uniform chance (1/%d)  : %.1f%%' % (lo['n_artists'], 100 * lo['uniform_chance']))
    print('   largest single artist  : %.1f%%' % (100 * lo['majority_class']))
    print('   permuted labels, mean  : %.1f%%  (max %.1f%%)'
          % (100 * lo['permuted_mean'], 100 * lo['permuted_max']))
    print('   p vs permuted          : %.4f' % lo['p'])

    print('\n2. NULL BASELINES (pitch-class count preserved, identity resampled)')
    nl = nulls()
    for mode in ('uniform', 'chord-scale'):
        print('   --- %s null ---' % mode)
        for k, v in nl[mode].items():
            print('     %-17s observed %7.3f   null %7.3f ± %-6.3f  z = %+6.2f  p = %.4f'
                  % (k, v['observed'], v['null_mean'], v['null_sd'], v['z'], v['p_two_sided']))

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS / 'validation.json', 'w') as fh:
        json.dump({'leave_one_solo_out': lo, 'nulls': nl}, fh, indent=2)
    print('\nSaved: %s' % (ANALYSIS / 'validation.json'))


if __name__ == '__main__':
    main()
