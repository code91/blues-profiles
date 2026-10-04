#!/usr/bin/env python3
"""
Does the player explain more than the era?

The study's hypothesis is comparative: improvisational strategy is more strongly
associated with individual identity than with historical era. Scripts 03-09 test
the two halves separately and never adjudicate between them, so this script does.

Four analyses, each a single test rather than a family:

  1. Identity    a player's solos resemble each other more than they resemble
                 other players' solos.
  2. Era         the same question with the style label in place of the player.
                 These are confounded, since each player carries one label.
  3. Control     the identity question again with era held constant, comparing
                 only solos that share a style label and shuffling player labels
                 only within style. This is the hypothesis as actually stated.
  4. Shared tune eight tunes in the corpus are played by two players each. Same
                 changes, different player: the cleanest control available for
                 the objection that the harmony drives the metrics.

Plus a secular-trend test on recording year, which is a sharper era measure than
five coarse style labels.

    python3 10_identity_vs_era.py
"""
import collections
import csv
import itertools
import json
import math
import random
import sqlite3
import statistics as st
from pathlib import Path

DATA = Path('data')
ANALYSIS = DATA / 'analysis'
DB_PATH = DATA / 'wjazzd.db'
SEED = 20261004
N_PERM = 10000

# Mean and standard deviation of each, per solo. The sd matters as much as the
# mean: a player is characterised by how much he varies as well as by his level.
FEATURES = ['density', 'entropy', 'dissonance_ratio', 'n_notes', 'cardinality']
MIN_PHRASES = 5


def load_profiles():
    """One z-scored feature vector per solo, with its metadata."""
    meta = {r['melid']: r for r in csv.DictReader(open(DATA / 'corpus_metadata.csv'))}
    rows = [r for r in csv.DictReader(open(DATA / 'timeseries' / 'timeseries_data.csv'))
            if r.get('density')]
    by_solo = collections.defaultdict(list)
    for r in rows:
        by_solo[r['melid']].append(r)

    raw = {}
    for mel, rs in by_solo.items():
        if len(rs) < MIN_PHRASES or mel not in meta:
            continue
        v = []
        for f in FEATURES:
            xs = [float(x[f]) for x in rs]
            v += [st.mean(xs), st.pstdev(xs)]
        raw[mel] = v

    dims = len(next(iter(raw.values())))
    mu = [st.mean([v[d] for v in raw.values()]) for d in range(dims)]
    sd = [st.pstdev([v[d] for v in raw.values()]) or 1.0 for d in range(dims)]
    prof = {k: [(v[d] - mu[d]) / sd[d] for d in range(dims)] for k, v in raw.items()}
    return prof, meta


def years():
    """Recording year per melid, from the WJazzD track table."""
    if not DB_PATH.exists():
        return {}
    con = sqlite3.connect(DB_PATH)
    out = {}
    q = ('SELECT s.melid, t.recordingdate FROM solo_info s '
         'JOIN track_info t ON t.trackid = s.trackid')
    try:
        for mel, date in con.execute(q):
            if not date:
                continue
            digits = ''.join(ch for ch in str(date) if ch.isdigit())
            for i in range(len(digits) - 3):
                y = int(digits[i:i + 4])
                if 1900 <= y <= 2020:
                    out[str(mel)] = y
                    break
    except sqlite3.Error:
        pass
    con.close()
    return out


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def contrast(prof, ids, labels, pairs=None):
    """Mean within-group and between-group distance over the given pairs."""
    pairs = pairs if pairs is not None else list(itertools.combinations(ids, 2))
    w, b = [], []
    for i, j in pairs:
        (w if labels[i] == labels[j] else b).append(dist(prof[i], prof[j]))
    if not w or not b:
        return None
    return sum(w) / len(w), sum(b) / len(b), len(w), len(b)


def permute(prof, ids, labels, pairs=None, strata=None, n=N_PERM, seed=SEED):
    """Shuffle labels and count how often the observed separation is matched.

    With `strata`, labels are shuffled only inside each stratum, which is what
    holds era constant while still destroying the player assignment.
    """
    base = contrast(prof, ids, labels, pairs)
    if base is None:
        return None
    obs = base[1] - base[0]
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for i in ids:
        groups[strata[i] if strata else '_'].append(i)
    hits = 0
    for _ in range(n):
        shuffled = {}
        for members in groups.values():
            vals = [labels[m] for m in members]
            rng.shuffle(vals)
            shuffled.update(dict(zip(members, vals)))
        got = contrast(prof, ids, shuffled, pairs)
        if got and (got[1] - got[0]) >= obs:
            hits += 1
    return {'within': base[0], 'between': base[1], 'ratio': base[0] / base[1],
            'n_within_pairs': base[2], 'n_between_pairs': base[3],
            'p': (hits + 1) / (n + 1), 'n_permutations': n}


def report(name, res, extra=''):
    if not res:
        print('  %-34s (not enough data)' % name)
        return
    print('  %-34s within %.3f  between %.3f  ratio %.3f  p = %.4f %s'
          % (name, res['within'], res['between'], res['ratio'], res['p'], extra))


def main():
    prof, meta = load_profiles()
    ids = sorted(prof)
    yr = years()
    print('Solos profiled: %d   features: %d   permutations: %d\n'
          % (len(ids), len(next(iter(prof.values()))), N_PERM))

    out = {'n_solos': len(ids), 'features': FEATURES, 'n_permutations': N_PERM}

    # 1 + 2: the two explanations, each on its own
    artist = {i: meta[i]['performer'] for i in ids}
    style = {i: meta[i]['style'] for i in ids}
    out['artist'] = permute(prof, ids, artist)
    out['era'] = permute(prof, ids, style)
    print('UNCONTROLLED')
    report('identity (artist)', out['artist'], '(%d artists)' % len(set(artist.values())))
    report('era (style label)', out['era'], '(%d styles)' % len(set(style.values())))

    # 3: the hypothesis as stated, with era held constant
    same_style = [(i, j) for i, j in itertools.combinations(ids, 2)
                  if style[i] == style[j]]
    out['artist_within_era'] = permute(prof, ids, artist, pairs=same_style, strata=style)
    print('\nCONTROLLED')
    report('identity, era held constant', out['artist_within_era'])

    # 4: same tune, different player
    title = {i: meta[i]['title'].strip().lower() for i in ids}
    counts = collections.defaultdict(set)
    for i in ids:
        counts[title[i]].add(artist[i])
    shared = {t for t, who in counts.items() if len(who) > 1}
    tune_pairs = [(i, j) for i, j in itertools.combinations(ids, 2)
                  if title[i] == title[j] and title[i] in shared]
    out['shared_tunes'] = sorted(shared)
    out['n_solos_on_shared_tunes'] = sum(1 for i in ids if title[i] in shared)
    print('\nSHARED-TUNE CONTROL')
    print('  %d tunes played by more than one artist, covering %d solos'
          % (len(shared), out['n_solos_on_shared_tunes']))
    if tune_pairs:
        same_artist = [dist(prof[i], prof[j]) for i, j in tune_pairs if artist[i] == artist[j]]
        diff_artist = [dist(prof[i], prof[j]) for i, j in tune_pairs if artist[i] != artist[j]]
        # how far apart are two players on the SAME tune, against the corpus baseline?
        all_diff = [dist(prof[i], prof[j]) for i, j in itertools.combinations(ids, 2)
                    if artist[i] != artist[j]]
        out['same_tune_diff_artist'] = {
            'mean': sum(diff_artist) / len(diff_artist), 'n': len(diff_artist)}
        out['any_tune_diff_artist'] = {'mean': sum(all_diff) / len(all_diff), 'n': len(all_diff)}
        if same_artist:
            out['same_tune_same_artist'] = {
                'mean': sum(same_artist) / len(same_artist), 'n': len(same_artist)}
        print('  different artists, same tune : %.3f (n=%d)'
              % (out['same_tune_diff_artist']['mean'], len(diff_artist)))
        print('  different artists, any tune  : %.3f (n=%d)'
              % (out['any_tune_diff_artist']['mean'], len(all_diff)))
        if same_artist:
            print('  same artist, same tune       : %.3f (n=%d)'
                  % (out['same_tune_same_artist']['mean'], len(same_artist)))

    # secular trend: a finer era measure than five style labels
    print('\nSECULAR TREND (recording year)')
    have = [i for i in ids if i in yr]
    out['n_with_year'] = len(have)
    if len(have) > 10:
        rows = [r for r in csv.DictReader(open(DATA / 'timeseries' / 'timeseries_data.csv'))
                if r.get('density')]
        bysolo = collections.defaultdict(list)
        for r in rows:
            bysolo[r['melid']].append(r)
        out['year_range'] = [min(yr[i] for i in have), max(yr[i] for i in have)]
        out['trends'] = {}
        print('  %d solos dated, %d to %d' % (len(have), *out['year_range']))
        for f in FEATURES:
            xs = [yr[i] for i in have]
            ys = [st.mean([float(x[f]) for x in bysolo[i]]) for i in have]
            mx, my = st.mean(xs), st.mean(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
            r = num / den if den else float('nan')
            rng = random.Random(SEED)
            hits = 0
            for _ in range(N_PERM):
                rng.shuffle(ys)
                n2 = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
                d2 = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
                if d2 and abs(n2 / d2) >= abs(r):
                    hits += 1
            p = (hits + 1) / (N_PERM + 1)
            out['trends'][f] = {'r': r, 'p': p}
            print('    %-18s r = %+.3f   p = %.4f' % (f, r, p))
    else:
        print('  no recording years available')

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS / 'identity_vs_era.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    print('\nSaved: %s' % (ANALYSIS / 'identity_vs_era.json'))


if __name__ == '__main__':
    main()
