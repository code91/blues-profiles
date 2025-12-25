#!/usr/bin/env python3
"""
02_segment_phrases_compute_ivs.py

Segments solos into phrases using WJD annotations and computes:
- Phrase-level: pitch-class set, interval vector, Forte number
- Sub-phrase: IV per chord segment within each phrase

Output:
- data/phrases/{melid}_{performer}_{title}.json

Research context:
Building nested IV structures to analyze anticipation within phrases.
"""

import json
import os
import sqlite3
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd


# Configuration
DB_PATH = Path("data/wjazzd.db")
METADATA_PATH = Path("data/corpus_metadata.csv")
MELODY_DIR = Path("data/melody")
PHRASES_DIR = Path("data/phrases")

# Forte number lookup (prime forms for set classes 1-6)
# We'll compute prime form and look up
FORTE_CATALOG = {
    # Cardinality 1
    (0,): "1-1",
    # Cardinality 2
    (0, 1): "2-1", (0, 2): "2-2", (0, 3): "2-3", (0, 4): "2-4", 
    (0, 5): "2-5", (0, 6): "2-6",
    # Cardinality 3
    (0, 1, 2): "3-1", (0, 1, 3): "3-2", (0, 1, 4): "3-3", (0, 1, 5): "3-4",
    (0, 1, 6): "3-5", (0, 2, 4): "3-6", (0, 2, 5): "3-7", (0, 2, 6): "3-8",
    (0, 2, 7): "3-9", (0, 3, 6): "3-10", (0, 3, 7): "3-11", (0, 4, 8): "3-12",
    # Cardinality 4
    (0, 1, 2, 3): "4-1", (0, 1, 2, 4): "4-2", (0, 1, 3, 4): "4-3",
    (0, 1, 2, 5): "4-4", (0, 1, 2, 6): "4-5", (0, 1, 2, 7): "4-6",
    (0, 1, 4, 5): "4-7", (0, 1, 5, 6): "4-8", (0, 1, 6, 7): "4-9",
    (0, 2, 3, 5): "4-10", (0, 1, 3, 5): "4-11", (0, 2, 3, 6): "4-12",
    (0, 1, 3, 6): "4-13", (0, 2, 3, 7): "4-14", (0, 1, 4, 6): "4-Z15",
    (0, 1, 5, 7): "4-16", (0, 3, 4, 7): "4-17", (0, 1, 4, 7): "4-18",
    (0, 1, 4, 8): "4-19", (0, 1, 5, 8): "4-20", (0, 2, 4, 6): "4-21",
    (0, 2, 4, 7): "4-22", (0, 2, 5, 7): "4-23", (0, 2, 4, 8): "4-24",
    (0, 2, 6, 8): "4-25", (0, 3, 5, 8): "4-26", (0, 2, 5, 8): "4-27",
    (0, 3, 6, 9): "4-28", (0, 1, 3, 7): "4-Z29",
    # Cardinality 5
    (0, 1, 2, 3, 4): "5-1", (0, 1, 2, 3, 5): "5-2", (0, 1, 2, 4, 5): "5-3",
    (0, 1, 2, 3, 6): "5-4", (0, 1, 2, 3, 7): "5-5", (0, 1, 2, 5, 6): "5-6",
    (0, 1, 2, 6, 7): "5-7", (0, 2, 3, 4, 6): "5-8", (0, 1, 2, 4, 6): "5-9",
    (0, 1, 3, 4, 6): "5-10", (0, 2, 3, 4, 7): "5-11", (0, 1, 3, 5, 6): "5-Z12",
    (0, 1, 2, 4, 8): "5-13", (0, 1, 2, 5, 7): "5-14", (0, 1, 2, 6, 8): "5-15",
    (0, 1, 3, 4, 7): "5-16", (0, 1, 3, 4, 8): "5-Z17", (0, 1, 4, 5, 7): "5-Z18",
    (0, 1, 3, 6, 7): "5-19", (0, 1, 5, 6, 8): "5-20", (0, 1, 4, 5, 8): "5-21",
    (0, 1, 4, 7, 8): "5-22", (0, 2, 3, 5, 7): "5-23", (0, 1, 3, 5, 7): "5-24",
    (0, 2, 3, 5, 8): "5-25", (0, 2, 4, 5, 8): "5-26", (0, 1, 3, 5, 8): "5-27",
    (0, 2, 3, 6, 8): "5-28", (0, 1, 3, 6, 8): "5-29", (0, 1, 4, 6, 8): "5-30",
    (0, 1, 3, 6, 9): "5-31", (0, 1, 4, 6, 9): "5-32", (0, 2, 4, 6, 8): "5-33",
    (0, 2, 4, 6, 9): "5-34", (0, 2, 4, 7, 9): "5-35", (0, 1, 2, 4, 7): "5-Z36",
    (0, 3, 4, 5, 8): "5-Z37", (0, 1, 2, 5, 8): "5-Z38",
    # Cardinality 6
    (0, 1, 2, 3, 4, 5): "6-1", (0, 1, 2, 3, 4, 6): "6-2", (0, 1, 2, 3, 5, 6): "6-Z3",
    (0, 1, 2, 4, 5, 6): "6-Z4", (0, 1, 2, 3, 6, 7): "6-5", (0, 1, 2, 5, 6, 7): "6-Z6",
    (0, 1, 2, 6, 7, 8): "6-7", (0, 2, 3, 4, 5, 7): "6-8", (0, 1, 2, 3, 5, 7): "6-9",
    (0, 1, 3, 4, 5, 7): "6-Z10", (0, 1, 2, 4, 5, 7): "6-Z11", (0, 1, 2, 4, 6, 7): "6-Z12",
    (0, 1, 3, 4, 6, 7): "6-Z13", (0, 1, 3, 4, 5, 8): "6-14", (0, 1, 2, 4, 5, 8): "6-15",
    (0, 1, 4, 5, 6, 8): "6-16", (0, 1, 2, 4, 7, 8): "6-Z17", (0, 1, 2, 5, 7, 8): "6-18",
    (0, 1, 3, 4, 7, 8): "6-Z19", (0, 1, 4, 5, 8, 9): "6-20", (0, 2, 3, 4, 6, 8): "6-21",
    (0, 1, 2, 4, 6, 8): "6-22", (0, 2, 3, 5, 6, 8): "6-Z23", (0, 1, 3, 4, 6, 8): "6-Z24",
    (0, 1, 3, 5, 6, 8): "6-Z25", (0, 1, 3, 5, 7, 8): "6-Z26", (0, 1, 3, 4, 6, 9): "6-27",
    (0, 1, 3, 5, 6, 9): "6-Z28", (0, 1, 3, 6, 8, 9): "6-Z29", (0, 1, 3, 6, 7, 9): "6-30",
    (0, 1, 4, 5, 7, 9): "6-31", (0, 2, 4, 5, 7, 9): "6-32", (0, 2, 3, 5, 7, 9): "6-33",
    (0, 1, 3, 5, 7, 9): "6-34", (0, 2, 4, 6, 8, 10): "6-35",
    (0, 1, 2, 3, 4, 7): "6-Z36", (0, 1, 2, 3, 4, 8): "6-Z37", (0, 1, 2, 3, 7, 8): "6-Z38",
    (0, 2, 3, 4, 5, 8): "6-Z39", (0, 1, 2, 3, 5, 8): "6-Z40", (0, 1, 2, 3, 6, 8): "6-Z41",
    (0, 1, 2, 3, 6, 9): "6-Z42", (0, 1, 2, 5, 6, 8): "6-Z43", (0, 1, 2, 5, 6, 9): "6-Z44",
    (0, 2, 3, 4, 6, 9): "6-Z45", (0, 1, 2, 4, 6, 9): "6-Z46", (0, 1, 2, 4, 7, 9): "6-Z47",
    (0, 1, 2, 5, 7, 9): "6-Z48", (0, 1, 3, 4, 7, 9): "6-Z49", (0, 1, 4, 6, 7, 9): "6-Z50",
}


def pitches_to_pcs(pitches):
    """Convert MIDI pitches to pitch-class set (sorted, unique, mod 12)."""
    pcs = sorted(set(int(p) % 12 for p in pitches))
    return pcs


def pcs_to_normal_form(pcs):
    """Compute normal form of a pitch-class set."""
    if len(pcs) <= 1:
        return tuple(pcs)
    
    n = len(pcs)
    rotations = []
    for i in range(n):
        rotation = [(pcs[(i + j) % n] - pcs[i]) % 12 for j in range(n)]
        rotations.append((rotation, pcs[i]))
    
    # Sort by most compact (smallest intervals from left)
    rotations.sort(key=lambda x: x[0])
    best_rotation, start_pc = rotations[0]
    
    return tuple(best_rotation)


def pcs_to_prime_form(pcs):
    """Compute prime form of a pitch-class set."""
    if len(pcs) <= 1:
        return tuple([0] if pcs else [])
    
    # Normal form
    normal = pcs_to_normal_form(pcs)
    
    # Inversion
    inverted = sorted((12 - pc) % 12 for pc in pcs)
    inv_normal = pcs_to_normal_form(inverted)
    
    # Compare and return more compact
    if inv_normal < normal:
        return inv_normal
    return normal


def pcs_to_interval_vector(pcs):
    """Compute interval vector from pitch-class set."""
    iv = [0, 0, 0, 0, 0, 0]  # intervals 1-6
    
    for p1, p2 in combinations(pcs, 2):
        interval = abs(p1 - p2)
        if interval > 6:
            interval = 12 - interval
        if interval > 0:
            iv[interval - 1] += 1
    
    return iv


def get_forte_number(pcs):
    """Look up Forte number from pitch-class set."""
    if len(pcs) == 0:
        return None
    if len(pcs) > 6:
        # For larger sets, just return cardinality
        return f"{len(pcs)}-x"
    
    prime = pcs_to_prime_form(pcs)
    return FORTE_CATALOG.get(prime, f"{len(pcs)}-?")


def get_sections(conn, melid, section_type):
    """Get sections of a specific type for a solo."""
    query = """
    SELECT start, end, value
    FROM sections
    WHERE melid = ? AND type = ?
    ORDER BY start
    """
    return pd.read_sql_query(query, conn, params=(melid, section_type))


def get_melody_with_position(melody_dir, melid, performer, title):
    """Load melody data with sequential position index (0-based)."""
    # Find the file
    performer_clean = performer.replace(" ", "_").replace(".", "")
    title_clean = title.replace(" ", "_").replace("'", "").replace("/", "-")
    filename = f"{melid}_{performer_clean}_{title_clean}.csv"
    filepath = melody_dir / filename
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath)
    # Sections table uses 0-based sequential position, not global eventid
    df = df.reset_index(drop=True)  # 0, 1, 2, ... as index
    return df


def process_solo(conn, melid, performer, title, melody_dir):
    """Process a single solo: extract phrases with nested chord IVs."""
    
    # Get melody data
    melody = get_melody_with_position(melody_dir, melid, performer, title)
    if melody is None:
        print(f"  WARNING: Melody file not found for {melid}")
        return None
    
    # Get phrase and chord annotations
    phrases_df = get_sections(conn, melid, 'PHRASE')
    chords_df = get_sections(conn, melid, 'CHORD')
    chorus_df = get_sections(conn, melid, 'CHORUS')
    
    if len(phrases_df) == 0:
        print(f"  WARNING: No phrase annotations for {melid}")
        return None
    
    # Build chorus lookup (event_id -> chorus_id)
    def get_chorus_for_event(event_id):
        for _, row in chorus_df.iterrows():
            if row['start'] <= event_id <= row['end']:
                try:
                    return int(row['value'])
                except:
                    return -1
        return -1
    
    # Process each phrase
    phrases = []
    for _, phrase_row in phrases_df.iterrows():
        start_event = phrase_row['start']
        end_event = phrase_row['end']
        phrase_num = phrase_row['value']
        
        # Get notes in this phrase (using positional index)
        phrase_notes = melody[(melody.index >= start_event) & (melody.index <= end_event)]
        
        if len(phrase_notes) == 0:
            continue
        
        # Phrase-level metrics
        pitches = phrase_notes['pitch'].tolist()
        pcs = pitches_to_pcs(pitches)
        iv = pcs_to_interval_vector(pcs)
        forte = get_forte_number(pcs)
        
        # Get bar/beat boundaries
        start_bar = int(phrase_notes.iloc[0]['bar'])
        start_beat = int(phrase_notes.iloc[0]['beat'])
        end_bar = int(phrase_notes.iloc[-1]['bar'])
        end_beat = int(phrase_notes.iloc[-1]['beat'])
        
        # Chorus
        chorus_id = get_chorus_for_event(start_event)
        
        # Find chords that overlap this phrase
        chord_segments = []
        for _, chord_row in chords_df.iterrows():
            chord_start = chord_row['start']
            chord_end = chord_row['end']
            chord_name = chord_row['value']
            
            # Check overlap
            if chord_end < start_event or chord_start > end_event:
                continue
            
            # Get notes in this chord segment within the phrase
            seg_start = max(chord_start, start_event)
            seg_end = min(chord_end, end_event)
            
            seg_notes = melody[(melody.index >= seg_start) & (melody.index <= seg_end)]
            
            if len(seg_notes) == 0:
                continue
            
            seg_pitches = seg_notes['pitch'].tolist()
            seg_pcs = pitches_to_pcs(seg_pitches)
            seg_iv = pcs_to_interval_vector(seg_pcs)
            seg_forte = get_forte_number(seg_pcs)
            
            seg_start_bar = int(seg_notes.iloc[0]['bar'])
            seg_end_bar = int(seg_notes.iloc[-1]['bar'])
            
            chord_segments.append({
                'chord': chord_name,
                'start_bar': seg_start_bar,
                'end_bar': seg_end_bar,
                'start_event': int(seg_start),
                'end_event': int(seg_end),
                'n_notes': len(seg_notes),
                'pitches': [int(p) for p in seg_pitches],
                'pcs': seg_pcs,
                'iv': seg_iv,
                'forte': seg_forte
            })
        
        phrases.append({
            'phrase_id': int(phrase_num) if phrase_num.isdigit() else phrase_num,
            'chorus_id': chorus_id,
            'start_bar': start_bar,
            'start_beat': start_beat,
            'end_bar': end_bar,
            'end_beat': end_beat,
            'start_event': int(start_event),
            'end_event': int(end_event),
            'n_notes': len(phrase_notes),
            'pitches': [int(p) for p in pitches],
            'pcs': pcs,
            'iv': iv,
            'forte': forte,
            'chord_segments': chord_segments
        })
    
    return {
        'melid': melid,
        'performer': performer,
        'title': title,
        'n_phrases': len(phrases),
        'phrases': phrases
    }


def main():
    # Setup
    PHRASES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    metadata = pd.read_csv(METADATA_PATH)
    print(f"Processing {len(metadata)} solos...\n")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Process each solo
    summary = []
    for _, row in metadata.iterrows():
        melid = row['melid']
        performer = row['performer']
        title = row['title']
        
        result = process_solo(conn, melid, performer, title, MELODY_DIR)
        
        if result is None:
            continue
        
        # Save JSON
        performer_clean = performer.replace(" ", "_").replace(".", "")
        title_clean = title.replace(" ", "_").replace("'", "").replace("/", "-")
        filename = f"{melid}_{performer_clean}_{title_clean}.json"
        
        with open(PHRASES_DIR / filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Summary stats
        n_phrases = result['n_phrases']
        total_segments = sum(len(p['chord_segments']) for p in result['phrases'])
        unique_ivs = len(set(tuple(p['iv']) for p in result['phrases']))
        
        print(f"  {melid}: {performer} - {title}")
        print(f"       {n_phrases} phrases, {total_segments} chord segments, {unique_ivs} unique IVs")
        
        summary.append({
            'melid': melid,
            'performer': performer,
            'title': title,
            'n_phrases': n_phrases,
            'n_chord_segments': total_segments,
            'unique_phrase_ivs': unique_ivs
        })
    
    conn.close()
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(PHRASES_DIR / 'summary.csv', index=False)
    
    # Print summary by artist
    print("\n" + "="*60)
    print("SUMMARY BY ARTIST")
    print("="*60)
    artist_summary = summary_df.groupby('performer').agg({
        'n_phrases': 'sum',
        'n_chord_segments': 'sum',
        'unique_phrase_ivs': 'sum'
    }).sort_values('n_phrases', ascending=False)
    
    for performer, row in artist_summary.iterrows():
        print(f"{performer}: {int(row['n_phrases'])} phrases, {int(row['unique_phrase_ivs'])} unique IVs")
    
    print(f"\nTotal: {summary_df['n_phrases'].sum()} phrases across {len(metadata)} solos")
    print(f"Output: {PHRASES_DIR}/")


if __name__ == "__main__":
    main()
