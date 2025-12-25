#!/usr/bin/env python3
"""
01_extract_blues_corpus.py

Extracts blues form (A12) solos from the Weimar Jazz Database
for artists with 3+ blues entries.

Output:
- data/corpus_metadata.csv: Solo metadata (performer, title, tempo, key, etc.)
- data/melody/: Note-level data per solo (pitch, onset, duration, bar, beat)
- data/beats/: Beat-level data per solo (chord, form, chorus)

Research context:
Cross-artist analysis of improvisational decision-making patterns on blues form.
"""

import os
import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd


# Configuration
WJD_URL = "https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db"
DB_PATH = Path("data/wjazzd.db")
OUTPUT_DIR = Path("data")
MELODY_DIR = OUTPUT_DIR / "melody"
BEATS_DIR = OUTPUT_DIR / "beats"
MIN_SOLOS = 3  # Minimum blues solos per artist


def download_database():
    """Download WJD if not present."""
    if DB_PATH.exists():
        print(f"Database already exists: {DB_PATH}")
        return
    
    print(f"Downloading Weimar Jazz Database from {WJD_URL}...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(WJD_URL, DB_PATH)
    print(f"Downloaded to {DB_PATH}")


def get_blues_artists(conn, min_solos=MIN_SOLOS):
    """Get artists with at least min_solos blues entries."""
    query = """
    SELECT s.performer, COUNT(*) as blues_count
    FROM solo_info s
    JOIN composition_info c ON s.compid = c.compid
    WHERE c.form = 'A12'
    GROUP BY s.performer
    HAVING COUNT(*) >= ?
    ORDER BY COUNT(*) DESC
    """
    df = pd.read_sql_query(query, conn, params=(min_solos,))
    return df


def get_blues_metadata(conn, artists):
    """Get metadata for all blues solos by specified artists."""
    placeholders = ",".join(["?" for _ in artists])
    query = f"""
    SELECT 
        s.melid,
        s.performer,
        s.title,
        c.composer,
        s.instrument,
        s.style,
        s.key,
        s.avgtempo,
        s.tempoclass,
        s.rhythmfeel,
        s.chorus_count,
        s.signature,
        s.chord_changes,
        c.form,
        c.tonalitytype
    FROM solo_info s
    JOIN composition_info c ON s.compid = c.compid
    WHERE c.form = 'A12'
    AND s.performer IN ({placeholders})
    ORDER BY s.performer, s.title
    """
    df = pd.read_sql_query(query, conn, params=artists)
    return df


def extract_melody(conn, melid):
    """Extract note-level data for a solo."""
    query = """
    SELECT 
        eventid,
        onset,
        pitch,
        duration,
        bar,
        beat,
        tatum,
        beatdur
    FROM melody
    WHERE melid = ?
    ORDER BY onset
    """
    return pd.read_sql_query(query, conn, params=(melid,))


def extract_beats(conn, melid):
    """Extract beat-level data for a solo."""
    query = """
    SELECT 
        beatid,
        onset,
        bar,
        beat,
        signature,
        chord,
        form,
        bass_pitch,
        chorus_id
    FROM beats
    WHERE melid = ?
    ORDER BY onset
    """
    return pd.read_sql_query(query, conn, params=(melid,))


def main():
    # Setup directories
    for d in [OUTPUT_DIR, MELODY_DIR, BEATS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Download database
    download_database()
    
    # Connect
    conn = sqlite3.connect(DB_PATH)
    
    # Get qualifying artists
    print(f"\nFinding artists with {MIN_SOLOS}+ blues solos...")
    artists_df = get_blues_artists(conn)
    artists = artists_df["performer"].tolist()
    print(f"Found {len(artists)} artists:")
    for _, row in artists_df.iterrows():
        print(f"  {row['performer']}: {row['blues_count']} solos")
    
    # Get metadata
    print("\nExtracting metadata...")
    metadata = get_blues_metadata(conn, artists)
    metadata.to_csv(OUTPUT_DIR / "corpus_metadata.csv", index=False)
    print(f"Saved metadata for {len(metadata)} solos")
    
    # Extract melody and beat data per solo
    print("\nExtracting melody and beat data...")
    for _, row in metadata.iterrows():
        melid = row["melid"]
        performer = row["performer"].replace(" ", "_").replace(".", "")
        title = row["title"].replace(" ", "_").replace("'", "").replace("/", "-")
        
        # Filename: melid_performer_title.csv
        basename = f"{melid}_{performer}_{title}"
        
        # Melody
        melody_df = extract_melody(conn, melid)
        melody_df.to_csv(MELODY_DIR / f"{basename}.csv", index=False)
        
        # Beats
        beats_df = extract_beats(conn, melid)
        beats_df.to_csv(BEATS_DIR / f"{basename}.csv", index=False)
        
        print(f"  {melid}: {row['performer']} - {row['title']} ({len(melody_df)} notes, {len(beats_df)} beats)")
    
    conn.close()
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print(f"Artists: {len(artists)}")
    print(f"Solos: {len(metadata)}")
    print(f"Metadata: {OUTPUT_DIR / 'corpus_metadata.csv'}")
    print(f"Melody data: {MELODY_DIR}/")
    print(f"Beat data: {BEATS_DIR}/")


if __name__ == "__main__":
    main()
