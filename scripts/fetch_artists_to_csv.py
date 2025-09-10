# scripts/fetch_artists_to_csv.py
import argparse, os
from typing import List
import pandas as pd
from src.spotify_api import (
    extract_id, resolve_artist_id,
    fetch_artist_top_tracks, fetch_artist_album_tracks,
    fetch_tracks_metadata_by_ids, save_df_csv
)

def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artists", nargs="*", help="Artist-IDs/URLs/Namen oder Datei", default=None)
    ap.add_argument("--from-file", help="Pfad zu einer Datei mit Artists (eine pro Zeile)", default=None)
    ap.add_argument("--out", required=True, help="Ziel-CSV")
    ap.add_argument("--market", default="DE")
    ap.add_argument("--only-top", action="store_true", help="Nur Top-Tracks (keine Albentracks)")
    ap.add_argument("--max-albums", type=int, default=10, help="Max. Alben pro Artist, wenn nicht only-top")
    ap.add_argument("--limit", type=int, default=100, help="Maximale Anzahl Songs pro Artist (Default=100)")
    ap.add_argument("--depth", type=int, default=1, help="Tiefe der Artist-Suche (Default=1, d.h. nur direkte Treffer)")
    args = ap.parse_args()

    inputs: List[str] = []
    if args.from_file:
        inputs.extend(read_lines(args.from_file))
    if args.artists:
        inputs.extend(args.artists)
    if not inputs:
        ap.error("Bitte --artists und/oder --from-file angeben.")

    artist_ids = []
    for s in inputs:
        # erst ID/URL rausziehen; wenn es ein Name ist, via search auflösen
        cid = extract_id(s)
        aid = resolve_artist_id(cid)
        if aid: artist_ids.append(aid)

    all_tids: List[str] = []
    for aid in dict.fromkeys(artist_ids):
        tops = fetch_artist_top_tracks(aid, market=args.market)
        tids = tops[:]
        if not args.only_top:
            tids.extend(fetch_artist_album_tracks(aid, market=args.market, max_albums=args.max_albums))
        # dedupe pro artist
        tids = list(dict.fromkeys(tids))
        print(f"[ARTIST] {aid} – Tracks: {len(tids)} (Top={len(tops)})")
        all_tids.extend(tids)

    all_tids = list(dict.fromkeys(all_tids))
    print(f"[INFO] Insgesamt {len(all_tids)} eindeutige Track-IDs. Hole Metadaten+Genres …")

    df = fetch_tracks_metadata_by_ids(all_tids)
    if "release_date" in df.columns:
        df["release_year"] = df["release_date"].str.slice(0, 4)
    save_df_csv(df, args.out)
    print(f"[OK] {len(df)} Zeilen → {args.out}")

if __name__ == "__main__":
    main()








