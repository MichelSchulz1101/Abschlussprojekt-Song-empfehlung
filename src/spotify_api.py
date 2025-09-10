# src/spotify_api.py
from __future__ import annotations
import os, time, re
from typing import List, Dict, Any, Optional, Generator, Iterable
import pandas as pd
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from spotipy.exceptions import SpotifyException

load_dotenv()

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID") or ""
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET") or ""
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

def _chunks(seq: List[str], n: int) -> Generator[List[str], None, None]:
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def _mk_public() -> Spotify:
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError("Fehlende SPOTIPY_* ENV Variablen.")
    print("[AUTH] Client-Credentials")
    return Spotify(auth_manager=SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET))

def _mk_user() -> Spotify:
    # Für private Playlists / mehr Stabilität beim Browsen (Genres etc.) – nicht zwingend nötig.
    print("[AUTH] Optionaler User-Login")
    return Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI,
        scope="user-read-email playlist-read-private playlist-read-collaborative",
        open_browser=True, cache_path=".cache"
    ))

_sp: Optional[Spotify] = None
def sp() -> Spotify:
    global _sp
    if _sp is None:
        _sp = _mk_public()
    return _sp

ID_RX = re.compile(r"(artist|playlist|track)/([A-Za-z0-9]{22})")

def extract_id(s: str) -> str:
    s = s.strip()
    m = ID_RX.search(s)
    if m:
        return m.group(2)
    if re.fullmatch(r"[A-Za-z0-9]{22}", s):
        return s
    return s  # ggf. als Name, wird via search aufgelöst

def resolve_artist_id(q: str) -> Optional[str]:
    """Nimmt ID/URL/Name – gibt Artist-ID oder None."""
    q = q.strip()
    if re.fullmatch(r"[A-Za-z0-9]{22}", q):
        return q
    m = ID_RX.search(q)
    if m:
        return m.group(2)
    # als Name suchen
    try:
        res = sp().search(q=q, type="artist", limit=1) or {}
        items = res.get("artists", {}).get("items", []) or []
        return items[0]["id"] if items else None
    except SpotifyException:
        return None

def fetch_artist_top_tracks(artist_id: str, market: str = "DE") -> List[str]:
    try:
        res = sp().artist_top_tracks(artist_id, country=market) or {}
        return [t["id"] for t in res.get("tracks", []) if t and t.get("id")]
    except SpotifyException:
        return []

def fetch_artist_album_tracks(artist_id: str, market: str = "DE", max_albums: int = 10) -> List[str]:
    tids: List[str] = []
    try:
        offs = 0
        albums: List[Dict[str, Any]] = []
        while True:
            res = sp().artist_albums(artist_id, include_groups="album,single,compilation,appears_on",
                                     limit=50, offset=offs, country=market) or {}
            items = res.get("items", [])
            albums.extend(items)
            if len(items) < 50: break
            offs += 50
        # dedupe & sort by release_date desc
        seen = set()
        uniq = []
        for a in albums:
            aid = a.get("id")
            if not aid or aid in seen: continue
            seen.add(aid); uniq.append(a)
        uniq.sort(key=lambda x: x.get("release_date") or "", reverse=True)
        if max_albums: uniq = uniq[:max_albums]
        # tracks je album
        for a in uniq:
            aid = a.get("id")
            o = 0
            while True:
                tr = sp().album_tracks(aid, limit=50, offset=o) or {}
                its = tr.get("items", [])
                tids.extend([t["id"] for t in its if t.get("id")])
                if len(its) < 50: break
                o += 50
            time.sleep(0.05)
    except SpotifyException:
        pass
    # global dedupe
    return list(dict.fromkeys(tids))

def fetch_tracks_metadata_by_ids(track_ids: List[str]) -> pd.DataFrame:
    """Nur Metadaten + Künstler-Genres (kein audio-features)."""
    if not track_ids: return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    artist_ids: List[str] = []
    # 1) Tracks-Metadaten
    for batch in _chunks(track_ids, 50):
        try:
            tr = sp().tracks(batch) or {}
            items = tr.get("tracks", []) or []
            for t in items:
                if not t: continue
                tid = t.get("id")
                if not tid: continue
                artists = t.get("artists") or []
                artist_names = ", ".join([a.get("name") for a in artists if a.get("name")])
                artist_ids.extend([a.get("id") for a in artists if a.get("id")])
                album = t.get("album") or {}
                rows.append({
                    "track_id": tid,
                    "track_name": t.get("name"),
                    "artist_name": artist_names,
                    "album_name": album.get("name"),
                    "release_date": album.get("release_date"),
                    "popularity": t.get("popularity"),
                    "duration_ms": t.get("duration_ms"),
                    "explicit": t.get("explicit"),
                })
        except SpotifyException:
            continue
        time.sleep(0.05)
    base_df = pd.DataFrame(rows).drop_duplicates(subset=["track_id"])
    if base_df.empty:
        return base_df

    # 2) Künstler-Genres nachladen
    artist_ids = list({aid for aid in artist_ids if aid})
    aid2genres: Dict[str, List[str]] = {}
    for batch in _chunks(artist_ids, 50):
        try:
            arts = sp().artists(batch) or {}
            for a in arts.get("artists", []) or []:
                aid = a.get("id")
                if not aid: continue
                aid2genres[aid] = a.get("genres") or []
        except SpotifyException:
            continue
        time.sleep(0.05)

    # 3) Genres je Track (vereinigt über alle beteiligten Künstler)
    def track_genres(tid: str) -> str:
        try:
            # wir müssen die artist ids erneut holen – effizient per cache:
            # (einfacher Weg: aus track->artists erneut abrufen)
            tr = sp().track(tid) or {}
            arts = tr.get("artists", []) or []
            gset = set()
            for a in arts:
                gset.update(aid2genres.get(a.get("id"), []))
            return ", ".join(sorted(gset))
        except SpotifyException:
            return ""
    base_df["artist_genres"] = [track_genres(t) for t in base_df["track_id"]]
    return base_df

def save_df_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)







