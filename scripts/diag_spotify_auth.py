import os
from dotenv import load_dotenv
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

# .env einlesen
load_dotenv()

sp = Spotify(auth_manager=SpotifyOAuth(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
    scope="user-read-email user-library-read playlist-read-private playlist-read-collaborative",
    open_browser=True,
    show_dialog=True,
))

me = sp.me()
print("✅ Eingeloggt als:", me.get("id"), "| E-Mail:", me.get("email"))

# Scope prüfen (aus Token)
tok = sp.auth_manager.get_cached_token()
print("🔎 Scopes:", tok.get("scope"))

# Audio-Features gegen eine öffentliche Track-ID testen (The Weeknd - Blinding Lights)
tid = "0VjIjW4GlUZAMYd2vXMi3b"
feats = sp.audio_features([tid])[0]
print("🎛  Audio-Features geladen?:", feats is not None)
if feats:
    print({k: feats[k] for k in ["danceability","energy","valence","tempo"]})
