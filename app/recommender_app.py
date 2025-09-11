from __future__ import annotations
import os, json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, hstack
import streamlit.components.v1 as components  # <— für Spotify-Embed

st.set_page_config(page_title="Song Recommender (Model Registry)", layout="wide")

MODELS_DIR = os.getenv("MODELS_DIR", "models")
REGISTRY = os.path.join(MODELS_DIR, "registry.json")

# ---------- Caching / Laden ----------
@st.cache_data(show_spinner=True)
def load_registry(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=True)
def load_model(path: str) -> dict:
    return joblib.load(path)

# ---------- Hilfen ----------
def build_query_vector(artifact: dict, row: pd.Series) -> csr_matrix:
    vec_genre = artifact["vec_genre"]
    vec_artist = artifact["vec_artist"]
    vec_title = artifact.get("vec_title", None)
    scaler = artifact["scaler"]
    cfg = artifact.get("config", {})

    gw = float(cfg.get("genre_weight", 1.0))
    aw = float(cfg.get("artist_weight", 1.0))
    tw = float(cfg.get("title_weight", 0.2))
    nw = float(cfg.get("numeric_weight", 0.6))

    Xg = vec_genre.transform([str(row.get("artist_genres", ""))]) if vec_genre else None
    Xa = vec_artist.transform([str(row.get("artist_name", ""))]) if vec_artist else None
    Xt = vec_title.transform([str(row.get("track_name", ""))]) if vec_title else None

    year = float(row.get("release_year", 0))
    explicit = float(row.get("explicit", 0))
    Xn = csr_matrix(scaler.transform(np.array([[year, explicit]]))) if scaler is not None else csr_matrix([[year, explicit]])

    blocks = []
    if Xg is not None: blocks.append(Xg.multiply(gw))
    if Xa is not None: blocks.append(Xa.multiply(aw))
    if Xt is not None: blocks.append(Xt.multiply(tw))
    blocks.append(Xn.multiply(nw))

    q = hstack(blocks).tocsr()
    qn = np.sqrt(q.multiply(q).sum())
    return q if qn == 0 else q.multiply(1.0 / qn)

def contains_mask(series: pd.Series, query: str) -> pd.Series:
    q = query.strip().lower()
    if not q:
        return pd.Series([True] * len(series), index=series.index)
    return series.fillna("").str.lower().str.contains(q, regex=False)

def extract_genre_vocab(df: pd.DataFrame, top_k: int = 200) -> list[str]:
    s = df.get("artist_genres", pd.Series([], dtype=str)).fillna("").astype(str)
    exploded = (
        s.str.lower()
         .str.replace(r"[;\|]", ",", regex=True)
         .str.split(",")
         .explode()
         .str.strip()
    )
    exploded = exploded[exploded.ne("")]
    vc = exploded.value_counts()
    return vc.head(top_k).index.tolist()

def genre_filter_mask(df: pd.DataFrame, selected: list[str]) -> pd.Series:
    if not selected:
        return pd.Series([True] * len(df), index=df.index)
    hay = df.get("artist_genres", "").fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for g in selected:
        mask |= hay.str.contains(g.lower(), regex=False)
    return mask

def search_candidates(df: pd.DataFrame, query: str, limit: int = 100) -> pd.Index:
    if not query.strip():
        return df.index[:limit]
    display = df["display"].fillna("")
    genres = df.get("artist_genres", "").fillna("")
    mask = contains_mask(display, query) | contains_mask(genres, query)
    idx = df.index[mask]
    try:
        from rapidfuzz import process, fuzz
        choices = [(int(i), display.loc[i]) for i in idx]
        scored = process.extract(query, dict(choices), scorer=fuzz.WRatio, limit=min(limit, len(choices)))
        ordered_idx = [int(label) for (label, _score, _) in scored]
        return pd.Index(ordered_idx)
    except Exception:
        return idx[:limit]

def spotify_embed_iframe(track_id: str, height: int = 80) -> None:
    """Zeigt Spotify-Embed-Player (funktioniert ohne preview_url)."""
    src = f"https://open.spotify.com/embed/track/{track_id}"
    components.iframe(src=src, height=height, scrolling=False)

# ---------- UI ----------
st.title("🎧 Content-Based Song Recommender (Modell-Auswahl, Suche & Filter)")

with st.sidebar:
    st.header("📦 Modell")
    reg = load_registry(REGISTRY)
    if not reg:
        st.error("Keine Modelle gefunden. Trainiere zuerst:\n\n`python -m scripts.train_models --csv data/raw/mix_master.csv --out models`")
        st.stop()
    model_names = sorted(reg.keys())
    model_choice = st.selectbox("Modell auswählen", model_names)
    model_path = reg[model_choice]["path"]

    top_k = st.number_input("Top-K Empfehlungen", min_value=3, max_value=50, value=10, step=1)

artifact = load_model(model_path)
df_meta: pd.DataFrame = artifact["df_meta"].copy()
nn: NearestNeighbors = artifact["nn"]
X_norm = artifact["X_norm"]

# Anzeige-Spalte & Jahr/Explicit bereinigen
df_meta["display"] = df_meta["track_name"].fillna("").astype(str) + " — " + df_meta["artist_name"].fillna("").astype(str)
if "release_year" not in df_meta.columns:
    df_meta["release_year"] = 0
df_meta["release_year"] = pd.to_numeric(df_meta["release_year"], errors="coerce").fillna(0).astype(int)
if "explicit" not in df_meta.columns:
    df_meta["explicit"] = 0
df_meta["explicit"] = pd.to_numeric(df_meta["explicit"], errors="coerce").fillna(0).astype(int)

# ---------- Filterleiste ----------
st.subheader("1) Song suchen **und** filtern")

col_q, col_lim = st.columns([3, 1])
with col_q:
    query = st.text_input("🔎 Suche (Titel, Künstler oder Genre)", placeholder="z. B. 'shape of you' | 'weeknd' | 'pop'")
with col_lim:
    max_hits = st.number_input("Trefferlimit", min_value=10, max_value=500, value=100, step=10)

min_year = int(df_meta["release_year"].replace(0, np.nan).min(skipna=True) or 0)
max_year = int(df_meta["release_year"].max())
if min_year == 0:
    min_year = 1900
yr1, yr2 = st.slider("📅 Erscheinungsjahr", min_value=min_year, max_value=max_year if max_year > 0 else 2030,
                     value=(min_year, max_year if max_year > 0 else min_year))

exp_choice = st.radio("🧼 Explizit-Status", ["Alle", "Nur Clean (explicit=0)", "Nur Explicit (explicit=1)"], horizontal=True)
genre_vocab = extract_genre_vocab(df_meta, top_k=200)
selected_genres = st.multiselect("🏷️ Genres (beliebig viele; Match = mind. eines)", options=genre_vocab)

mask_year = (df_meta["release_year"] >= yr1) & (df_meta["release_year"] <= yr2)
if exp_choice == "Nur Clean (explicit=0)":
    mask_exp = df_meta["explicit"].eq(0)
elif exp_choice == "Nur Explicit (explicit=1)":
    mask_exp = df_meta["explicit"].eq(1)
else:
    mask_exp = pd.Series([True] * len(df_meta), index=df_meta.index)
mask_genre = genre_filter_mask(df_meta, selected_genres)
mask_all = mask_year & mask_exp & mask_genre

df_view = df_meta[mask_all].copy()
if df_view.empty:
    st.warning("Kein Eintrag erfüllt aktuell die Filter. Passe Filter/Zeitraum an.")
    st.stop()

# ---------- Suche in gefilterter Sicht ----------
if query.strip():
    cand_idx = search_candidates(df_view, query, limit=max_hits)
    if len(cand_idx) == 0:
        st.warning("Keine Treffer innerhalb der gesetzten Filter. Suchbegriff oder Filter ändern.")
        st.stop()
    choice = st.selectbox(
        f"Treffer ({len(cand_idx)}):",
        options=cand_idx,
        format_func=lambda i: df_view.loc[i, "display"],
        index=0,
    )
else:
    st.caption("Tipp: Nutze die Suche oben, um schneller zu finden.")
    all_idx = df_view.index
    if len(all_idx) > max_hits:
        st.caption(f"Zeige die ersten {max_hits} von {len(all_idx)} gefilterten Songs.")
    choice = st.selectbox(
        "Gefilterte Songs:",
        options=all_idx[:max_hits],
        format_func=lambda i: df_view.loc[i, "display"],
        index=0,
    )

# ---------- Empfehlungen ----------
st.subheader("2) Empfehlungen")

# Optionen für Darstellung der Ergebnisse
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    show_players = st.checkbox("🎵 Inline-Player anzeigen", value=True, help="Zeigt Spotify-Embed-Player für die Top-Ergebnisse.")
with c2:
    players_n = st.number_input("Anzahl Player", min_value=1, max_value=20, value=5, step=1)
with c3:
    open_links_newtab = st.checkbox("Links in neuem Tab öffnen", value=True)

if st.button("Empfehlungen berechnen", type="primary", use_container_width=True):
    src_row = df_meta.loc[int(choice)]
    q = build_query_vector(artifact, src_row)

    distances, indices = nn.kneighbors(q, n_neighbors=min(int(top_k) + 1, len(df_meta)))
    idxs_all = [i for i in indices[0].tolist() if i != int(choice)]
    # auf Filter einschränken
    idxs = [i for i in idxs_all if i in df_view.index][: int(top_k)]

    i2pos = {i: p for p, i in enumerate(idxs_all)}
    scores = [1.0 - distances[0][i2pos[i]] for i in idxs]

    rec_df = df_meta.iloc[idxs][
        ["track_name", "artist_name", "artist_genres", "release_year", "explicit", "track_id"]
    ].copy()
    rec_df.insert(0, "score", np.round(scores, 4))
    rec_df["spotify_url"] = "https://open.spotify.com/track/" + rec_df["track_id"].astype(str)

    # 🔗 Klickbare Links in der Tabelle
    st.dataframe(
        rec_df.rename(columns={"spotify_url": "link"}),
        use_container_width=True,
        hide_index=True,
        column_config={
            "link": st.column_config.LinkColumn(
                "Spotify", display_text="Öffnen", help="Im Browser öffnen",
            ),
            "score": st.column_config.NumberColumn("score", format="%.3f"),
            "explicit": st.column_config.NumberColumn("explicit"),
        },
    )

    st.download_button(
        "⤓ Empfehlungen als CSV",
        data=rec_df.to_csv(index=False).encode("utf-8"),
        file_name=f"recs_{model_choice}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # 🎧 Inline-Player (Embed) für die ersten N
    if show_players:
        st.markdown("#### Sofort anhören")
        show_n = min(players_n, len(rec_df))
        for i in range(show_n):
            t = rec_df.iloc[i]
            with st.container(border=True):
                st.markdown(f"**{t.track_name} — {t.artist_name}**  ·  {t.release_year}  ·  {t.artist_genres}")
                # Spotify-Embed (funktioniert ohne preview_url)
                spotify_embed_iframe(str(t.track_id), height=80)
                if open_links_newtab:
                    st.markdown(f"[In Spotify öffnen]({t.spotify_url})")
                else:
                    st.link_button("In Spotify öffnen", t.spotify_url, use_container_width=False)




