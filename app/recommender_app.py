from __future__ import annotations
import os, json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, hstack
import streamlit.components.v1 as components

# ----------------------------------------------------
# App-Setup
# ----------------------------------------------------
st.set_page_config(page_title="Song Recommender (Model Registry)", layout="wide")

MODELS_DIR = os.getenv("MODELS_DIR", "models")
REGISTRY = os.path.join(MODELS_DIR, "registry.json")

# ----------------------------------------------------
# Caching
# ----------------------------------------------------
@st.cache_data(show_spinner=True)
def load_registry(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=True)
def load_model(path: str) -> dict:
    """
    Lädt ein .joblib und normalisiert es auf ein Dict-Artifact
    mit Keys: df_meta, nn, X_norm, vec_genre, vec_artist, vec_title, scaler, config.
    """
    obj = joblib.load(path)

    # Fall A: schon kompatibles Dict
    if isinstance(obj, dict):
        art = obj.copy()
    else:
        # Fall B: Pipeline/Objekt/Tuple heuristisch zerlegen
        art = {}
        for k in ["df_meta", "nn", "X_norm", "vec_genre", "vec_artist", "vec_title", "scaler", "config"]:
            try:
                v = getattr(obj, k)
                if v is not None:
                    art[k] = v
            except Exception:
                pass

        if not art and isinstance(obj, (list, tuple)):
            for item in obj:
                # df_meta erkennen
                try:
                    if hasattr(item, "columns") and {"track_name", "artist_name"}.issubset(set(item.columns)):
                        art["df_meta"] = item
                        continue
                except Exception:
                    pass
                # NN erkennen
                try:
                    if isinstance(item, NearestNeighbors):
                        art["nn"] = item
                        continue
                except Exception:
                    pass
                # Matrix (X_norm) erkennen
                try:
                    if hasattr(item, "shape") and "X_norm" not in art:
                        art["X_norm"] = item
                        continue
                except Exception:
                    pass
                # config erkennen
                if isinstance(item, dict) and any(k in item for k in ["genre_weight","artist_weight","title_weight","numeric_weight"]):
                    art["config"] = item

    # Defaults absichern
    art.setdefault("df_meta", pd.DataFrame())
    cfg = art.get("config", {}) or {}
    cfg.setdefault("genre_weight", 1.0)
    cfg.setdefault("artist_weight", 1.0)
    cfg.setdefault("title_weight", 0.2)
    cfg.setdefault("numeric_weight", 0.6)
    art["config"] = cfg

    # df_meta konsistent machen
    if not art["df_meta"].empty:
        dfm = art["df_meta"].copy()
        if "display" not in dfm.columns:
            dfm["display"] = dfm.get("track_name", "").astype(str) + " — " + dfm.get("artist_name", "").astype(str)
        if "release_year" not in dfm.columns:
            dfm["release_year"] = 0
        if "explicit" not in dfm.columns:
            dfm["explicit"] = 0
        dfm["release_year"] = pd.to_numeric(dfm["release_year"], errors="coerce").fillna(0).astype(int)
        dfm["explicit"] = pd.to_numeric(dfm["explicit"], errors="coerce").fillna(0).astype(int)
        art["df_meta"] = dfm

    return art

# ----------------------------------------------------
# Helper
# ----------------------------------------------------
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

def genre_filter_mask(df: pd.DataFrame, selected: list[str], all_genres: bool) -> pd.Series:
    if all_genres or not selected:
        return pd.Series([True] * len(df), index=df.index)
    hay = df.get("artist_genres", "").fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for g in selected:
        mask |= hay.str.contains(g.lower(), regex=False)
    return mask

def contains_mask(series: pd.Series, query: str) -> pd.Series:
    q = query.strip().lower()
    if not q:
        return pd.Series([True] * len(series), index=series.index)
    return series.fillna("").str.lower().str.contains(q, regex=False)

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
    src = f"https://open.spotify.com/embed/track/{track_id}"
    components.iframe(src=src, height=height, scrolling=False)

def build_query_vector(artifact: dict, row: pd.Series,
                       gw: float, aw: float, tw: float, nw: float) -> csr_matrix:
    vec_genre = artifact.get("vec_genre")
    vec_artist = artifact.get("vec_artist")
    vec_title = artifact.get("vec_title")
    scaler    = artifact.get("scaler")

    Xg = vec_genre.transform([str(row.get("artist_genres", ""))]) if vec_genre is not None else None
    Xa = vec_artist.transform([str(row.get("artist_name", ""))])  if vec_artist is not None else None
    Xt = vec_title.transform([str(row.get("track_name", ""))])    if vec_title is not None else None

    year = float(row.get("release_year", 0))
    explicit = float(row.get("explicit", 0))
    if scaler is not None:
        Xn = csr_matrix(scaler.transform(np.array([[year, explicit]])))
    else:
        Xn = csr_matrix([[year, explicit]])

    blocks = []
    if Xg is not None: blocks.append(Xg.multiply(gw))
    if Xa is not None: blocks.append(Xa.multiply(aw))
    if Xt is not None: blocks.append(Xt.multiply(tw))
    blocks.append(Xn.multiply(nw))

    q = hstack(blocks).tocsr()
    qn = np.sqrt(q.multiply(q).sum())
    return q if qn == 0 else q.multiply(1.0 / qn)

def diversify_by_artist(rec_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Max. 1 Track pro Artist – behält Reihenfolge/Score bei."""
    seen = set()
    rows = []
    for _, r in rec_df.iterrows():
        artist = str(r.get("artist_name", "")).strip().lower()
        if artist and artist not in seen:
            rows.append(r)
            seen.add(artist)
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)

# ----------------------------------------------------
# UI – Sidebar
# ----------------------------------------------------
with st.sidebar:
    st.header("📦 Modell wählen")

    # Registry laden + Reload-Button
    colA, colB = st.columns([3, 1])
    with colA:
        st.caption(f"Registry: `{REGISTRY}`")
    with colB:
        if st.button("🔄 Reload", help="Registry & Cache neu laden"):
            load_registry.clear()
            load_model.clear()
            st.experimental_rerun()

    reg = load_registry(REGISTRY)
    if not reg:
        st.error("Keine Modelle gefunden. Trainiere z. B.:\n\n"
                 "`python -m scripts.train_models --csv data/raw/mix_master.csv --out models`")
        st.stop()

    model_names = sorted(reg.keys())
    model_choice = st.selectbox("Modell", model_names)
    model_path = reg[model_choice]["path"]

    st.divider()
    st.header("⚖️ Gewichtung (überschreibt Modell-Config)")

    # Safe load für Defaults
    try:
        _tmp_artifact = load_model(model_path)
    except Exception as e:
        st.error(f"Fehler beim Laden des Modells:\n{e}")
        st.stop()

    cfg = _tmp_artifact["config"]  # durch Loader garantiert

    gw0 = float(cfg["genre_weight"])
    aw0 = float(cfg["artist_weight"])
    tw0 = float(cfg["title_weight"])
    nw0 = float(cfg["numeric_weight"])

    gw = st.slider("Genre", 0.0, 3.0, gw0, 0.1)
    aw = st.slider("Artist", 0.0, 3.0, aw0, 0.1)
    tw = st.slider("Titel", 0.0, 3.0, tw0, 0.1)
    nw = st.slider("Numerisch (Jahr, explicit)", 0.0, 3.0, nw0, 0.1)

    st.divider()
    st.header("🔧 Empfehlungseinstellungen")
    top_k = st.number_input("Top-K", min_value=3, max_value=50, value=10, step=1)
    score_thr = st.slider("Score-Schwelle (≥)", 0.0, 1.0, 0.20, 0.01,
                          help="Nur Empfehlungen mit Score ≥ Schwelle (Score = 1 − Distanz)")
    one_per_artist = st.checkbox("Max. 1 Track pro Artist", value=True)

# ----------------------------------------------------
# Haupt-Inhalt
# ----------------------------------------------------
st.title("🎧 Content-Based Song Recommender")

# Artefakt laden
try:
    artifact = load_model(model_path)
except Exception as e:
    st.error(f"Fehler beim Laden des Modells:\n{e}")
    st.stop()

df_meta: pd.DataFrame = artifact["df_meta"].copy()
nn: NearestNeighbors | None = artifact.get("nn")
X_norm = artifact.get("X_norm")

if df_meta.empty or nn is None or X_norm is None:
    st.error("Modell-Artefakt unvollständig: Erwartet `df_meta`, `nn`, `X_norm`.")
    st.stop()

# Anzeige & Felder
df_meta["display"] = df_meta["track_name"].fillna("").astype(str) + " — " + df_meta["artist_name"].fillna("").astype(str)
if "release_year" not in df_meta.columns:
    df_meta["release_year"] = 0
df_meta["release_year"] = pd.to_numeric(df_meta["release_year"], errors="coerce").fillna(0).astype(int)
if "explicit" not in df_meta.columns:
    df_meta["explicit"] = 0
df_meta["explicit"] = pd.to_numeric(df_meta["explicit"], errors="coerce").fillna(0).astype(int)

# ----------------------------------------------------
# 1) Suchen & Filtern
# ----------------------------------------------------
st.subheader("1) Suchen & Filtern")

col_q, col_lim = st.columns([3, 1])
with col_q:
    query = st.text_input("🔎 Suche (Titel, Künstler oder Genre)", placeholder="z. B. 'shape of you' | 'weeknd' | 'pop'")
with col_lim:
    max_hits = st.number_input("Trefferlimit", min_value=10, max_value=500, value=100, step=10)

min_year = int(df_meta["release_year"].replace(0, np.nan).min(skipna=True) or 0)
max_year = int(df_meta["release_year"].max())
if min_year == 0:
    min_year = 1900
yr1, yr2 = st.slider(
    "📅 Erscheinungsjahr",
    min_value=min_year,
    max_value=max_year if max_year > 0 else 2030,
    value=(min_year, max_year if max_year > 0 else min_year),
)

exp_choice = st.radio(
    "🧼 Explizit-Status",
    ["Alle", "Nur Clean (explicit=0)", "Nur Explicit (explicit=1)"],
    horizontal=True,
    help="Spotify markiert Songs mit 'explicit' (Schimpfwörter etc.).",
)

genre_vocab = extract_genre_vocab(df_meta, top_k=200)
col_g1, col_g2 = st.columns([3, 1])
with col_g1:
    selected_genres = st.multiselect("🏷️ Genres (Match = enthält eines)", options=genre_vocab)
with col_g2:
    all_genres = st.checkbox("Alle Genres", value=True, help="Wenn aktiv, wird nicht nach Genres gefiltert.")

# Masken
mask_year = (df_meta["release_year"] >= yr1) & (df_meta["release_year"] <= yr2)
if exp_choice == "Nur Clean (explicit=0)":
    mask_exp = df_meta["explicit"].eq(0)
elif exp_choice == "Nur Explicit (explicit=1)":
    mask_exp = df_meta["explicit"].eq(1)
else:
    mask_exp = pd.Series([True] * len(df_meta), index=df_meta.index)
mask_genre = genre_filter_mask(df_meta, selected_genres, all_genres)
mask_all = mask_year & mask_exp & mask_genre

df_view = df_meta[mask_all].copy()
if df_view.empty:
    st.warning("Kein Eintrag erfüllt aktuell die Filter. Passe Filter/Zeitraum an.")
    st.stop()

# Trefferliste
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

# ----------------------------------------------------
# 2) Empfehlungen
# ----------------------------------------------------
st.subheader("2) Empfehlungen")

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    show_players = st.checkbox("🎵 Inline-Player anzeigen", value=True,
                               help="Zeigt Spotify-Embed-Player für die Top-Ergebnisse.")
with c2:
    players_n = st.number_input("Anzahl Player", min_value=1, max_value=20, value=5, step=1)
with c3:
    open_links_newtab = st.checkbox("Links in neuem Tab öffnen", value=True)

if st.button("Empfehlungen berechnen", type="primary", use_container_width=True):
    src_row = df_meta.loc[int(choice)]
    q = build_query_vector(artifact, src_row, gw=gw, aw=aw, tw=tw, nw=nw)

    # KNN (etwas Oversampling, um Filter/Diversität anwenden zu können)
    k_req = min(int(top_k) + 50, len(df_meta))
    distances, indices = nn.kneighbors(q, n_neighbors=k_req)
    idxs_all = [i for i in indices[0].tolist() if i != int(choice)]

    # Scores (= 1 − Distanz)
    i2pos = {i: p for p, i in enumerate(indices[0].tolist())}
    scores_all = {i: float(1.0 - distances[0][i2pos[i]]) for i in idxs_all}

    # Nach Filter-Sicht einschränken
    idxs = [i for i in idxs_all if i in df_view.index]
    # Score-Schwelle
    idxs = [i for i in idxs if scores_all.get(i, 0.0) >= float(score_thr)]

    # Ergebnis-DF
    cols = ["track_name", "artist_name", "artist_genres", "release_year", "explicit", "track_id"]
    cols = [c for c in cols if c in df_meta.columns]
    rec_df = df_meta.iloc[idxs][cols].copy()
    rec_df.insert(0, "score", rec_df.index.map(lambda i: np.round(scores_all.get(i, 0.0), 4)))
    rec_df = rec_df.sort_values("score", ascending=False)

    # Diversität
    if one_per_artist:
        rec_df = diversify_by_artist(rec_df, top_k=int(top_k))
    else:
        rec_df = rec_df.head(int(top_k))

    # Spotify-Links
    if "track_id" in rec_df.columns:
        rec_df["spotify_url"] = "https://open.spotify.com/track/" + rec_df["track_id"].astype(str)
    else:
        rec_df["spotify_url"] = ""

    # Tabelle
    colcfg = {
        "score": st.column_config.NumberColumn("score", format="%.3f"),
    }
    if "explicit" in rec_df.columns:
        colcfg["explicit"] = st.column_config.NumberColumn("explicit")
    colcfg["spotify_url"] = st.column_config.LinkColumn("Spotify", display_text="Öffnen")

    st.dataframe(
        rec_df.rename(columns={"spotify_url": "spotify_url"}),
        use_container_width=True, hide_index=True,
        column_config=colcfg,
    )

    # Download
    st.download_button(
        "⤓ Empfehlungen als CSV",
        data=rec_df.to_csv(index=False).encode("utf-8"),
        file_name=f"recs_{model_choice}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Inline-Player
    if show_players and "track_id" in rec_df.columns and len(rec_df):
        st.markdown("#### Sofort anhören")
        show_n = min(int(players_n), len(rec_df))
        for i in range(show_n):
            t = rec_df.iloc[i]
            with st.container(border=True):
                meta = f"{int(t.release_year) if 'release_year' in rec_df.columns and pd.notnull(t.release_year) else ''}"
                st.markdown(f"**{t.get('track_name','')} — {t.get('artist_name','')}**  ·  {meta}  ·  {t.get('artist_genres','')}")
                if pd.notnull(t.get("track_id", "")) and str(t.get("track_id", "")).strip():
                    spotify_embed_iframe(str(t.track_id), height=80)
                url = t.get("spotify_url", "")
                if isinstance(url, str) and url:
                    if open_links_newtab:
                        st.markdown(f"[In Spotify öffnen]({url})")
                    else:
                        st.link_button("In Spotify öffnen", url, use_container_width=False)

