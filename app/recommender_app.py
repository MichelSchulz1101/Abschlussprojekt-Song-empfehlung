from __future__ import annotations
import os, json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, hstack, coo_matrix
from sklearn.preprocessing import normalize as sk_normalize
import streamlit.components.v1 as components
import plotly.express as px

st.set_page_config(page_title="Song Recommender (Model Registry)", layout="wide")

MODELS_DIR = os.getenv("MODELS_DIR", "models")
REGISTRY = os.path.join(MODELS_DIR, "registry.json")

# ---------- Cache ----------
@st.cache_data(show_spinner=True)
def load_registry(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=True)
def load_model(path: str) -> dict:
    return joblib.load(path)

# ---------- Hilfsfunktionen ----------
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

    Xg = vec_genre.transform([str(row.get("artist_genres", ""))]) if vec_genre else None
    Xa = vec_artist.transform([str(row.get("artist_name", ""))])  if vec_artist else None
    Xt = vec_title.transform([str(row.get("track_name", ""))])    if vec_title else None

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

# ----- RF-Artist Query Builder -----
def build_content_row_raw(artifact: dict, row: pd.Series,
                          gw: float, aw: float, tw: float, nw: float) -> csr_matrix:
    vec_genre = artifact.get("vec_genre")
    vec_artist = artifact.get("vec_artist")
    vec_title = artifact.get("vec_title")
    scaler    = artifact.get("scaler")

    Xg = vec_genre.transform([str(row.get("artist_genres", ""))]) if vec_genre else None
    Xa = vec_artist.transform([str(row.get("artist_name", ""))])  if vec_artist else None
    Xt = vec_title.transform([str(row.get("track_name", ""))])    if vec_title else None

    year = float(row.get("release_year", 0))
    explicit = float(row.get("explicit", 0))
    Xn = csr_matrix(scaler.transform(np.array([[year, explicit]]))) if scaler is not None else csr_matrix([[year, explicit]])

    blocks = []
    if Xg is not None: blocks.append(Xg.multiply(gw))
    if Xa is not None: blocks.append(Xa.multiply(aw))
    if Xt is not None: blocks.append(Xt.multiply(tw))
    blocks.append(Xn.multiply(nw))
    return hstack(blocks).tocsr()

def build_query_vector_rfleaf(artifact: dict, row: pd.Series,
                              gw: float, aw: float, tw: float, nw: float) -> csr_matrix:
    rf = artifact["rf"]
    leaf_maps = artifact["rf_leaf_maps"]
    dim = int(artifact["rf_leaf_dim"])

    Xs = build_content_row_raw(artifact, row, gw, aw, tw, nw)
    X_dense = Xs.astype(np.float32).toarray()
    leaves = rf.apply(X_dense)

    rows, cols, data = [], [], []
    for t, leaf_id in enumerate(leaves[0]):
        col = leaf_maps[t].get(int(leaf_id))
        if col is not None:
            rows.append(0); cols.append(int(col)); data.append(1.0)

    X_leaf = coo_matrix((data, (rows, cols)), shape=(1, dim)).tocsr()
    X_leaf = sk_normalize(X_leaf, norm="l2", axis=1, copy=False)
    return X_leaf

def diversify_by_artist(rec_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
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

# ---------- UI ----------
st.title("🎧 Content-Based Song Recommender")

with st.sidebar:
    st.header("📦 Modell wählen")
    reg = load_registry(REGISTRY)
    if not reg:
        st.error("Keine Modelle gefunden.")
        st.stop()
    model_names = sorted(reg.keys())
    model_choice = st.selectbox("Modell", model_names)
    model_path = reg[model_choice]["path"]

    st.divider()
    st.header("⚖️ Gewichtung")
    _tmp_artifact = load_model(model_path)
    cfg = _tmp_artifact.get("config", {})
    gw0 = float(cfg.get("genre_weight", 1.0))
    aw0 = float(cfg.get("artist_weight", 1.0))
    tw0 = float(cfg.get("title_weight", 0.2))
    nw0 = float(cfg.get("numeric_weight", 0.6))

    gw = st.slider("Genre", 0.0, 3.0, gw0, 0.1)
    aw = st.slider("Artist", 0.0, 3.0, aw0, 0.1)
    tw = st.slider("Titel", 0.0, 3.0, tw0, 0.1)
    nw = st.slider("Numerisch", 0.0, 3.0, nw0, 0.1)

    st.divider()
    st.header("🔧 Empfehlungseinstellungen")
    top_k = st.number_input("Top-K", min_value=3, max_value=50, value=10, step=1)
    score_thr = st.slider("Score-Schwelle", 0.0, 1.0, 0.20, 0.01)
    one_per_artist = st.checkbox("Max. 1 Track pro Artist", value=True)

artifact = load_model(model_path)
df_meta: pd.DataFrame = artifact["df_meta"].copy()
nn: NearestNeighbors = artifact["nn"]
X_norm = artifact["X_norm"]

df_meta["display"] = df_meta["track_name"].fillna("").astype(str) + " — " + df_meta["artist_name"].fillna("").astype(str)
df_meta["release_year"] = pd.to_numeric(df_meta["release_year"], errors="coerce").fillna(0).astype(int)
df_meta["explicit"] = pd.to_numeric(df_meta["explicit"], errors="coerce").fillna(0).astype(int)

# ---------- Filter ----------
st.subheader("1) Suchen & Filtern")

query = st.text_input("🔎 Suche", placeholder="Titel, Künstler oder Genre")
max_hits = st.number_input("Trefferlimit", min_value=10, max_value=500, value=100, step=10)

min_year = int(df_meta["release_year"].replace(0, np.nan).min(skipna=True) or 1900)
max_year = int(df_meta["release_year"].max() or 2030)
yr1, yr2 = st.slider("📅 Erscheinungsjahr", min_value=min_year, max_value=max_year, value=(min_year, max_year))

exp_choice = st.radio("🧼 Explizit", ["Alle", "Nur Clean", "Nur Explicit"], horizontal=True)
genre_vocab = extract_genre_vocab(df_meta, top_k=200)
selected_genres = st.multiselect("🏷️ Genres", options=genre_vocab)
all_genres = st.checkbox("Alle Genres", value=True)

mask_year = (df_meta["release_year"] >= yr1) & (df_meta["release_year"] <= yr2)
mask_exp = pd.Series([True] * len(df_meta), index=df_meta.index)
if exp_choice == "Nur Clean":
    mask_exp = df_meta["explicit"].eq(0)
elif exp_choice == "Nur Explicit":
    mask_exp = df_meta["explicit"].eq(1)
mask_genre = genre_filter_mask(df_meta, selected_genres, all_genres)
mask_all = mask_year & mask_exp & mask_genre

df_view = df_meta[mask_all].copy()
if df_view.empty:
    st.warning("Keine Songs gefunden.")
    st.stop()

if query.strip():
    cand_idx = search_candidates(df_view, query, limit=max_hits)
    if len(cand_idx) == 0:
        st.warning("Keine Treffer.")
        st.stop()
    choice = st.selectbox("Treffer:", options=cand_idx, format_func=lambda i: df_view.loc[i, "display"])
else:
    all_idx = df_view.index
    choice = st.selectbox("Gefilterte Songs:", options=all_idx[:max_hits], format_func=lambda i: df_view.loc[i, "display"])

# ---------- Empfehlungen ----------
st.subheader("2) Empfehlungen")

if st.button("Empfehlungen berechnen", type="primary"):
    src_row = df_meta.loc[int(choice)]
    model_type = artifact.get("config", {}).get("model_type", "")

    if model_type == "rf-artist":
        q = build_query_vector_rfleaf(artifact, src_row, gw, aw, tw, nw)
    else:
        q = build_query_vector(artifact, src_row, gw, aw, tw, nw)

    k_req = min(int(top_k) + 50, len(df_meta))
    distances, indices = nn.kneighbors(q, n_neighbors=k_req)
    idxs_all = [i for i in indices[0].tolist() if i != int(choice)]
    scores_all = {i: float(1.0 - distances[0][j]) for j, i in enumerate(indices[0]) if i != int(choice)}

    idxs = [i for i in idxs_all if i in df_view.index and scores_all.get(i, 0.0) >= float(score_thr)]

    rec_df = df_meta.iloc[idxs][["track_name", "artist_name", "artist_genres", "release_year", "explicit", "track_id"]].copy()
    rec_df.insert(0, "score", rec_df.index.map(lambda i: np.round(scores_all.get(i, 0.0), 4)))
    rec_df = rec_df.sort_values("score", ascending=False)
    if one_per_artist:
        rec_df = diversify_by_artist(rec_df, top_k=int(top_k))
    else:
        rec_df = rec_df.head(int(top_k))
    rec_df["spotify_url"] = "https://open.spotify.com/track/" + rec_df["track_id"].astype(str)

    st.dataframe(rec_df, use_container_width=True)

# ---------- Cluster-Visualisierung ----------
st.subheader("3) Cluster-Visualisierung")

def project_2d(_model_choice: str, _X_norm: csr_matrix):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    return svd.fit_transform(_X_norm)

has_kmeans = ("kmeans" in artifact) and ("cluster_kmeans" in df_meta.columns)
with st.spinner("Berechne 2D-Projektion …"):
    coords = project_2d(model_choice, X_norm)
vis_df = df_meta.copy()
vis_df["x"] = coords[:, 0]
vis_df["y"] = coords[:, 1]

fig = px.scatter(vis_df, x="x", y="y", color="cluster_kmeans" if has_kmeans else None,
                 hover_name="display", opacity=0.7, title="2D-Embedding")
st.plotly_chart(fig, use_container_width=True)
