from __future__ import annotations
import os, json, math, random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix, hstack, coo_matrix
from sklearn.preprocessing import normalize as sk_normalize
import streamlit.components.v1 as components
import plotly.express as px

st.set_page_config(page_title="Song-Empfehlung", layout="wide")

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
    vec_title  = artifact.get("vec_title")
    scaler     = artifact.get("scaler")

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
    vec_title  = artifact.get("vec_title")
    scaler     = artifact.get("scaler")

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
st.title("🎧 Song-Empfehlung 🎵")

with st.sidebar:
    st.header("Modell wählen")
    reg = load_registry(REGISTRY)
    if not reg:
        st.error("Keine Modelle gefunden. Trainiere z. B.: `python -m scripts.train_models --csv data/raw/mix_master.csv --out models`")
        st.stop()
    model_names = sorted(reg.keys())
    model_choice = st.selectbox("Modell", model_names)
    model_path = reg[model_choice]["path"]

    st.divider()
    st.header("Gewichtung (überschreibt Modell-Config)")
    _tmp_artifact = load_model(model_path)
    cfg = _tmp_artifact.get("config", {})
    gw0 = float(cfg.get("genre_weight", 1.0))
    aw0 = float(cfg.get("artist_weight", 1.0))
    tw0 = float(cfg.get("title_weight", 0.2))
    nw0 = float(cfg.get("numeric_weight", 0.6))

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

artifact = load_model(model_path)
df_meta: pd.DataFrame = artifact["df_meta"].copy()
nn: NearestNeighbors = artifact["nn"]
X_norm = artifact["X_norm"]

# Anzeige + Felder konsistent setzen
df_meta["display"] = df_meta["track_name"].fillna("").astype(str) + " — " + df_meta["artist_name"].fillna("").astype(str)
if "release_year" not in df_meta.columns:
    df_meta["release_year"] = 0
df_meta["release_year"] = pd.to_numeric(df_meta["release_year"], errors="coerce").fillna(0).astype(int)
if "explicit" not in df_meta.columns:
    df_meta["explicit"] = 0
df_meta["explicit"] = pd.to_numeric(df_meta["explicit"], errors="coerce").fillna(0).astype(int)

# ---------- Filterleiste ----------
st.subheader("1) Suchen & Filtern nach Song vorlage")

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

exp_choice = st.radio("🧼 Explizit-Status",
                      ["Alle", "Nur Clean (explicit=0)", "Nur Explicit (explicit=1)"],
                      horizontal=True,
                      help="Spotify markiert Songs mit 'explicit' (Schimpfwörter etc.).")

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

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    show_players = st.checkbox("🎵 Inline-Player anzeigen", value=True,
                               help="Zeigt Spotify-Embed-Player oder Preview-Clip für die Top-Ergebnisse.")
with c2:
    players_n = st.number_input("Anzahl Player", min_value=1, max_value=20, value=5, step=1)
with c3:
    open_links_newtab = st.checkbox("Links in neuem Tab öffnen", value=True)

def _recommend_indices_for_row(_artifact, _nn, _df_meta, _row, _gw, _aw, _tw, _nw, k):
    model_type = _artifact.get("config", {}).get("model_type", "")
    if model_type == "rf-artist":
        q = build_query_vector_rfleaf(_artifact, _row, _gw, _aw, _tw, _nw)
    else:
        q = build_query_vector(_artifact, _row, _gw, _aw, _tw, _nw)
    k_req = min(int(k), len(_df_meta))
    distances, indices = _nn.kneighbors(q, n_neighbors=k_req)
    idxs_all = [i for i in indices[0].tolist()]
    return distances[0], idxs_all

if st.button("Empfehlungen berechnen", type="primary", use_container_width=True):
    src_row = df_meta.loc[int(choice)]
    # KNN abrufen (etwas mehr als Top-K, damit wir filtern/diversifizieren können)
    distances, idxs_all = _recommend_indices_for_row(artifact, nn, df_meta, src_row, gw, aw, tw, nw, k=min(int(top_k)+50, len(df_meta)))
    # Self entfernen
    idxs_all = [i for i in idxs_all if i != int(choice)]

    # Scores (= 1 − Distanz)
    i2pos = {i: p for p, i in enumerate(idxs_all)}
    # Achtung: distances bezieht sich auf die top-Liste inklusive self; robust ermitteln:
    # wir nehmen Score später aus tatsächlicher Nachbarschaftsreihenfolge erneut:
    scores_all = {}
    for rank, i in enumerate(idxs_all):
        d = nn.kneighbors_graph(df_meta.iloc[[int(choice)]], n_neighbors=rank+2).toarray() if False else None  # placeholder
        # einfache Rekonstruktion: wir haben keine 1:1 Distanz hier; nutze nochmal kneighbors für q:
        pass
    # Einfach korrekt: berechne noch einmal Distanzliste über das ursprüngliche call-Result:
    # Wir rechnen Scores direkt mit erneutem Call (klein k), aber effizienter:
    # Besser: hole komplettes Paket inkl. Distanzen:
    # Um es stabil zu halten, machen wir den ersten Call nochmal mit exakt dieser Länge und nutzen distances.
    kfull = min(len(df_meta), int(top_k)+50)
    if True:
        distances_full, indices_full = nn.kneighbors(
            build_query_vector(artifact, src_row, gw, aw, tw, nw)
            if artifact.get("config", {}).get("model_type","")!="rf-artist"
            else build_query_vector_rfleaf(artifact, src_row, gw, aw, tw, nw),
            n_neighbors=kfull
        )
        neigh = [(int(indices_full[0][i]), float(1.0 - distances_full[0][i])) for i in range(len(indices_full[0]))]
        neigh = [(idx, sc) for idx, sc in neigh if idx != int(choice)]
        idxs_all = [idx for idx, _ in neigh]
        scores_all = {idx: sc for idx, sc in neigh}

    # Auf Filter-Sicht einschränken
    idxs = [i for i in idxs_all if i in df_view.index]

    # Score-Schwelle anwenden
    idxs = [i for i in idxs if scores_all.get(i, 0.0) >= float(score_thr)]

    # DataFrame bauen
    rec_df = df_meta.iloc[idxs][
        ["track_name", "artist_name", "artist_genres", "release_year", "explicit", "track_id"]
    ].copy()
    rec_df.insert(0, "score", rec_df.index.map(lambda i: np.round(scores_all.get(i, 0.0), 4)))
    rec_df = rec_df.sort_values("score", ascending=False)

    # Diversität: max. 1 Track pro Artist
    if one_per_artist:
        rec_df = diversify_by_artist(rec_df, top_k=int(top_k))
    else:
        rec_df = rec_df.head(int(top_k))

    # Links
    rec_df["spotify_url"] = "https://open.spotify.com/track/" + rec_df["track_id"].astype(str)

    # Tabelle
    st.dataframe(
        rec_df.rename(columns={"spotify_url": "link"}),
        use_container_width=True, hide_index=True,
        column_config={
            "link": st.column_config.LinkColumn("Spotify", display_text="Öffnen"),
            "score": st.column_config.NumberColumn("score", format="%.3f"),
            "explicit": st.column_config.NumberColumn("explicit"),
        },
    )

    # Download
    st.download_button(
        "⤓ Empfehlungen als CSV",
        data=rec_df.to_csv(index=False).encode("utf-8"),
        file_name=f"recs_{model_choice}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Speichere letzte Indizes für Visualisierung (optional)
    st.session_state["last_recs_idx"] = rec_df.index.tolist()
    st.session_state["last_query_idx"] = int(choice)

    # Inline-Player
    if show_players and len(rec_df):
        st.markdown("#### Sofort anhören")
        show_n = min(int(players_n), len(rec_df))
        for i in range(show_n):
            t = rec_df.iloc[i]
            with st.container(border=True):
                meta = f"{int(t.release_year) if pd.notnull(t.release_year) else ''}"
                st.markdown(f"**{t.track_name} — {t.artist_name}**  ·  {meta}  ·  {t.artist_genres}")
                # 30s-Preview, falls vorhanden
                preview_url = None
                if "preview_url" in df_meta.columns:
                    pu = df_meta.loc[df_meta["track_id"] == t.track_id, "preview_url"]
                    if len(pu) and pd.notnull(pu.iloc[0]) and str(pu.iloc[0]).strip():
                        preview_url = str(pu.iloc[0]).strip()
                if preview_url:
                    st.audio(preview_url, format="audio/mp3")
                else:
                    spotify_embed_iframe(str(t.track_id), height=80)
                if open_links_newtab:
                    st.markdown(f"[In Spotify öffnen]({t.spotify_url})")
                else:
                    st.link_button("In Spotify öffnen", t.spotify_url, use_container_width=False)

# ---------- 3) Modellbewertung ----------
st.subheader("3) Modellbewertung")

col_e1, col_e2, col_e3, col_e4 = st.columns([1.2, 1, 1, 1.2])
with col_e1:
    eval_mode = st.selectbox(
        "Relevanzdefinition",
        ["Artist-Match", "Genre-Overlap", "Artist ODER Genre"],
        help="Ohne Ground-Truth definieren wir Relevanz über Artist-Gleichheit und/oder Genre-Token-Overlap."
    )
with col_e2:
    eval_k = st.number_input("K für @K-Metriken", min_value=5, max_value=50, value=10, step=1)
with col_e3:
    n_seeds = st.number_input("Anzahl Seeds", min_value=50, max_value=2000, value=200, step=50,
                              help="Wie viele Zufalls-Songs als Query für die Bewertung.")
with col_e4:
    random_state = st.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)

def _tokenize_genres(s: str) -> set[str]:
    s = (s or "").lower().replace(";", ",").replace("|", ",")
    toks = [t.strip() for t in s.split(",") if t.strip()]
    return set(toks)

def _relevant_mask(df: pd.DataFrame, idx: int, mode: str) -> pd.Series:
    a = str(df.loc[idx, "artist_name"]).strip().lower()
    g = _tokenize_genres(df.loc[idx, "artist_genres"] if "artist_genres" in df.columns else "")
    if mode == "Artist-Match":
        return df["artist_name"].fillna("").str.strip().str.lower().eq(a)
    elif mode == "Genre-Overlap":
        if not g:
            return pd.Series(False, index=df.index)
        # mind. 1 Token gleich
        return df["artist_genres"].fillna("").apply(lambda x: len(_tokenize_genres(x).intersection(g)) > 0)
    else:
        m1 = df["artist_name"].fillna("").str.strip().str.lower().eq(a)
        m2 = df["artist_genres"].fillna("").apply(lambda x: len(_tokenize_genres(x).intersection(g)) > 0)
        return m1 | m2

def _precision_recall_f1(rels: list[int], k: int, n_rel: int) -> tuple[float,float,float]:
    hits = sum(rels[:k])
    prec = hits / k if k > 0 else 0.0
    rec  = hits / n_rel if n_rel > 0 else 0.0
    f1   = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return prec, rec, f1

def _avg_precision(rels: list[int], k: int) -> float:
    # MAP@K: Durchschnitt der Precision an jeder relevanten Position
    num, denom = 0.0, 0
    hit_so_far = 0
    for i in range(min(k, len(rels))):
        if rels[i]:
            hit_so_far += 1
            num += hit_so_far / (i+1)
            denom += 1
    return 0.0 if denom == 0 else num / denom

def _ndcg_binary(rels: list[int], k: int) -> float:
    def dcg(r):
        return sum((r[i] / math.log2(i+2) for i in range(len(r))))
    r = rels[:k]
    ideal = sorted(r, reverse=True)
    d = dcg(r)
    idcg = dcg(ideal)
    return 0.0 if idcg == 0 else d / idcg

def _recommend_top_indices(_artifact, _nn, _df, _idx, _gw,_aw,_tw,_nw, topk):
    row = _df.loc[int(_idx)]
    model_type = _artifact.get("config", {}).get("model_type", "")
    if model_type == "rf-artist":
        q = build_query_vector_rfleaf(_artifact, row, _gw,_aw,_tw,_nw)
    else:
        q = build_query_vector(_artifact, row, _gw,_aw,_tw,_nw)
    k_req = min(int(topk)+1, len(_df))
    distances, indices = _nn.kneighbors(q, n_neighbors=k_req)
    idxs = [int(i) for i in indices[0].tolist() if int(i) != int(_idx)]
    return idxs[:int(topk)]

def evaluate_model(_artifact, _nn, _df, _gw,_aw,_tw,_nw, mode: str, K: int, seeds: int, rng: int):
    rnd = random.Random(int(rng))
    all_indices = list(_df.index)
    if len(all_indices) == 0:
        return None
    # Seeds (ohne Duplikate, gleichmäßig)
    if seeds >= len(all_indices):
        seeds_idx = all_indices
    else:
        seeds_idx = rnd.sample(all_indices, k=int(seeds))

    precisions, recalls, f1s, maps, ndcgs = [], [], [], [], []
    all_recommended = set()
    artist_div_counts = []

    pb = st.progress(0)
    for t, sidx in enumerate(seeds_idx, start=1):
        rel_mask = _relevant_mask(_df, sidx, mode)
        rel_mask.iloc[sidx] = False  # Seed nicht mitzählen
        n_rel = int(rel_mask.sum())

        # Empfohlene Indizes
        recs = _recommend_top_indices(_artifact, _nn, _df, sidx, _gw,_aw,_tw,_nw, topk=K)
        all_recommended.update(recs)

        # Binäre Relevanzliste in Reihenfolge
        rels = [1 if rel_mask.iloc[r] else 0 for r in recs]

        p, r, f = _precision_recall_f1(rels, K, n_rel)
        ap = _avg_precision(rels, K)
        nd = _ndcg_binary(rels, K)

        precisions.append(p); recalls.append(r); f1s.append(f); maps.append(ap); ndcgs.append(nd)

        # Artist-Diversität in den Empfehlungen dieses Seeds
        arts = _df.loc[recs, "artist_name"].fillna("").str.lower().tolist()
        if len(recs) > 0:
            artist_div_counts.append(len(set(arts)) / len(recs))

        pb.progress(t / max(1, len(seeds_idx)))

    metrics = {
        "Precision@K": float(np.mean(precisions) if precisions else 0.0),
        "Recall@K":    float(np.mean(recalls) if recalls else 0.0),
        "F1@K":        float(np.mean(f1s) if f1s else 0.0),
        "MAP@K":       float(np.mean(maps) if maps else 0.0),
        "NDCG@K":      float(np.mean(ndcgs) if ndcgs else 0.0),
        "Artist-Diversity": float(np.mean(artist_div_counts) if artist_div_counts else 0.0),
        "Coverage":    float(len(all_recommended) / len(_df)) if len(_df) else 0.0,
    }
    return metrics

if st.button("Bewertung berechnen", type="secondary", use_container_width=True):
    with st.spinner("Bewertung läuft …"):
        metrics = evaluate_model(artifact, nn, df_meta, gw, aw, tw, nw,
                                 mode=eval_mode, K=int(eval_k), seeds=int(n_seeds), rng=int(random_state))
    if metrics is None:
        st.warning("Keine Daten für Bewertung.")
    else:
        # Tabelle
        mdf = (pd.Series(metrics).to_frame("Wert")
               .reset_index().rename(columns={"index": "Metrik"}))
        # Anzeige mit 3 Dezimalen
        mdf["Wert"] = mdf["Wert"].map(lambda x: round(float(x), 3))
        st.dataframe(mdf, hide_index=True, use_container_width=True)

        # Balkengrafik
        figm = px.bar(mdf, x="Metrik", y="Wert", title=f"Modellbewertung – {model_choice}",
                      range_y=[0, 1], text="Wert")
        figm.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(figm, use_container_width=True)

# =============== 4) Daten-Exploration (KPIs & Verteilungen) ===============
st.markdown("---")
st.header("4) Daten-Exploration")

# ---- robuste Hilfsfunktionen (gecacht) ----
@st.cache_data(show_spinner=False)
def _prep_meta(_df: pd.DataFrame) -> pd.DataFrame:
    df = _df.copy()
    for c in ["track_name","artist_name","artist_genres"]:
        if c not in df.columns: df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    for c in ["release_year","explicit"]:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def _genre_counts(_df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
    s = (
        _df["artist_genres"]
        .str.lower()
        .str.replace(r"[;\|]", ",", regex=True)
        .str.split(",")
        .explode()
        .str.strip()
    )
    s = s[s.ne("")]
    vc = s.value_counts().head(top_n)
    return vc.rename_axis("genre").reset_index(name="count")

@st.cache_data(show_spinner=False)
def _year_counts(_df: pd.DataFrame) -> pd.DataFrame:
    ser = _df.loc[_df["release_year"]>0, "release_year"]
    vc = ser.value_counts().sort_index()
    return vc.rename_axis("year").reset_index(name="count")

@st.cache_data(show_spinner=False)
def _artist_counts(_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    vc = _df["artist_name"].value_counts().head(top_n)
    return vc.rename_axis("artist").reset_index(name="count")

@st.cache_data(show_spinner=False)
def _quality_table(_df: pd.DataFrame) -> pd.DataFrame:
    miss_pct = _df.isna().mean().round(4)*100
    # zähle "leere" Strings mit
    empty_pct = (_df.select_dtypes(include="object").eq("").mean().reindex(_df.columns).fillna(0)).round(4)*100
    out = pd.DataFrame({
        "missing_%": miss_pct,
        "empty_string_%": empty_pct
    }).reset_index(names="column")
    return out

dfE = _prep_meta(df_meta)

# ---- KPI-Zeile ----
c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.metric("Songs (unique)", f"{dfE.shape[0]:,}")
with c2:
    st.metric("Künstler (unique)", f"{dfE['artist_name'].nunique():,}")
with c3:
    yrs = dfE.loc[dfE["release_year"]>0,"release_year"]
    span = f"{int(yrs.min())}–{int(yrs.max())}" if len(yrs) else "–"
    st.metric("Jahr-Spanne", span)
with c4:
    p_exp = (dfE["explicit"].eq(1).mean()*100.0) if len(dfE) else 0.0
    st.metric("Explicit-Anteil", f"{p_exp:.1f}%")
with c5:
    gdf = _genre_counts(dfE, top_n=9999)
    st.metric("Genres (unique tokens)", f"{gdf['genre'].nunique():,}")

# ---- Tabs für Verteilungen ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Überblick", "Genres", "Jahr", "Künstler", "Datenqualität"]
)

import plotly.express as px

with tab1:
    colA, colB = st.columns(2)
    with colA:
        topg = _genre_counts(dfE, top_n=15)
        st.subheader("Top-Genres")
        fig = px.bar(topg, x="genre", y="count", title=None)
        fig.update_layout(xaxis_title=None, yaxis_title="Anzahl", height=420)
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.subheader("Explicit vs. Clean")
        counts = dfE["explicit"].map({0:"Clean",1:"Explicit"}).value_counts().rename_axis("label").reset_index(name="count")
        fig = px.pie(counts, names="label", values="count", hole=0.5)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Genre-Häufigkeiten")
    top_n = st.slider("Top-N Genres", 5, 50, 20, 1)
    topg = _genre_counts(dfE, top_n=top_n)
    fig = px.bar(topg, x="genre", y="count")
    fig.update_layout(xaxis_title=None, yaxis_title="Anzahl", height=520)
    fig.update_xaxes(tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Songs pro Jahr")
    yd = _year_counts(dfE)
    smooth = st.checkbox("Geglättete Linie", value=True, help="Rolling-Mittel über 3 Jahre")
    if smooth and len(yd) >= 3:
        yd["smooth"] = yd["count"].rolling(3, center=True, min_periods=1).mean()
        fig = px.line(yd, x="year", y="smooth")
    else:
        fig = px.line(yd, x="year", y="count")
    fig.update_layout(xaxis_title="Jahr", yaxis_title="Anzahl", height=520)
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Produktivste Künstler (Anzahl Songs im Datensatz)")
    top_n = st.slider("Top-N Künstler", 5, 40, 20, 1, key="top_artists_n")
    ta = _artist_counts(dfE, top_n=top_n)
    fig = px.bar(ta, x="artist", y="count")
    fig.update_layout(xaxis_title=None, yaxis_title="Anzahl", height=520)
    fig.update_xaxes(tickangle=-35)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Datenqualität")
    qt = _quality_table(dfE)
    # zusätzliche Kennzahlen
    dups = dfE.duplicated(subset=["track_id"]).sum() if "track_id" in dfE.columns else 0
    colq1, colq2, colq3 = st.columns(3)
    with colq1: st.metric("Duplikate (track_id)", f"{dups:,}")
    with colq2: st.metric("Spalten", f"{dfE.shape[1]:,}")
    with colq3: st.metric("Zeilen", f"{dfE.shape[0]:,}")
    st.dataframe(qt, use_container_width=True, hide_index=True)


# ---------- Cluster-Visualisierung ----------
st.subheader("5) Cluster-Visualisierung")

@st.cache_resource(show_spinner=False)
def project_2d(_model_name: str, _X_norm: csr_matrix):
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    return svd.fit_transform(_X_norm)

has_kmeans = ("kmeans" in artifact) and ("cluster_kmeans" in df_meta.columns)
with st.spinner("Berechne 2D-Projektion …"):
    coords = project_2d(model_choice, X_norm)
vis_df = df_meta.copy()
vis_df["x"] = coords[:, 0]
vis_df["y"] = coords[:, 1]
vis_df["display"] = vis_df["track_name"].fillna("").astype(str) + " — " + vis_df["artist_name"].fillna("").astype(str)

fig = px.scatter(
    vis_df,
    x="x", y="y",
    color="cluster_kmeans" if has_kmeans else None,
    hover_name="display",
    opacity=0.7,
    title="2D-Embedding" + (" (K-Means-Cluster)" if has_kmeans else ""),
    height=600,
)
st.plotly_chart(fig, use_container_width=True)

# ======================= 3b) Cluster-Profil (nur wenn K-Means vorhanden) =======================
if ("kmeans" in artifact) and ("cluster_kmeans" in df_meta.columns):
    st.markdown("### 🔍 Cluster-Profil")

    @st.cache_data(show_spinner=False)
    def _compute_cluster_profiles(_df_meta: pd.DataFrame):
        """Aggregiert Kennzahlen je Cluster (Top-Genres/-Artists, Größe, Explicit, Median-Jahr)."""
        profiles = {}
        if "cluster_kmeans" not in _df_meta.columns:
            return profiles

        # Vorbereiten
        dfm = _df_meta.copy()
        dfm["release_year"] = pd.to_numeric(dfm["release_year"], errors="coerce")
        dfm["explicit"] = pd.to_numeric(dfm["explicit"], errors="coerce").fillna(0).astype(int)

        for c, g in dfm.groupby("cluster_kmeans"):
            # KPIs
            size = int(len(g))
            explicit_share = float((g["explicit"] == 1).mean())
            year_median = int(np.nanmedian(g["release_year"])) if size else 0

            # Genres aufdröseln
            genres = (
                g["artist_genres"].fillna("").str.lower()
                 .str.replace(r"[;\|]", ",", regex=True)
                 .str.split(",").explode().str.strip()
            )
            genres = genres[genres.ne("")]
            top_genres = genres.value_counts().head(12)

            # Artists
            top_artists = g["artist_name"].astype(str).value_counts().head(10)

            profiles[int(c)] = {
                "size": size,
                "explicit_share": explicit_share,
                "year_median": year_median,
                "top_genres": top_genres,
                "top_artists": top_artists,
            }
        return profiles

    # Versuche den Query-Cluster (mit den aktuellen Gewichten)
    q_cluster = None
    try:
        _src = df_meta.loc[int(choice)] if "choice" in locals() else df_meta.iloc[0]
        _q = build_query_vector(artifact, _src, gw=gw, aw=aw, tw=tw, nw=nw)
        q_cluster = int(artifact["kmeans"].predict(_q)[0])
    except Exception:
        q_cluster = None

    clusters = sorted(df_meta["cluster_kmeans"].dropna().astype(int).unique().tolist())
    label_query = "Query-Cluster" + (f" ({q_cluster})" if q_cluster is not None else "")
    options = ([label_query] if q_cluster is not None else []) + clusters

    sel = st.selectbox("Cluster wählen", options=options, index=0)
    if isinstance(sel, str):
        c_id = q_cluster if q_cluster is not None else clusters[0]
    else:
        c_id = int(sel)

    # Profile berechnen
    profiles = _compute_cluster_profiles(df_meta)
    if c_id not in profiles:
        st.info("Kein Profil für den gewählten Cluster gefunden.")
    else:
        p = profiles[c_id]

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Cluster-ID", c_id)
        k2.metric("Größe", f"{p['size']:,}".replace(",", "."))
        k3.metric("Explicit-Anteil", f"{p['explicit_share']*100:.1f}%")
        k4.metric("Median-Jahr", f"{p['year_median']}")

        # Visuelle Profile (Genres, Artists, Jahresverteilung)
        import plotly.express as px

        colA, colB = st.columns([1.2, 1])
        with colA:
            if len(p["top_genres"]) > 0:
                gbar = px.bar(
                    p["top_genres"].sort_values(ascending=True).to_frame("Anzahl"),
                    orientation="h",
                    labels={"index": "Genre", "Anzahl": "Anzahl"},
                    title="Top-Genres im Cluster",
                    height=420,
                )
                gbar.update_layout(margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(gbar, use_container_width=True)
            else:
                st.caption("Keine Genres erkennbar.")

        with colB:
            top_art_df = (
                p["top_artists"]
                .to_frame(name="Anzahl")
                .reset_index()
                .rename(columns={"index": "Artist"})
            )
            st.markdown("**Top-Artists**")
            st.dataframe(top_art_df, use_container_width=True, hide_index=True)

        # Jahresverteilung (direkt aus df_meta gefiltert, damit wir nicht doppelt tokenisieren)
        gmask = df_meta["cluster_kmeans"].eq(c_id)
        yrs = pd.to_numeric(df_meta.loc[gmask, "release_year"], errors="coerce").dropna().astype(int)
        if len(yrs):
            h = px.histogram(yrs, nbins=30, labels={"value": "Erscheinungsjahr", "count": "Anzahl"},
                             title="Jahresverteilung im Cluster", height=320)
            h.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(h, use_container_width=True)
        else:
            st.caption("Keine Jahresangaben für diesen Cluster.")
else:
    st.info("Dieses Modell enthält kein K-Means – Cluster-Profil ist nur mit K-Means verfügbar.")
# ======================= Ende Cluster-Profil =======================



# =========================
# THESEN
# =========================

st.header("📚 Thesen & Nachweise")

# ---------- Helfer: sichere Auswahl eines Query-Songs ----------
def _pick_query_index(df: pd.DataFrame) -> int:
    # Nimm zuletzt genutzte Query, sonst erste sichtbare, sonst 0
    if st.session_state.get("last_query_idx") is not None:
        qid = int(st.session_state["last_query_idx"])
        if 0 <= qid < len(df):
            return qid
    return int(df.index[0])

# ---------- THESE 1: Cluster-Verteilung (K-Means) ----------
with st.expander("These 1: K-Means-Cluster zeigen sinnvolle Verteilung über den gesamten Datensatz", expanded=True):
    has_kmeans = ("kmeans" in artifact) and ("cluster_kmeans" in df_meta.columns)
    if not has_kmeans:
        st.info("Für diese These bitte ein K-Means-Modell (train_app_model_variants.py mit --model kmeans) verwenden. "
                "Aktuelles Artefakt enthält keine K-Means-Labels.")
    else:
        # Saubere, kollisionsfreie Aggregation ohne reset_index-Problem
        dfc = df_meta[["cluster_kmeans"]].copy()
        dfc = dfc.rename(columns={"cluster_kmeans": "cluster"})
        grp = dfc.groupby("cluster", as_index=False).size()  # -> Spalten: cluster, size
        total = int(grp["size"].sum())
        grp["share"] = grp["size"] / total

        # Sortierung für schönere Darstellung
        grp = grp.sort_values("cluster").reset_index(drop=True)

        st.write("**Verteilung aller Songs auf die Cluster** (Anzahl + Anteil):")
        st.dataframe(
            grp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "cluster": st.column_config.NumberColumn("Cluster"),
                "size": st.column_config.NumberColumn("Anzahl"),
                "share": st.column_config.NumberColumn("Anteil", format="%.2f"),
            },
        )

        fig = px.bar(
            grp,
            x="cluster", y="size",
            hover_data={"share": ":.2f", "cluster": True, "size": True},
            title="Cluster-Histogramm (K-Means)",
            height=400,
        )
        # dünne Referenzlinie: gleicher Erwartungswert pro Cluster
        avg = total / max(1, grp["cluster"].nunique())
        fig.add_hline(y=avg, line_dash="dash", line_color="red", annotation_text="Ø pro Cluster", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Interpretation: Sind die Balken sehr unausgeglichen, sammelt ein Cluster überproportional viele Songs — "
            "das kann ein sehr „grobes“/häufiges Merkmal (z. B. Mainstream-Pop) oder unbalancierte Features widerspiegeln."
        )

# ---------- THESE 2: Diversität — „max. 1 Track pro Artist“ erhöht die Anzahl einzigartiger Artists in Top-K ----------
with st.expander("These 2: Diversität — „max. 1 Track pro Artist“ steigert die Anzahl verschiedener Artists in den Top-K", expanded=True):
    # Wir nehmen als Query entweder die zuletzt gewählte oder den ersten Song.
    q_idx = _pick_query_index(df_meta)
    src_row_t2 = df_meta.loc[q_idx]

    # Hilfsfunktion: baue Query-Vektor, hole Kandidaten, rechne Metrik
    def _neighbors_for_query(q_row: pd.Series, top_extra: int = 200):
        q_vec = build_query_vector(artifact, q_row, gw=gw, aw=aw, tw=tw, nw=nw)
        k_req = min(int(top_extra), len(df_meta))
        distances, indices = nn.kneighbors(q_vec, n_neighbors=k_req)
        idxs = [i for i in indices[0].tolist() if i != int(q_row.name)]
        # Scores (= 1−Distanz), nützlich für spätere Filter — hier optional
        i2pos = {i: p for p, i in enumerate(indices[0].tolist())}
        scores_all = {i: float(1.0 - distances[0][i2pos[i]]) for i in idxs}
        return idxs, scores_all

    try:
        base_neighbors, _scores = _neighbors_for_query(src_row_t2, top_extra=1000)
    except Exception as e:
        st.error(f"Konnte Nachbarn für die Query nicht bestimmen: {e}")
        base_neighbors = []

    if not base_neighbors:
        st.warning("Keine Nachbarn gefunden. Wähle einen anderen Song oder reduziere Filter.")
    else:
        ks = [5, 10, 15, 20, 25, 30, 40, 50]

        # ohne Diversität
        uniq_no_div = []
        # mit Diversität
        uniq_div = []

        for K in ks:
            cand = base_neighbors[:K]
            # ohne Beschränkung: Anzahl verschiedener Artists
            artists = df_meta.iloc[cand]["artist_name"].fillna("").astype(str).str.lower()
            uniq_no_div.append(int(artists.nunique()))

            # mit „max. 1 pro Artist“
            seen = set()
            pick = []
            for i in base_neighbors:  # aus dem größeren Pool, bis wir K erreicht haben
                a = str(df_meta.loc[i, "artist_name"]).strip().lower()
                if a not in seen:
                    pick.append(i)
                    seen.add(a)
                if len(pick) >= K:
                    break
            uniq_div.append(int(df_meta.iloc[pick]["artist_name"].str.lower().nunique()))

        # DataFrame für Plot
        t2 = pd.DataFrame({
            "K": ks,
            "unique_artists_no_div": uniq_no_div,
            "unique_artists_div": uniq_div,
        })

        # Fallback: falls eine Reihe konstant/leer wäre, setze Minimum 1
        t2["unique_artists_no_div"] = t2["unique_artists_no_div"].clip(lower=1)
        t2["unique_artists_div"] = t2["unique_artists_div"].clip(lower=1)

        # Plot: zwei Linien – baseline (ohne) & dunkelblau (mit Diversität)
        fig2 = px.line(
            t2.melt(id_vars="K", var_name="var", value_name="unique_artists"),
            x="K", y="unique_artists", color="var",
            title=f"Diversität über Top-K (Query: {src_row_t2['track_name']} — {src_row_t2['artist_name']})",
            markers=True, height=420,
        )
        # Schöne Legenden-Texte
        fig2.for_each_trace(lambda tr: tr.update(
            name="ohne Diversität" if tr.name == "unique_artists_no_div" else "mit Diversität (max. 1/Artist)"
        ))

        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Lesart: Für jedes K (Anzahl Empfehlungen) vergleichen wir, wie viele **verschiedene Artists** in den Top-K landen – "
            "einmal ohne Einschränkung, einmal mit „max. 1 Track pro Artist“. Die dunkelblaue Linie (mit Diversität) "
            "liegt typischerweise **gleichauf oder oberhalb** der Basislinie, d. h. die Empfehlungen werden breiter gestreut."
        )

# ---------- THESE 3: Songs im selben K-Means-Cluster liegen im 2D-Embedding näher beieinander ----------
with st.expander("These 3: Songs im selben K-Means-Cluster liegen im 2D-Embedding dichter beisammen", expanded=True):
    # Wir nutzen die gleiche 2D-Projektion wie oben (UMAP/TSNE/ PCA – je nach deiner project_2d-Implementierung)
    try:
        coords_t3 = project_2d(model_choice, X_norm)  # nutzt Cache/Fix aus früherem Schritt
    except Exception as e:
        st.error(f"2D-Projektion fehlgeschlagen: {e}")
        coords_t3 = None

    has_kmeans = ("kmeans" in artifact) and ("cluster_kmeans" in df_meta.columns)

    if coords_t3 is None:
        st.stop()

    plot_df = df_meta.copy()
    plot_df["x"] = coords_t3[:, 0]
    plot_df["y"] = coords_t3[:, 1]
    plot_df["display"] = plot_df["track_name"].fillna("").astype(str) + " — " + plot_df["artist_name"].fillna("").astype(str)

    if not has_kmeans:
        st.info("Für die Cluster-Hervorhebung bitte ein K-Means-Artefakt wählen (model='kmeans'). "
                "Wir zeigen trotzdem den 2D-Scatter.")
        fig = px.scatter(
            plot_df, x="x", y="y",
            opacity=0.7, hover_name="display",
            title="2D-Embedding der Songs (ohne Cluster-Färbung)",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Cluster-Auswahl
        clusters = sorted(plot_df["cluster_kmeans"].dropna().unique().tolist())
        c_sel = st.selectbox("Cluster auswählen", clusters, index=0)

        # Punkte markieren
        plot_df["is_sel"] = plot_df["cluster_kmeans"].eq(c_sel)

        # Erst Basis-Scatter (alle Punkte leicht transparent, nach Cluster gefärbt)
        fig = px.scatter(
            plot_df,
            x="x", y="y",
            color="cluster_kmeans",
            opacity=0.35,
            hover_name="display",
            hover_data={"track_name": True, "artist_name": True, "release_year": True, "explicit": True, "x": False, "y": False},
            title=f"2D-Embedding der Songs — Cluster {c_sel} hervorgehoben",
            height=620,
        )

        # Dann gewählten Cluster groß & klar drüberzeichnen
        sel_df = plot_df[plot_df["is_sel"]]
        if not sel_df.empty:
            fig.add_scatter(
                x=sel_df["x"], y=sel_df["y"],
                mode="markers",
                marker=dict(size=11, line=dict(width=1, color="red")),
                name=f"Cluster {c_sel} (hervorgehoben)",
                hovertext=sel_df["display"],
            )

        st.plotly_chart(fig, use_container_width=True)

        # ===== Kompaktheits-Nachweis (einfach & schnell) =====
        # Idee: Mittelwert der euklidischen Distanz im 2D-Plot
        #       - innerhalb des gewählten Clusters (intra)
        #       - von Cluster-Punkten zu Punkten anderer Cluster (extra)
        # Für Geschwindigkeit ggf. samplen.
        import numpy as np

        max_intra = 1500  # max Punkte für intra
        max_extra = 3000  # max Punkte für extra
        sel_idx = sel_df.index.to_numpy()
        other_idx = plot_df.index[~plot_df["is_sel"]].to_numpy()

        # Sampling
        if len(sel_idx) > max_intra:
            sel_idx = np.random.choice(sel_idx, size=max_intra, replace=False)
        if len(other_idx) > max_extra:
            other_idx = np.random.choice(other_idx, size=max_extra, replace=False)

        A = plot_df.loc[sel_idx, ["x", "y"]].to_numpy(dtype=float)
        B = plot_df.loc[other_idx, ["x", "y"]].to_numpy(dtype=float) if len(other_idx) else np.empty((0, 2))

        # Intra-Cluster: mittlere Distanz zum Cluster-Zentrum
        if len(A) >= 2:
            centroid_A = A.mean(axis=0, keepdims=True)
            intra_d = np.sqrt(((A - centroid_A) ** 2).sum(axis=1)).mean()
        else:
            intra_d = float("nan")

        # Extra-Cluster: mittlere Distanz von A-Punkten zum globalen Datensatz-Zentrum außerhalb
        if len(B) >= 1:
            centroid_B = B.mean(axis=0, keepdims=True)
            extra_d = np.sqrt(((A - centroid_B) ** 2).sum(axis=1)).mean()
        else:
            extra_d = float("nan")

        # Anzeige als kleine Kennzahlenkarte + Mini-Balkenplot
        c1, c2 = st.columns(2)
        with c1:
            st.metric(label="∅ Intra-Cluster-Distanz", value=f"{intra_d:0.3f}")
        with c2:
            st.metric(label="∅ Distanz zu anderen Clustern", value=f"{extra_d:0.3f}")

        bars = pd.DataFrame({
            "Kategorie": ["intra (Cluster selbst)", "extra (andere Cluster)"],
            "Distanz": [intra_d, extra_d],
        })
        fig_bar = px.bar(
            bars, x="Kategorie", y="Distanz", text="Distanz",
            title="Kompaktheitsvergleich im 2D-Embedding",
            height=380,
        )
        fig_bar.update_traces(texttemplate='%{text:.3f}', textposition='outside', cliponaxis=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.caption(
            "Interpretation: Ist die **Intra-Cluster-Distanz** deutlich kleiner als die Distanz zu anderen Clustern, "
            "liegen die Songs eines Clusters im 2D-Embedding **dichter** zusammen. Das stützt die These, "
            "dass K-Means thematisch ähnliche Songs bündelt."
        )


# ---------- THESE 4: Explizite Songs im Zeitverlauf ----------
with st.expander("These 4: Explizite Songs sind in den letzten Jahren häufiger geworden", expanded=True):
    if "release_year" not in df_meta.columns or "explicit" not in df_meta.columns:
        st.warning("Diese These benötigt die Spalten 'release_year' und 'explicit'.")
    else:
        df_year = (
            df_meta.groupby("release_year")["explicit"]
            .mean()
            .reset_index(name="explicit_share")
        )

        import plotly.express as px
        fig_t4 = px.line(
            df_year,
            x="release_year", y="explicit_share",
            markers=True,
            title="Anteil expliziter Songs pro Jahr",
        )
        fig_t4.update_yaxes(tickformat=".0%")

        st.plotly_chart(fig_t4, use_container_width=True)

        st.caption(
            "Interpretation: Der Anteil expliziter Songs wird über die Jahre hinweg berechnet "
            "(1 = explicit, 0 = clean). Ein steigender Trend bedeutet, dass **moderne Songs öfter "
            "als 'explicit' markiert sind** als ältere Veröffentlichungen."
        )

# ---------- THESE 5: Manche Künstler dominieren ----------
with st.expander("These 5: Manche Künstler dominieren den Datensatz", expanded=True):
    if "artist_name" not in df_meta.columns:
        st.warning("Diese These benötigt die Spalte 'artist_name'.")
    else:
        top_artists = (
            df_meta["artist_name"].fillna("").str.strip().value_counts().head(20).reset_index()
        )
        top_artists.columns = ["artist_name", "count"]

        import plotly.express as px
        fig_t5 = px.bar(
            top_artists.sort_values("count"),
            x="count", y="artist_name",
            orientation="h",
            title="Top 20 Künstler nach Anzahl Songs im Datensatz",
        )

        st.plotly_chart(fig_t5, use_container_width=True)

        st.caption(
            "Interpretation: Hier sieht man die **Künstler mit den meisten Songs im Datensatz**. "
            "Einige dominieren stark – das kann die Empfehlungen verzerren, weil diese Namen häufiger auftauchen."
        )






