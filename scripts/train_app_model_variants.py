# scripts/train_app_model_variants.py
from __future__ import annotations
import argparse, os, joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack, issparse, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


REQUIRED_META = ["track_id", "track_name", "artist_name",
                 "artist_genres", "release_year", "explicit"]


# ---------- Utilities ----------
def ensure_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in REQUIRED_META:
        if c not in df.columns:
            df[c] = 0 if c in ("release_year", "explicit") else ""
    df["track_name"]    = df["track_name"].astype(str)
    df["artist_name"]   = df["artist_name"].astype(str)
    df["artist_genres"] = df["artist_genres"].fillna("").astype(str)
    df["release_year"]  = pd.to_numeric(df["release_year"], errors="coerce").fillna(0).astype(int)
    df["explicit"]      = pd.to_numeric(df["explicit"], errors="coerce").fillna(0).astype(int)
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    return df


def to_csr(x):
    if issparse(x): return x.tocsr()
    return csr_matrix(np.asarray(x))


def fit_text_vectors(
    df: pd.DataFrame,
    min_df_genre: int,
    min_df_title: int,
    max_feat_genre: int,
    max_feat_artist: int,
    max_feat_title: int,
):
    # Genres: Delimiter bereinigen
    g_docs = (
        df["artist_genres"]
        .str.lower().str.replace(r"[;\|]", ",", regex=True)
        .str.replace(",", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip().tolist()
    )
    vec_genre = TfidfVectorizer(
        min_df=min_df_genre,
        max_features=max_feat_genre or None,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        lowercase=True,
    )
    Xg = vec_genre.fit_transform(g_docs)

    # Artist-Name
    a_docs = df["artist_name"].str.lower().str.strip().tolist()
    vec_artist = TfidfVectorizer(
        min_df=1,
        max_features=max_feat_artist or None,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 1),
        lowercase=True,
    )
    Xa = vec_artist.fit_transform(a_docs)

    # Titel
    t_docs = df["track_name"].str.lower().str.strip().tolist()
    vec_title = TfidfVectorizer(
        min_df=min_df_title,
        max_features=max_feat_title or None,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        lowercase=True,
    )
    Xt = vec_title.fit_transform(t_docs)

    return vec_genre, Xg, vec_artist, Xa, vec_title, Xt


def build_numeric(df: pd.DataFrame):
    num = df[["release_year", "explicit"]].to_numpy(dtype=float)
    scaler = StandardScaler()
    Xn = scaler.fit_transform(num)
    return scaler, to_csr(Xn)


def build_content_matrix(
    df: pd.DataFrame,
    w_genre: float, w_artist: float, w_title: float, w_num: float,
    min_df_genre: int, min_df_title: int,
    max_feat_genre: int, max_feat_artist: int, max_feat_title: int,
):
    vec_genre, Xg, vec_artist, Xa, vec_title, Xt = fit_text_vectors(
        df, min_df_genre, min_df_title,
        max_feat_genre, max_feat_artist, max_feat_title
    )
    scaler, Xn = build_numeric(df)

    blocks = []
    if Xg.shape[1] > 0: blocks.append(Xg.multiply(float(w_genre)))
    if Xa.shape[1] > 0: blocks.append(Xa.multiply(float(w_artist)))
    if Xt.shape[1] > 0: blocks.append(Xt.multiply(float(w_title)))
    blocks.append(Xn.multiply(float(w_num)))

    X = hstack(blocks).tocsr()
    return {
        "X": X,
        "vec_genre": vec_genre, "Xg": Xg,
        "vec_artist": vec_artist, "Xa": Xa,
        "vec_title": vec_title, "Xt": Xt,
        "scaler": scaler, "Xn": Xn,
    }


def fit_nn(X_norm, n_neighbors: int):
    k = max(1, min(int(n_neighbors), X_norm.shape[0]))
    return NearestNeighbors(n_neighbors=k, metric="cosine").fit(X_norm)


# ---------- RandomForest Leaf-Embedding (Helper, falls du es separat brauchst) ----------
def rf_leaf_embedding(rf: RandomForestClassifier, X_dense: np.ndarray) -> csr_matrix:
    """One-Hot über Leaves aller Bäume, L2-normalisiert."""
    leaves = rf.apply(X_dense)  # (n_samples, n_trees)
    n_samples, n_trees = leaves.shape
    cols = []
    offset = 0
    for t in range(n_trees):
        uniq, inv = np.unique(leaves[:, t], return_inverse=True)
        cols.append(inv + offset)
        offset += uniq.size
    all_cols = np.stack(cols, axis=1)
    rows = np.repeat(np.arange(n_samples), n_trees)
    cols_flat = all_cols.ravel()
    data = np.ones_like(cols_flat, dtype=np.float32)
    mat = coo_matrix((data, (rows, cols_flat)), shape=(n_samples, int(offset))).tocsr()
    mat = normalize(mat, norm="l2", axis=1, copy=False)
    return mat


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Trainiert App-kompatible Modelle (neighbors | kmeans | rf-artist).")
    ap.add_argument("--csv", required=True, help="Input CSV, z. B. data/raw/mix_master.csv")
    ap.add_argument("--out", required=True, help="Output .joblib (z. B. models/kmeans.joblib)")
    ap.add_argument("--model", choices=["neighbors", "kmeans", "rf-artist"], default="neighbors")

    # gemeinsame Hyperparameter
    ap.add_argument("--neighbors", type=int, default=50)
    ap.add_argument("--w-genre",  type=float, default=1.0)
    ap.add_argument("--w-artist", type=float, default=1.0)
    ap.add_argument("--w-title",  type=float, default=0.2)
    ap.add_argument("--w-num",    type=float, default=0.6)
    ap.add_argument("--min-df-genre", type=int, default=2)
    ap.add_argument("--min-df-title", type=int, default=2)
    ap.add_argument("--max-feat-genre",  type=int, default=50000)
    ap.add_argument("--max-feat-artist", type=int, default=10000)
    ap.add_argument("--max-feat-title",  type=int, default=60000)

    # kmeans
    ap.add_argument("--k", type=int, default=100, help="Anzahl Cluster für KMeans")

    # random forest
    ap.add_argument("--rf-n-estimators", type=int, default=200)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)

    args = ap.parse_args()

    print(f"[LOAD] {args.csv}")
    df = pd.read_csv(args.csv)
    df = ensure_meta(df)
    print(f"[DATA] {len(df)} rows | unique tracks={df['track_id'].nunique()}")

    print("[FEATURES] Baue Content-Matrix …")
    feats = build_content_matrix(
        df,
        args.w_genre, args.w_artist, args.w_title, args.w_num,
        args.min_df_genre, args.min_df_title,
        args.max_feat_genre, args.max_feat_artist, args.max_feat_title,
    )
    X = feats["X"]
    X_norm_base = normalize(X, norm="l2", axis=1, copy=False)  # Standard-Content-Space

    artifact = {
        "df_meta": df[REQUIRED_META].copy(),
        "vec_genre": feats["vec_genre"],
        "vec_artist": feats["vec_artist"],
        "vec_title": feats["vec_title"],
        "scaler": feats["scaler"],
        "config": {
            "genre_weight":  float(args.w_genre),
            "artist_weight": float(args.w_artist),
            "title_weight":  float(args.w_title),
            "numeric_weight":float(args.w_num),
            "model_type": args.model,
        },
        "version": 2,
    }

    if args.model == "neighbors":
        print("[MODEL] neighbors → KNN auf Content-Embedding")
        X_norm = X_norm_base
        nn = fit_nn(X_norm, args.neighbors)
        artifact.update({"X_norm": X_norm, "nn": nn})

    elif args.model == "kmeans":
        print(f"[MODEL] kmeans(k={args.k}) → Cluster-Labels + KNN auf Content-Embedding")
        km = KMeans(n_clusters=int(args.k), n_init="auto", random_state=42)
        labels = km.fit_predict(X_norm_base)
        artifact["df_meta"]["cluster_kmeans"] = labels
        artifact["kmeans"] = km
        X_norm = X_norm_base
        nn = fit_nn(X_norm, args.neighbors)
        artifact.update({"X_norm": X_norm, "nn": nn})

    elif args.model == "rf-artist":
        print("[MODEL] rf-artist → RandomForest auf Artist als Ziel + Leaf-Embedding + KNN")
        # RF auf Content-Space trainieren
        X_dense = feats["X"].astype(np.float32).toarray()
        le = LabelEncoder()
        y = le.fit_transform(df["artist_name"].astype(str))

        rf = RandomForestClassifier(
            n_estimators=int(args.rf_n_estimators),
            max_depth=args.rf_max_depth,
            min_samples_leaf=int(args.rf_min_samples_leaf),
            n_jobs=-1,
            random_state=42,
        ).fit(X_dense, y)

        # ---- Leaf-Embedding + Mapping sichern (für App-Query im gleichen Raum) ----
        leaves = rf.apply(X_dense)  # (n_samples, n_trees)
        n_samples, n_trees = leaves.shape
        leaf_maps = []     # Liste[dict{leaf_id:int -> global_col:int}]
        offset = 0

        for t in range(n_trees):
            uniq = np.unique(leaves[:, t])
            mapping = {int(leaf_id): int(offset + i) for i, leaf_id in enumerate(uniq)}
            leaf_maps.append(mapping)
            offset += len(uniq)

        rf_leaf_dim = int(offset)

        # Trainings-Leaf-Embedding aufbauen (sparse, L2-normalisiert)
        rows, cols, data = [], [], []
        for i in range(n_samples):
            for t in range(n_trees):
                leaf_id = int(leaves[i, t])
                col = leaf_maps[t][leaf_id]
                rows.append(i); cols.append(col); data.append(1.0)

        X_leaf = coo_matrix((data, (rows, cols)), shape=(n_samples, rf_leaf_dim)).tocsr()
        X_norm = normalize(X_leaf, norm="l2", axis=1, copy=False)
        nn = fit_nn(X_norm, args.neighbors)

        artifact.update({
            "X_norm": X_norm,
            "nn": nn,
            "rf": rf,
            "rf_label_encoder": le,
            "rf_leaf_maps": leaf_maps,   # << wichtig für die App
            "rf_leaf_dim": rf_leaf_dim,  # << wichtig für die App
        })

    else:
        raise ValueError(f"Unbekannter --model: {args.model}")

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(artifact, args.out)
    print(f"[OK] Artefakt gespeichert → {args.out}")


if __name__ == "__main__":
    main()

