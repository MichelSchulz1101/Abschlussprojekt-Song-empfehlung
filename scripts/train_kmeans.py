# scripts/train_kmeans.py
import argparse, os, re, json, joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def _to_year(s):
    if pd.isna(s): return np.nan
    s = str(s)
    m = re.match(r"(\d{4})", s)
    return int(m.group(1)) if m else np.nan

def load_df(path):
    df = pd.read_csv(path)
    # Spalten absichern
    for col in ["artist_genres","artist_name","track_name","popularity","explicit","release_date"]:
        if col not in df.columns: df[col] = np.nan
    # year, explicit_num
    df["release_year"] = df["release_date"].apply(_to_year)
    df["explicit_num"] = df["explicit"].astype(float) if str(df["explicit"].dtype) != "bool" else df["explicit"].astype(int)
    # fehlende füllen
    df["artist_genres"] = df["artist_genres"].fillna("")
    df["artist_name"]   = df["artist_name"].fillna("")
    df["track_name"]    = df["track_name"].fillna("")
    df["popularity"]    = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["release_year"]  = pd.to_numeric(df["release_year"], errors="coerce").fillna(df["release_year"].median())
    df["explicit_num"]  = df["explicit_num"].fillna(0)
    return df

def build_pipeline(n_clusters: int):
    text_genre = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-zA-ZäöüÄÖÜß\-]+", min_df=2, max_df=0.9, ngram_range=(1,2)
    )
    text_title = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-zA-Z0-9äöüÄÖÜß\-]+", min_df=3, max_df=0.9, ngram_range=(1,2)
    )
    cat_artist = OneHotEncoder(handle_unknown="ignore", min_frequency=10)

    num_cols = ["popularity","release_year","explicit_num"]
    pre = ColumnTransformer(
        transformers=[
            ("genres", text_genre, "artist_genres"),
            ("title",  text_title,  "track_name"),
            ("artist", cat_artist,  ["artist_name"]),
            ("num",    StandardScaler(), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    pipe = Pipeline([("pre", pre), ("kmeans", km)])
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Pfad zu mix_master.csv")
    ap.add_argument("--out", default="models/", help="Output-Ordner")
    ap.add_argument("--clusters", type=int, default=12, help="Anzahl Cluster")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_df(args.csv)
    pipe = build_pipeline(args.clusters)
    X = df[["artist_genres","track_name","artist_name","popularity","release_year","explicit_num"]]
    pipe.fit(X)

    # Silhouette
    # Achtung: silhouette braucht Dense; bei großen Daten evtl. Sample:
    try:
        # Transform (nur Vorverarbeitung) für Silhouette
        pre = pipe.named_steps["pre"]
        Xtrans = pre.fit_transform(X)
        if hasattr(Xtrans, "toarray"):  # sparse
            # Samplen für Performance
            n = min(8000, Xtrans.shape[0])
            idx = np.random.default_rng(42).choice(Xtrans.shape[0], size=n, replace=False)
            Xt_small = Xtrans[idx].toarray()
            labels_small = pipe.named_steps["kmeans"].fit_predict(Xt_small)
            sil = float(silhouette_score(Xt_small, labels_small))
        else:
            labels = pipe.named_steps["kmeans"].labels_
            sil = float(silhouette_score(Xtrans, labels))
    except Exception as e:
        sil = None

    joblib.dump(pipe, os.path.join(args.out, "kmeans_noaudio.joblib"))
    meta = {"clusters": args.clusters, "silhouette": sil}
    with open(os.path.join(args.out, "kmeans_noaudio.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("[OK] KMeans gespeichert →", os.path.join(args.out, "kmeans_noaudio.joblib"))
    if sil is not None:
        print(f"[METRIC] Silhouette ≈ {sil:.3f}")
    else:
        print("[WARN] Silhouette konnte nicht berechnet werden (ggf. zu groß/sparse).")

if __name__ == "__main__":
    main()
