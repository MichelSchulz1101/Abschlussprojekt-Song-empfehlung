from __future__ import annotations
import argparse
import os, json, time

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.features import FeatureConfig, FeatureBuilder

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

def build_and_save_model(
    df: pd.DataFrame,
    name: str,
    cfg: FeatureConfig,
    out_dir: str = "models",
    n_neighbors: int = 50,
):
    fb = FeatureBuilder(cfg)
    X_norm, df2 = fb.fit_transform(df)

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X_norm)

    artifact = {
        "name": name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg.__dict__,
        "df_meta": df2[["track_id", "track_name", "artist_name", "artist_genres", "release_year", "explicit"]].copy(),
        "nn": nn,
        "X_norm": X_norm,
        "vec_genre": fb.vec_genre,
        "vec_artist": fb.vec_artist,
        "vec_title": fb.vec_title,  # kann None sein
        "scaler": fb.scaler,
        "version": "1.0",
    }

    ensure_dirs(out_dir)
    path = os.path.join(out_dir, f"{name}.joblib")
    joblib.dump(artifact, path, compress=3)
    return path

def update_registry(model_path: str, model_name: str, registry_path: str = "models/registry.json"):
    reg = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    reg[model_name] = {"path": model_path}
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Pfad zu mix_master.csv (oder beliebige CSV mit Spalten)")
    ap.add_argument("--out", default="models", help="Modell-Ausgabeordner")
    ap.add_argument("--neighbors", type=int, default=50, help="k für NearestNeighbors (Indexgröße)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # === Beispiel: 3 Varianten ===
    variants = {
        "cb_genre_artist": FeatureConfig(use_title=False, genre_weight=1.5, artist_weight=1.0, numeric_weight=0.5),
        "cb_genre_artist_title": FeatureConfig(use_title=True, genre_weight=1.2, artist_weight=1.0, title_weight=0.3, numeric_weight=0.6),
        "cb_artist_heavy": FeatureConfig(use_title=False, genre_weight=0.8, artist_weight=2.0, numeric_weight=0.4),
    }

    for name, cfg in variants.items():
        p = build_and_save_model(df, name=name, cfg=cfg, out_dir=args.out, n_neighbors=args.neighbors)
        update_registry(p, name, registry_path=os.path.join(args.out, "registry.json"))
        print(f"[OK] Modell gespeichert: {p}")

if __name__ == "__main__":
    main()
