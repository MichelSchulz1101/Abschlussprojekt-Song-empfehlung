# src/features.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.sparse import hstack, csr_matrix

def build_feature_matrix(df: pd.DataFrame) -> Tuple[csr_matrix, dict]:
    # Text-Features: Genres
    gen_series = (df.get("artist_genres") or pd.Series([""]*len(df))).fillna("").astype(str)
    tfidf = TfidfVectorizer(tokenizer=lambda s: [t.strip() for t in s.split(",") if t.strip()],
                            lowercase=True, min_df=2, max_features=1500)
    X_gen = tfidf.fit_transform(gen_series)

    # numerisch
    num_cols = []
    if "popularity" in df.columns: num_cols.append("popularity")
    if "duration_ms" in df.columns: num_cols.append("duration_ms")
    X_num = csr_matrix((len(df), 0))
    if num_cols:
        scaler = MinMaxScaler()
        Xn = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
        X_num = csr_matrix(Xn)

    # Jahr (Bucket oder OneHot)
    year = df.get("release_year")
    X_year = csr_matrix((len(df), 0))
    if year is not None:
        y = year.fillna("unknown").astype(str).to_numpy().reshape(-1,1)
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        X_year = ohe.fit_transform(y)

    X = hstack([X_gen, X_num, X_year], format="csr")
    meta = {
        "tfidf": tfidf,
        "num_cols": num_cols,
        "scalers": None,  # (MinMaxScaler intern nicht nötig zurückzugeben)
        "ohe_year": ("release_year" in df.columns),
        "X_shape": X.shape,
    }
    return X, meta
