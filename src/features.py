from __future__ import annotations
import re
import json
from dataclasses import dataclass, asdict
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

@dataclass
class FeatureConfig:
    use_title: bool = False
    genre_weight: float = 1.0
    artist_weight: float = 1.0
    title_weight: float = 0.2
    numeric_weight: float = 0.6
    ngram_genre: Tuple[int, int] = (1, 2)
    ngram_artist: Tuple[int, int] = (1, 1)
    ngram_title: Tuple[int, int] = (1, 2)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

class FeatureBuilder:
    """
    Baut Sparse-Features aus artist_genres / artist_name / (optional) track_name
    + numerisch (release_year, explicit). Liefert normalisierte Matrix (Zeilenl2=1).
    """
    def __init__(self, cfg: FeatureConfig):
        self.cfg = cfg
        self.vec_genre: Optional[TfidfVectorizer] = None
        self.vec_artist: Optional[TfidfVectorizer] = None
        self.vec_title: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "track_id" not in df.columns:
            df["track_id"] = (df.get("track_name", "").astype(str) + " :: " + df.get("artist_name", "").astype(str)).map(_norm_text)
        if "track_name" not in df.columns:
            df["track_name"] = ""
        if "artist_name" not in df.columns:
            df["artist_name"] = ""
        if "artist_genres" not in df.columns:
            df["artist_genres"] = ""
        if "release_date" in df.columns:
            year = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        else:
            year = pd.Series([np.nan] * len(df))
        df["release_year"] = year.fillna(0).astype(int)
        if "explicit" not in df.columns:
            df["explicit"] = 0
        return df

    def fit_transform(self, df: pd.DataFrame) -> Tuple[csr_matrix, pd.DataFrame]:
        df = self._ensure_columns(df)
        df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)

        # Text
        self.vec_genre = TfidfVectorizer(stop_words="english", ngram_range=self.cfg.ngram_genre)
        Xg = self.vec_genre.fit_transform(df["artist_genres"].fillna("").astype(str))

        self.vec_artist = TfidfVectorizer(stop_words="english", ngram_range=self.cfg.ngram_artist)
        Xa = self.vec_artist.fit_transform(df["artist_name"].fillna("").astype(str))

        if self.cfg.use_title:
            self.vec_title = TfidfVectorizer(stop_words="english", ngram_range=self.cfg.ngram_title)
            Xt = self.vec_title.fit_transform(df["track_name"].fillna("").astype(str))
        else:
            self.vec_title = None
            Xt = None

        # Numeric
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        num = df[["release_year", "explicit"]].fillna(0).astype(float).values
        Xn = self.scaler.fit_transform(num)
        Xn = csr_matrix(Xn)

        blocks = [
            Xg.multiply(self.cfg.genre_weight),
            Xa.multiply(self.cfg.artist_weight),
        ]
        if Xt is not None:
            blocks.append(Xt.multiply(self.cfg.title_weight))
        blocks.append(Xn.multiply(self.cfg.numeric_weight))

        X = hstack(blocks).tocsr()

        # row-norm (cosine via dot)
        norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        X_norm = X.multiply(1.0 / norms[:, None]).tocsr()

        return X_norm, df

    def transform_query(self, row: pd.Series) -> csr_matrix:
        # Text
        Xg = self.vec_genre.transform([_safe_str(row["artist_genres"])]) if self.vec_genre else None
        Xa = self.vec_artist.transform([_safe_str(row["artist_name"])]) if self.vec_artist else None
        Xt = self.vec_title.transform([_safe_str(row["track_name"])]) if self.vec_title else None

        # Numeric
        num = np.array([[float(row.get("release_year", 0)), float(row.get("explicit", 0))]])
        Xn = csr_matrix(self.scaler.transform(num)) if self.scaler else csr_matrix(num)

        blocks = []
        if Xg is not None: blocks.append(Xg.multiply(self.cfg.genre_weight))
        if Xa is not None: blocks.append(Xa.multiply(self.cfg.artist_weight))
        if Xt is not None: blocks.append(Xt.multiply(self.cfg.title_weight))
        blocks.append(Xn.multiply(self.cfg.numeric_weight))

        q = hstack(blocks).tocsr()
        qn = np.sqrt(q.multiply(q).sum())
        return q if qn == 0 else q.multiply(1.0 / qn)

