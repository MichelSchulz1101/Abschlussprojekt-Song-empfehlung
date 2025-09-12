
import argparse, os, re, json, joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def _to_year(s):
    if pd.isna(s): return np.nan
    s = str(s)
    m = re.match(r"(\d{4})", s)
    return int(m.group(1)) if m else np.nan

def primary_genre(s):
    if pd.isna(s) or not str(s).strip():
        return "unknown"
    # Split on comma/semicolon/pipe -> erster Token
    s = str(s).lower()
    s = re.sub(r"[;|]", ",", s)
    first = s.split(",")[0].strip()
    return first if first else "unknown"

def load_df(path):
    df = pd.read_csv(path)
    for col in ["artist_genres","artist_name","track_name","popularity","explicit","release_date"]:
        if col not in df.columns: df[col] = np.nan
    df["release_year"] = df["release_date"].apply(_to_year)
    df["explicit_num"] = df["explicit"].astype(float) if str(df["explicit"].dtype) != "bool" else df["explicit"].astype(int)
    df["artist_genres"] = df["artist_genres"].fillna("")
    df["artist_name"]   = df["artist_name"].fillna("")
    df["track_name"]    = df["track_name"].fillna("")
    df["popularity"]    = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["release_year"]  = pd.to_numeric(df["release_year"], errors="coerce").fillna(df["release_year"].median())
    df["explicit_num"]  = df["explicit_num"].fillna(0)

    df["y_primary_genre"] = df["artist_genres"].apply(primary_genre)
    # seltene Klassen bündeln
    vc = df["y_primary_genre"].value_counts()
    rare = set(vc[vc < 20].index)
    df["y_primary_genre"] = df["y_primary_genre"].apply(lambda g: "other" if g in rare else g)
    return df

def build_pipeline():
    text_genre = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-zA-ZäöüÄÖÜß\-]+", min_df=2, max_df=0.95, ngram_range=(1,2)
    )
    text_title = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-zA-Z0-9äöüÄÖÜß\-]+", min_df=3, max_df=0.95, ngram_range=(1,2)
    )
    cat_artist = OneHotEncoder(handle_unknown="ignore", min_frequency=20)
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

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
    )
    pipe = Pipeline([("pre", pre), ("rf", rf)])
    return pipe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Pfad zu mix_master.csv")
    ap.add_argument("--out", default="models/", help="Output-Ordner")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_df(args.csv)
    X = df[["artist_genres","track_name","artist_name","popularity","release_year","explicit_num"]]
    y = df["y_primary_genre"].astype(str)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)

    report = classification_report(yte, yhat, zero_division=0, output_dict=True)
    with open(os.path.join(args.out, "rf_genre_noaudio_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    joblib.dump(pipe, os.path.join(args.out, "rf_genre_noaudio.joblib"))

    # kurze Konsolen-Zusammenfassung
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    print("[OK] RF gespeichert →", os.path.join(args.out, "rf_genre_noaudio.joblib"))
    print(f"[METRIC] macro-F1={macro_f1:.3f} | weighted-F1={weighted_f1:.3f}")
    print("[INFO] Vollständiger Report → models/rf_genre_noaudio_report.json")

if __name__ == "__main__":
    main()
