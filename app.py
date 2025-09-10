# app.py
import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from src.features import build_feature_matrix

st.set_page_config(page_title="Song-Empfehlungen (ohne Audio-Features)", layout="wide")
st.title("🎧 Song-Empfehlungen (Genres + Meta)")

@st.cache_resource
def load_data_and_model(csv_path: str):
    df = pd.read_csv(csv_path)
    if "release_date" in df.columns and "release_year" not in df.columns:
        df["release_year"] = df["release_date"].astype(str).str[:4]
    X, meta = build_feature_matrix(df)
    nn = NearestNeighbors(metric="cosine", algorithm="brute").fit(X)
    return df, X, nn, meta

csv_path = st.sidebar.text_input("Pfad zur CSV", "data/raw/mix_part1.csv")
if not csv_path:
    st.stop()

df, X, nn, meta = load_data_and_model(csv_path)

name_col = "track_name" if "track_name" in df.columns else df.columns[0]
artist_col = "artist_name" if "artist_name" in df.columns else None

def label_row(row):
    a = f" – {row[artist_col]}" if artist_col and pd.notna(row.get(artist_col)) else ""
    return f"{row[name_col]}{a}"

df["label"] = df.apply(label_row, axis=1)
choice = st.sidebar.selectbox("Song auswählen", options=df["label"].tolist())

k = st.sidebar.slider("Anzahl Empfehlungen", 5, 50, 20)

row = df[df["label"] == choice].iloc[0]
idx = row.name

dist, ind = nn.kneighbors(X[idx:idx+1], n_neighbors=min(k+1, len(df)))
neighbors = ind[0][1:]
scores = 1 - dist[0][1:]

recs = df.iloc[neighbors].copy()
recs["score"] = scores
show_cols = [c for c in [name_col, artist_col, "score", "artist_genres", "popularity", "release_year", "duration_ms"] if c in recs.columns]
st.subheader(f"Ausgewählt: {row[name_col]}{(' – ' + row[artist_col]) if artist_col else ''}")
st.dataframe(recs[show_cols].sort_values("score", ascending=False).reset_index(drop=True))
