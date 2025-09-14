# app/model_lab.py
import os, re, json, joblib
import numpy as np
import pandas as pd
import streamlit as st

from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- Utilities ----------
st.set_page_config(page_title="Model Lab", layout="wide")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Minimum-Spalten absichern
    for col in ["artist_genres","artist_name","track_name","popularity","explicit","release_date"]:
        if col not in df.columns: df[col] = np.nan
    # release_year + explicit_num bauen
    def _to_year(s):
        if pd.isna(s): return np.nan
        m = re.match(r"(\d{4})", str(s))
        return int(m.group(1)) if m else np.nan

    df["release_year"] = df["release_date"].apply(_to_year)
    # explicit kann bool/str sein → nach 0/1
    if df["explicit"].dtype == "bool":
        df["explicit_num"] = df["explicit"].astype(int)
    else:
        df["explicit_num"] = pd.to_numeric(df["explicit"], errors="coerce").fillna(0).astype(int)

    # Füllen
    df["artist_genres"] = df["artist_genres"].fillna("")
    df["artist_name"]   = df["artist_name"].fillna("")
    df["track_name"]    = df["track_name"].fillna("")
    df["popularity"]    = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    df["release_year"]  = pd.to_numeric(df["release_year"], errors="coerce").fillna(df["release_year"].median())

    return df

def build_preprocessor(min_df_genre=2, min_df_title=3, artist_min_freq=10):
    text_genre = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-zA-ZäöüÄÖÜß\\-]+", min_df=min_df_genre, max_df=0.95, ngram_range=(1,2)
    )
    text_title = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-zA-Z0-9äöüÄÖÜß\\-]+", min_df=min_df_title, max_df=0.95, ngram_range=(1,2)
    )
    cat_artist = OneHotEncoder(handle_unknown="ignore", min_frequency=artist_min_freq)

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
    return pre

def ensure_primary_genre(df: pd.DataFrame) -> pd.Series:
    def primary_genre(s):
        if pd.isna(s) or not str(s).strip():
            return "unknown"
        s = str(s).lower()
        s = re.sub(r"[;|]", ",", s)
        first = s.split(",")[0].strip()
        return first if first else "unknown"

    y = df["artist_genres"].apply(primary_genre)
    vc = y.value_counts()
    rare = set(vc[vc < 20].index)
    y = y.apply(lambda g: "other" if g in rare else g)
    return y.astype(str)

def save_model(pipe: Pipeline, name: str, meta: Optional[dict] = None):
    path = os.path.join(MODELS_DIR, name + ".joblib")
    joblib.dump(pipe, path)
    if meta is not None:
        with open(os.path.join(MODELS_DIR, name + ".json"), "w") as f:
            json.dump(meta, f, indent=2)
    return path

def list_saved_models():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    return sorted(files)

def load_model(filename: str) -> Pipeline:
    return joblib.load(os.path.join(MODELS_DIR, filename))

# ---------- Sidebar: Daten & globale Optionen ----------
st.sidebar.header("Datenquelle")
default_csv = "data/raw/mix_master.csv"
csv_path = st.sidebar.text_input("Pfad zur CSV", value=default_csv)
uploaded = st.sidebar.file_uploader("…oder CSV hochladen", type=["csv"])

if uploaded is not None:
    # temporär speichern, damit Cache funktioniert
    tmp_path = os.path.join("data", "tmp_uploaded.csv")
    os.makedirs("data", exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    csv_path = tmp_path

if not os.path.exists(csv_path):
    st.warning(f"CSV nicht gefunden: {csv_path}")
    st.stop()

df = load_csv(csv_path)
st.sidebar.success(f"Geladen: {len(df):,} Zeilen")

st.sidebar.header("Vorverarbeitung")
min_df_genre = st.sidebar.slider("min_df (Genres-TF-IDF)", 1, 10, 2)
min_df_title = st.sidebar.slider("min_df (Titel-TF-IDF)", 1, 10, 3)
artist_min_freq = st.sidebar.slider("min_frequency (Artist OneHot)", 1, 50, 10)

# ---------- Layout ----------
st.title("Model Lab – Modelle trainieren & vergleichen (ohne Audio-Features)")
st.caption("Interaktive Modellwahl, Hyperparameter-Tuning, Training, Metriken & Speichern.")

tabs = st.tabs(["📊 Daten-Check", "🧩 KMeans (Clustering)", "🌳 RandomForest (Genre-Classifier)", "💾 Modelle laden"])

# ====== Tab 1: Daten-Check ======
with tabs[0]:
    st.subheader("Daten-Überblick")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Artists", f"{df['artist_name'].nunique():,}")
    c3.metric("Tracks", f"{df['track_name'].nunique():,}")
    c4.metric("Genres (raw)", f"{(df['artist_genres'].str.len()>0).sum():,}")

    st.write(df.head(20))

    st.markdown("**Fehlende Werte (Top 10 Spalten):**")
    st.write(df.isna().sum().sort_values(ascending=False).head(10))

# ====== Tab 2: KMeans ======
with tabs[1]:
    st.subheader("KMeans – Clustering von Songs")
    st.markdown("Nutze Text (Genres/Titel), Artist (OneHot) und numerische Features für Cluster.")

    # Hyperparameter
    clusters = st.slider("Anzahl Cluster (k)", 4, 40, 12, step=1)
    n_init = st.selectbox("n_init", options=["auto", 10, 20, 50], index=0)
    random_state = st.number_input("random_state", value=42, step=1)

    colA, colB = st.columns(2)
    with colA:
        train_btn = st.button("🚀 Trainieren", key="kmeans_train")
    with colB:
        save_name = st.text_input("Speichername", value=f"kmeans_noaudio_k{clusters}")

    if train_btn:
        with st.spinner("Trainiere KMeans…"):
            pre = build_preprocessor(min_df_genre=min_df_genre, min_df_title=min_df_title, artist_min_freq=artist_min_freq)
            km = KMeans(n_clusters=clusters, n_init=n_init, random_state=random_state)
            pipe = Pipeline([("pre", pre), ("kmeans", km)])

            X = df[["artist_genres","track_name","artist_name","popularity","release_year","explicit_num"]]
            pipe.fit(X)

            # Silhouette (Sample aus Performance-Gründen)
            sil = None
            try:
                Xt = pipe.named_steps["pre"].transform(X)
                if hasattr(Xt, "toarray"):
                    n = min(6000, Xt.shape[0])
                    idx = np.random.default_rng(42).choice(Xt.shape[0], size=n, replace=False)
                    Xt_small = Xt[idx].toarray()
                    labels_small = pipe.named_steps["kmeans"].fit_predict(Xt_small)
                    sil = float(silhouette_score(Xt_small, labels_small))
                else:
                    sil = float(silhouette_score(Xt, pipe.named_steps["kmeans"].labels_))
            except Exception as e:
                sil = None

        st.success("Training abgeschlossen.")
        if sil is not None:
            st.metric("Silhouette (Sample)", f"{sil:.3f}")
        else:
            st.info("Silhouette konnte nicht berechnet werden (zu groß/sparse).")

        if st.button("💾 Modell speichern", key="kmeans_save") and save_name.strip():
            meta = {
                "type": "kmeans_noaudio",
                "clusters": clusters,
                "n_init": n_init,
                "random_state": random_state,
                "pre": {"min_df_genre": min_df_genre, "min_df_title": min_df_title, "artist_min_freq": artist_min_freq},
                "silhouette_sample": sil,
                "csv_used": csv_path,
            }
            path = save_model(pipe, save_name.strip(), meta)
            st.success(f"Gespeichert → {path}")

# ====== Tab 3: RandomForest (Genre) ======
with tabs[2]:
    st.subheader("RandomForest – Primär-Genre vorhersagen (aus artist_genres abgeleitet)")
    st.markdown("Label = erster Genre-Token (seltene Klassen → 'other').")

    # Hyperparameter
    n_estimators = st.slider("n_estimators", 100, 1000, 300, step=50)
    max_depth = st.selectbox("max_depth", options=[None, 10, 20, 40, 80], index=0)
    min_samples_leaf = st.slider("min_samples_leaf", 1, 10, 2)
    test_size = st.slider("Testgröße", 0.1, 0.4, 0.2, step=0.05)

    colA, colB = st.columns(2)
    with colA:
        train_rf_btn = st.button("🚀 Trainieren", key="rf_train")
    with colB:
        rf_save_name = st.text_input("Speichername", value=f"rf_genre_noaudio_{n_estimators}")

    if train_rf_btn:
        with st.spinner("Trainiere RandomForest…"):
            y = ensure_primary_genre(df)
            X = df[["artist_genres","track_name","artist_name","popularity","release_year","explicit_num"]]
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

            pre = build_preprocessor(min_df_genre=min_df_genre, min_df_title=min_df_title, artist_min_freq=artist_min_freq)
            rf = RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=None if max_depth is None else int(max_depth),
                min_samples_leaf=int(min_samples_leaf),
                n_jobs=-1,
                random_state=42
            )
            pipe = Pipeline([("pre", pre), ("rf", rf)])
            pipe.fit(Xtr, ytr)
            yhat = pipe.predict(Xte)

            report = classification_report(yte, yhat, zero_division=0, output_dict=True)
            macro_f1 = report["macro avg"]["f1-score"]
            weighted_f1 = report["weighted avg"]["f1-score"]

        st.success("Training abgeschlossen.")
        c1, c2 = st.columns(2)
        c1.metric("macro-F1", f"{macro_f1:.3f}")
        c2.metric("weighted-F1", f"{weighted_f1:.3f}")

        st.markdown("**Klassifikationsreport**")
        st.json(report)

        if st.button("💾 Modell speichern", key="rf_save") and rf_save_name.strip():
            meta = {
                "type": "rf_genre_noaudio",
                "n_estimators": int(n_estimators),
                "max_depth": None if max_depth is None else int(max_depth),
                "min_samples_leaf": int(min_samples_leaf),
                "pre": {"min_df_genre": min_df_genre, "min_df_title": min_df_title, "artist_min_freq": artist_min_freq},
                "metrics": {"macro_f1": macro_f1, "weighted_f1": weighted_f1},
                "csv_used": csv_path,
                "test_size": test_size,
            }
            path = save_model(pipe, rf_save_name.strip(), meta)
            # optional: Report separat sichern
            with open(os.path.join(MODELS_DIR, rf_save_name.strip() + "_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            st.success(f"Gespeichert → {path}")

# ====== Tab 4: Modelle laden ======
with tabs[3]:
    st.subheader("Gespeicherte Modelle")
    files = list_saved_models()
    if not files:
        st.info("Noch keine Modelle gespeichert.")
    else:
        pick = st.selectbox("Modell auswählen (.joblib)", files)
        if st.button("📦 Laden"):
            pipe = load_model(pick)
            st.success(f"{pick} geladen.")
            st.write(pipe)
            meta_path = os.path.join(MODELS_DIR, pick.replace(".joblib", ".json"))
            if os.path.exists(meta_path):
                st.markdown("**Meta-Informationen**")
                with open(meta_path) as f:
                    st.json(json.load(f))
