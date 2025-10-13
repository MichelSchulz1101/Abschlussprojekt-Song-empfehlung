# 🎧 Song Recommender – README

Ein schlanker, in **Streamlit** gebauter Content-Based Recommender für Songs.  
Wir nutzen **Text** (Artist, Titel, Genres) und **Numeric** (Jahr, explicit) und erstellen daraus Vektoren, K-Means-Cluster und KNN-Nachbarschaften. Modelle sind als **Joblib-Artefakte** speicher- und wiederladbar.

---

## ⚙️ Voraussetzungen

- **Python 3.10 – 3.13** (empfohlen: 3.10/3.11)
- Git, Internetzugang (nur falls du optional Spotify-Daten ziehst)

---

## 🚀 Quickstart (nur App starten)

> Du hast bereits `models/` (mit `.joblib`) und `models/registry.json` im Repo?  
> Dann reichen 5 Befehle:

### GIT LFS Einrichten

**1. Installation:**

- MacOS

```bash
brew install git-lfs
git lfs install
```

- Windows

```cmd
git lfs install
```

```bash
# 1) Repo holen
git clone <DEIN_REPO_URL>
git lfs pull
cd <DEIN_PROJEKT_ORDNER>

# 2) Virtuelle Umgebung
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

# 3) Pip aktualisieren
python -m pip install --upgrade pip

# 4) Abhängigkeiten
pip install -r requirements.txt

# 5) App starten
streamlit run app/recommender_app.py
```

Öffne den Link aus dem Terminal (Standard: `http://localhost:8501`).

---

## 🧭 App – Kurzanleitung

**Seitenleiste (links):**
- **Modell wählen**: aus `models/registry.json`.
- **Gewichtungs-Slider**: Genre/Artist/Titel/Numerisch (überschreibt die Modell-Defaults).
- **Empfehlungseinstellungen**: Top-K, Score-Schwelle, max. 1 Track pro Artist.

**Hauptbereich:**
1. **Suchen & Filtern**  
   Suche nach Titel/Künstler/Genres, filtere nach Jahr, Explicit und (optional) Genres.
2. **Empfehlungen**  
   - Track auswählen → „Empfehlungen berechnen“  
   - Ergebnisliste mit Score & Spotify-Link  
   - Optional: **Inline-Player** (30s-Preview, wenn verfügbar)  
   - CSV-Download der Empfehlungen
3. **Cluster-Visualisierung**  
   2D-Projektion aller Songs; farbige Cluster, wenn das gewählte Modell K-Means enthält.
4. **(Optional) Modell-Bewertung**  
   Einfache Offline-Metriken (z. B. Recall@K / nDCG@K) als Orientierung.

---

## 📁 Projektstruktur (vereinfachte Ansicht)

```
.
├─ app/
│  └─ recommender_app.py          # Streamlit-App
├─ scripts/
│  ├─ train_app_model_variants.py # training neighbors/kmeans/rf-artist (app-kompatibel)
│  └─ (weitere Hilfsskripte)      
├─ data/
│  └─ raw/
│     └─ mix_master.csv           # Dein Master-CSV
├─ models/
│  ├─ neighbors_*.joblib
│  ├─ kmeans_*.joblib
│  ├─ rf_artist_*.joblib
│  └─ registry.json               # Modell-Index für die App
├─ requirements.txt
└─ README.md
```

---

## 🧪 Eigene Modelle trainieren (empfohlen)

### 1) Datensatz vorbereiten

CSV: `data/raw/mix_master.csv`  
**Erwartete Spalten:**
- `track_id`, `track_name`, `artist_name`, `artist_genres`, `release_year`, `explicit`

### 2) Training – App-kompatible Artefakte erzeugen

**KNN-Nachbarn:**
```bash
python -m scripts.train_app_model_variants --model neighbors --csv data/raw/mix_master.csv --out models/neighbors.joblib --neighbors 50
```

**K-Means:**
```bash
python -m scripts.train_app_model_variants --model kmeans --csv data/raw/mix_master.csv --out models/kmeans.joblib --k 100 --neighbors 50
```

**Random-Forest-Artist:**
```bash
python -m scripts.train_app_model_variants --model rf-artist --csv data/raw/mix_master.csv --out models/rf_artist.joblib --neighbors 50 --rf-n-estimators 300 --rf-max-depth 25 --rf-min-samples-leaf 2
```

### 3) Registry aktualisieren

`models/registry.json` Beispiel:
```json
{
  "neighbors_genre_artist_title": { "path": "models/neighbors.joblib" },
  "kmeans_100":                   { "path": "models/kmeans.joblib"    },
  "rf_artist_depth25":            { "path": "models/rf_artist.joblib" }
}
```

---

## 🧰 Nützliche Terminal-Kommandos

**Cache leeren:**
```bash
streamlit cache clear
```

**Andere Portnummer:**
```bash
streamlit run app/recommender_app.py --server.port 8502
```

---

## 📊 Visualisierung & Metriken

- **2D-Embedding-Plot:** Jeder Punkt = Song. Nähe = Ähnlichkeit.  
  K-Means = farbige Cluster.  
- **Metriken:**  
  - Recall@K = Anteil relevanter Songs in Top-K  
  - Precision@K = Anteil der Top-K, die relevant sind  
  - nDCG@K = bewertet auch die Reihenfolge  

---

Viel Spaß & gute Empfehlungen! 🎶
