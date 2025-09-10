#!/bin/zsh
set -e

PAUSE=300      # Sekunden warten nach jedem Artist-Block
MAX_ALBUMS=1   # pro Artist nur 1 Album (konservativ starten)
MASTER="data/raw/mix_master.csv"

for PART in data/parts/*.txt; do
  BASE=$(basename "$PART" .txt)
  OUT="data/raw/run_albums_${BASE}.csv"

  [ -f "$OUT" ] && { echo "[SKIP] $OUT existiert"; continue; }

  echo "== Alben: $PART → $OUT (max-albums=$MAX_ALBUMS) =="

  if ! python -m scripts.fetch_artists_to_csv \
        --from-file "$PART" \
        --max-albums $MAX_ALBUMS \
        --out "$OUT" 2> >(tee /tmp/_albums_err.log >&2); then
    echo "[WARN] Fehler beim Fetch. Überspringe $PART."
    rm -f "$OUT"
    continue
  fi

  if grep -qiE "rate/request limit|429" /tmp/_albums_err.log; then
    echo "[HALT] Rate-Limit erkannt. Pause einlegen, dann Skript erneut starten."
    break
  fi

  python - "$OUT" "$MASTER" <<'PY'
import os, sys, pandas as pd
part, master = sys.argv[1], sys.argv[2]
dfp = pd.read_csv(part)
df  = pd.concat([pd.read_csv(master), dfp], ignore_index=True) if os.path.exists(master) else dfp
if "track_id" in df.columns: df = df.drop_duplicates(subset=["track_id"])
df.to_csv(master, index=False)
print("[MASTER]", df["track_id"].nunique() if "track_id" in df.columns else len(df), "unique")
PY

  echo "[SLEEP] ${PAUSE}s …"; sleep $PAUSE
done
