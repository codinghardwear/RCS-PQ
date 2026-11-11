#!/usr/bin/env bash
set -euo pipefail

# Generic standalone PQ-only thread sweep.
# Required: INPUT=/path/to/dataset.bin
# Optional: OUT_ROOT=./results RUN_ID=... THREADS_LIST="1 10 20 40 80"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${RCSPQ_BIN:-$ROOT/build/rcspq}"
INPUT="${INPUT:?Set INPUT to a DiskANN-style float32 .bin dataset}"
RUN_ID="${RUN_ID:-rcspq_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/results/$RUN_ID}"
LOG_ROOT="${LOG_ROOT:-$OUT_ROOT/logs}"
THREADS_LIST="${THREADS_LIST:-1 10 20 40 80}"
SUBDIM="${SUBDIM:-16}"
CENTROIDS="${CENTROIDS:-256}"
TRAIN_RATIO="${TRAIN_RATIO:-0.0256}"
SAMPLE_SEED="${SAMPLE_SEED:-20261111}"
KMEANS_SEED="${KMEANS_SEED:-20261111}"
KMEANS_ITERS="${KMEANS_ITERS:-15}"
BATCH_SIZE="${BATCH_SIZE:-262144}"
METRIC="${METRIC:-l2}"

if [[ ! -x "$BIN" ]]; then
  echo "ERROR: executable not found or not executable: $BIN" >&2
  echo "Build first: cmake -S '$ROOT' -B '$ROOT/build' -DCMAKE_BUILD_TYPE=Release && cmake --build '$ROOT/build' -j" >&2
  exit 1
fi
if [[ ! -f "$INPUT" ]]; then
  echo "ERROR: input file not found: $INPUT" >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
if [[ -e "$OUT_ROOT/summary.csv" ]]; then
  echo "ERROR: summary already exists; choose a new RUN_ID/OUT_ROOT: $OUT_ROOT" >&2
  exit 1
fi
printf 'dataset,threads,subdim,centroids,train_ratio,status,log,output_dir\n' > "$OUT_ROOT/summary.csv"

for T in $THREADS_LIST; do
  OUT="$OUT_ROOT/thread_${T}"
  LOG="$LOG_ROOT/thread_${T}.log"
  echo "[$(date -u +%FT%TZ)] START thread=$T input=$INPUT" | tee "$LOG"
  if OMP_PLACES=cores OMP_PROC_BIND=spread "$BIN" \
      --input "$INPUT" --output "$OUT" \
      --subdim "$SUBDIM" --centroids "$CENTROIDS" \
      --train-ratio "$TRAIN_RATIO" --sample-seed "$SAMPLE_SEED" \
      --kmeans-seed "$KMEANS_SEED" --kmeans-iters "$KMEANS_ITERS" \
      --threads "$T" --batch-size "$BATCH_SIZE" --metric "$METRIC" \
      >> "$LOG" 2>&1; then
    printf '"%s",%s,%s,%s,%s,SUCCESS,"%s","%s"\n' \
      "$INPUT" "$T" "$SUBDIM" "$CENTROIDS" "$TRAIN_RATIO" "$LOG" "$OUT" >> "$OUT_ROOT/summary.csv"
    echo "[$(date -u +%FT%TZ)] SUCCESS thread=$T" | tee -a "$LOG"
  else
    printf '"%s",%s,%s,%s,%s,FAILED,"%s","%s"\n' \
      "$INPUT" "$T" "$SUBDIM" "$CENTROIDS" "$TRAIN_RATIO" "$LOG" "$OUT" >> "$OUT_ROOT/summary.csv"
    echo "[$(date -u +%FT%TZ)] FAILED thread=$T; see $LOG" >&2
    exit 1
  fi
done

echo "Completed. Summary: $OUT_ROOT/summary.csv"
