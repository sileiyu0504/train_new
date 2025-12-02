#!/usr/bin/env bash
set -euo pipefail

# Generate train_full.txt, train_50.txt, train_20.txt for YOLO_RGB_carton
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

FULL_LIST="${ROOT}/train_full.txt"
LIST_50="${ROOT}/train_50.txt"
LIST_20="${ROOT}/train_20.txt"

echo "[INFO] Building full train list at ${FULL_LIST}"
find datasets/YOLO_RGB_carton/images/train2017 \
  -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \
  | sort > "${FULL_LIST}"

echo "[INFO] Shuffling and slicing with seed=42"
python - <<'PY'
import random, pathlib
root = pathlib.Path(".")
full = root / "train_full.txt"
lines = full.read_text().splitlines()
random.Random(42).shuffle(lines)
n = len(lines)
cut50 = int(n * 0.5)
cut20 = int(n * 0.2)
(root / "train_50.txt").write_text("\n".join(lines[:cut50]) + "\n")
(root / "train_20.txt").write_text("\n".join(lines[:cut20]) + "\n")
print(f"Total={n}, 50%={cut50}, 20%={cut20}")
PY

echo "[INFO] Done. Lists: ${FULL_LIST}, ${LIST_50}, ${LIST_20}"
