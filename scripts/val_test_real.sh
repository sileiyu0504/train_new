#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

declare -A RUNS=(
  [run1_base_full]="runs/exp/run1_base_full/weights/best.pt"
  [run2_rgbd_full]="runs/exp/run2_rgbd_full/weights/best.pt"
  [run3_rgbd_50]="runs/exp/run3_rgbd_50/weights/best.pt"
  [run4_rgbd_20]="runs/exp/run4_rgbd_20/weights/best.pt"
)

for name in "${!RUNS[@]}"; do
  weight="${RUNS[$name]}"
  log_file="logs/test_${name}_real.log"
  echo "[INFO] Validating ${name} on real RGBD test set -> ${log_file}"
  yolo task=detect mode=val \
    model="${weight}" \
    data=data_test_real.yaml \
    save_json=true \
    > "${log_file}" 2>&1
done

echo "[INFO] Validation complete. See logs/test_*_real.log"
