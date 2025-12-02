#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

declare -A RUNS=(
  [run1_base_full]="runs/exp/run1_base_full/weights/best.pt"
  [run2_rgbd_full]="runs/exp/run2_rgbd_full/weights/best.pt"
)

for name in "${!RUNS[@]}"; do
  weight="${RUNS[$name]}"
  log_file="logs/pred_${name}_hn.log"
  echo "[INFO] Predicting HN set with ${name} -> ${log_file}"
  yolo task=detect mode=predict \
    model="${weight}" \
    data=data_test_hn.yaml \
    save_txt=true save_conf=true \
    project=runs/hn --name "${name}" \
    > "${log_file}" 2>&1
done

echo "[INFO] HN predictions done. Outputs in runs/hn/<run>/labels"
