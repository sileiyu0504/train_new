#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

yolo task=detect mode=train \
  model=ultralytics/ultralytics/cfg/models/11/yolo11.yaml \
  data=data_rgb.yaml \
  epochs=100 imgsz=640 project=runs/exp name=run1_base_full \
  | tee logs/run1_base_full.log

echo "[INFO] Run1 Baseline completed. Logs: logs/run1_base_full.log"
