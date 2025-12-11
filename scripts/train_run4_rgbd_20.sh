#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

yolo task=detect mode=train \
  model=ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml \
  data=data_rgbd_pseudo.yaml \
  fraction=0.2 \
  epochs=100 imgsz=640 project=runs/exp name=run4_rgbd_20 \
  | tee logs/run4_rgbd_20.log

echo "[INFO] Run4 RGBD 20% completed. Logs: logs/run4_rgbd_20.log"
