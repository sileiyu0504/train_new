#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

yolo task=detect mode=train \
  model=ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml \
  data=data_rgbd_pseudo.yaml \
  fraction=0.5 \
  epochs=100 imgsz=640 project=runs/exp name=run3_rgbd_50 \
  | tee logs/run3_rgbd_50.log

echo "[INFO] Run3 RGBD 50% completed. Logs: logs/run3_rgbd_50.log"
