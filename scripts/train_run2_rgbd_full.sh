#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

yolo task=detect mode=train \
  model=ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml \
  data=data_rgbd_pseudo.yaml \
  train=train_full.txt \
  epochs=100 imgsz=640 project=runs/exp --name run2_rgbd_full \
  > logs/run2_rgbd_full.log 2>&1

echo "[INFO] Run2 RGBD Full completed. Logs: logs/run2_rgbd_full.log"
