#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

yolo task=detect mode=train \
  model=ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml \
  data=data_rgbd_pseudo.yaml \
  train=train_50.txt \
  epochs=100 imgsz=640 project=runs/exp --name run3_rgbd_50 \
  > logs/run3_rgbd_50.log 2>&1

echo "[INFO] Run3 RGBD 50% completed. Logs: logs/run3_rgbd_50.log"
