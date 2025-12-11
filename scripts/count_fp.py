#!/usr/bin/env python3
"""Count false positives on HN predictions (every detection is FP)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "runs" / "hn"
runs = ["run1_base_full", "run2_rgbd_full"]

for name in runs:
    label_dir = base / name / "labels"
    if not label_dir.exists():
        label_dir.mkdir(parents=True, exist_ok=True)
        print(f"{name}: labels dir missing, created -> {label_dir}")
    total = 0
    for f in label_dir.glob("*.txt"):
        with f.open() as fh:
            for _ in fh:
                total += 1
    print(f"{name}: FP={total}")
