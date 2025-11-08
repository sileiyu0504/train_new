#!/usr/bin/env python3
"""
用于 RGB + 深度 YOLO11 流程的数据集整理辅助脚本。

脚本会把 `unique` 文件夹中的 RGB 图像与对应的深度图和 YOLO 标签配对，
再按照标准 YOLO 目录结构（`images/*`、`labels/*`、`depth/*`）重新组织，
同时在 RGB 图像上以黑色边框绘制标注结果，便于快速验证框是否合理。
"""

from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

try:
    from PIL import Image, ImageDraw
except ImportError as exc:  # pragma: no cover - 依赖检查
    raise SystemExit(
        "需要安装 Pillow 才能绘制验证图，可运行 `pip install pillow`。"
    ) from exc


@dataclass(frozen=True)
class Sample:
    """保存同一场景的三种同步数据。"""

    stem: str
    rgb_path: Path
    depth_path: Path
    label_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pair RGB, depth, and YOLO label files, copy them into an output "
            "dataset folder, and create black-box verification renders."
        )
    )
    parser.add_argument(
        "--rgb-dir",
        type=Path,
        default=Path("unique"),
        help="Directory that stores the curated RGB images (default: unique).",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=Path("output_depth_images"),
        help="Directory that stores the matching depth maps (default: output_depth_images).",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("annotation"),
        help="Directory that stores YOLO txt labels (default: annotation).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared_dataset"),
        help="Root folder for the reorganized dataset (default: prepared_dataset).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples to place into the val split (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling before the train/val split.",
    )
    parser.add_argument(
        "--verification-folder-name",
        default="verification",
        help="Folder (created under the output root) that will hold annotated renders.",
    )
    return parser.parse_args()


def collect_samples(rgb_dir: Path, depth_dir: Path, label_dir: Path) -> Tuple[List[Sample], List[str]]:
    samples: List[Sample] = []
    diagnostics: List[str] = []

    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    if not depth_dir.exists():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    for rgb_path in sorted(rgb_dir.glob("*")):
        if not rgb_path.is_file():
            continue
        if rgb_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        stem = rgb_path.stem
        label_path = label_dir / f"{stem}.txt"
        depth_filename = rgb_path.name.replace("_color_", "_depth_", 1)
        depth_path = depth_dir / depth_filename

        missing_parts = []
        if not label_path.exists():
            missing_parts.append("label")
        if not depth_path.exists():
            missing_parts.append("depth")

        if missing_parts:
            diagnostics.append(
                f"Skipping {rgb_path.name}: missing {', '.join(missing_parts)}"
            )
            continue

        samples.append(
            Sample(
                stem=stem,
                rgb_path=rgb_path,
                depth_path=depth_path,
                label_path=label_path,
            )
        )

    return samples, diagnostics


def split_samples(
    samples: Sequence[Sample], val_ratio: float, seed: int
) -> Tuple[List[Sample], List[Sample]]:
    if not samples:
        return [], []

    ratio = min(max(val_ratio, 0.0), 0.9)
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * ratio)) if ratio > 0 else 0
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:] or val_samples[:1]

    # 确保在存在样本时两个划分都不为空。
    if not train_samples:
        train_samples, val_samples = val_samples, train_samples

    return train_samples, val_samples


def ensure_structure(root: Path, verification_folder_name: str) -> Tuple[dict, dict]:
    layout = {
        "train": {
            "images": root / "images" / "train",
            "labels": root / "labels" / "train",
            "depth": root / "depth" / "train",
        },
        "val": {
            "images": root / "images" / "val",
            "labels": root / "labels" / "val",
            "depth": root / "depth" / "val",
        },
    }
    verification_dirs = {
        split: root / verification_folder_name / split for split in layout.keys()
    }

    for split_paths in layout.values():
        for path in split_paths.values():
            path.mkdir(parents=True, exist_ok=True)

    for path in verification_dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return layout, verification_dirs


def copy_assets(sample: Sample, split: str, layout: dict) -> None:
    target = layout[split]
    shutil.copy2(sample.rgb_path, target["images"] / sample.rgb_path.name)
    shutil.copy2(sample.label_path, target["labels"] / sample.label_path.name)
    shutil.copy2(sample.depth_path, target["depth"] / sample.depth_path.name)


def read_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes: List[Tuple[int, float, float, float, float]] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x, y, w, h = parts
            boxes.append((int(cls), float(x), float(y), float(w), float(h)))
    return boxes


def create_verification_render(
    sample: Sample, verification_dir: Path, outline_color: str = "black"
) -> None:
    boxes = read_boxes(sample.label_path)

    with Image.open(sample.rgb_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        draw = ImageDraw.Draw(image)
        for _, x_c, y_c, bw, bh in boxes:
            x_min = (x_c - bw / 2) * width
            y_min = (y_c - bh / 2) * height
            x_max = (x_c + bw / 2) * width
            y_max = (y_c + bh / 2) * height
            draw.rectangle(
                [(x_min, y_min), (x_max, y_max)],
                outline=outline_color,
                width=2,
            )

        output_path = verification_dir / sample.rgb_path.name
        image.save(output_path)


def summarize(total: int, diagnostics: Iterable[str]) -> None:
    print(f"Prepared {total} paired samples.")
    skipped = list(diagnostics)
    if skipped:
        print("Skipped items:")
        for item in skipped:
            print(f"  - {item}")


def main() -> None:
    args = parse_args()
    samples, diagnostics = collect_samples(args.rgb_dir, args.depth_dir, args.label_dir)
    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    layout, verification_dirs = ensure_structure(args.output_root, args.verification_folder_name)

    for sample in train_samples:
        copy_assets(sample, "train", layout)
        create_verification_render(sample, verification_dirs["train"])

    for sample in val_samples:
        copy_assets(sample, "val", layout)
        create_verification_render(sample, verification_dirs["val"])

    summarize(len(samples), diagnostics)
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Dataset root: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
