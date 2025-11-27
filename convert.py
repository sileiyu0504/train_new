#!/usr/bin/env python3
"""
Dataset conversion and hard-negative construction helper.

功能概览：
1) 将 RGB_carton (COCO 格式纸箱数据集) 转换为 YOLO 格式数据集；
2) 将 RGB_window (VOC 格式窗户数据集，xml 里 name: window 单一类别) 转换为 YOLO 格式数据集；
3) 基于 RGB_window + COCO(纸箱 COCO 中包含的家电类) 构建 Hard Negative YOLO 数据集 YOLO_HN：
   - 6 个类别：["window", "tv", "laptop", "microwave", "oven", "refrigerator"]
   - 每类 200 张去重图片，总计 1200 张
   - 划分为 1000 张 train + 200 张 val
4) 统一输出格式，适配 YOLO 训练；全流程带 tqdm 进度条和步骤提示。
"""

from __future__ import annotations

import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm.auto import tqdm

# 这里假定你已经把本地 ultralytics 源码桥接好了：
# /mnt/e/Desktop/train_new/ultralytics/__init__.py 中导出了内层包
from ultralytics.data.converter import convert_coco

# ====================== 路径和配置 ======================

# 所有转换后的数据集输出到这里
DATASETS_ROOT = Path("datasets")

# 原始数据集路径：
# 1) RGB_carton: COCO 格式纸箱数据集
RGB_CARTON_SRC = Path("Dataset/RGB_carton")
# 2) RGB_window: VOC 格式窗户数据集（xml 中 name: window 单一类别）
RGB_WINDOW_SRC = Path("Dataset/RGB_window/dataset")

# 随机数生成器（用于采样、打乱，保证可复现）
RNG = random.Random(42)

# Hard Negative 数据集的类别定义（内部索引 0~5）
HN_CLASSES = ["window", "tv", "laptop", "microwave", "oven", "refrigerator"]

# 每个 HN 类别需要的图像数量（去重图片数）
IMAGES_PER_CLASS = 200

# 从 COCO(YOLO 类别编号) 到 HN 类别编号的映射：
# 注意：这里的 62/63/68/69/72 是 YOLO 使用的 COCO 类索引，而非原始 COCO category_id
# 参考 Ultralytics 的 coco.yaml：
#   62: tvmonitor
#   63: laptop
#   68: microwave
#   69: oven
#   72: refrigerator
COCO_TO_HN: Dict[int, int] = {
    62: HN_CLASSES.index("tv"),
    63: HN_CLASSES.index("laptop"),
    68: HN_CLASSES.index("microwave"),
    69: HN_CLASSES.index("oven"),
    72: HN_CLASSES.index("refrigerator"),
}


# ====================== 数据结构定义 ======================

@dataclass
class Sample:
    """
    表示一张 YOLO 图片 + 标签，可以被重新导出到新的数据集中。

    image_path: 图片路径
    label_lines: YOLO txt 中的每一行（"cls x y w h"）
    classes: 这张图中出现的类别集合（HN 内部类别 id）
    tag: 来源标签（例如 "window_train", "coco_train2017"）
    """

    image_path: Path
    label_lines: List[str]
    classes: set[int]
    tag: str

    @property
    def base_name(self) -> str:
        """根据来源 + 原图名，生成在新数据集中的唯一基名。"""
        return f"{self.tag}_{self.image_path.stem}"


# ====================== 通用小工具 ======================

def reset_dir(path: Path) -> None:
    """删除并重建目录。"""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# ====================== 步骤 1: 转换 RGB_carton (COCO -> YOLO) ======================

def convert_rgb_carton() -> Path:
    """
    将 COCO 格式的 RGB_carton 数据集转换为 YOLO 格式。

    输入目录结构假定为：
        Dataset/RGB_carton/
          annotations/  (COCO json)
          images/
            train2017/
            val2017/
            ...

    输出目录结构：
        datasets/YOLO_RGB_carton/
          images/train2017/
          images/val2017/
          labels/train2017/
          labels/val2017/
    """
    annotations = RGB_CARTON_SRC / "annotations"
    images_root = RGB_CARTON_SRC / "images"
    output = DATASETS_ROOT / "YOLO_RGB_carton"

    if not annotations.exists() or not images_root.exists():
        raise FileNotFoundError("RGB_carton 数据集结构不完整，请确认 annotations/ 和 images/ 是否存在。")

    # 清理输出目录
    if output.exists():
        shutil.rmtree(output)

    print("[1/3] COCO -> YOLO: 正在转换 RGB_carton (纸箱 COCO 数据集)...")

    # 使用 ultralytics 内置的 COCO 转 YOLO 工具
    convert_coco(labels_dir=str(annotations), save_dir=str(output),
                 use_segments=False, use_keypoints=False)

    # 将所有 images 子目录复制到 YOLO 输出目录下，保持子目录名称 (train2017/val2017)
    subsets = [p for p in images_root.iterdir() if p.is_dir()]
    for subset in tqdm(subsets, desc="[1/3] 复制 RGB_carton 图像子目录", unit="dir"):
        dst = output / "images" / subset.name
        if dst.exists():
            shutil.rmtree(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(subset, dst)

    print(f"[1/3] 完成: RGB_carton 已转换为 YOLO 格式 -> {output}")
    return output


# ====================== 步骤 2: 转换 RGB_window (VOC -> YOLO) ======================

def parse_voc_box(obj: ET.Element, width: int, height: int) -> str | None:
    """
    从 VOC 的 <object> 中解析出 1 个 YOLO bbox 行文本。
    注意：原始 RGB_window 数据集是单一类别，xml 里有 name: window。

    只保留 name == "window" 的框，映射为 YOLO 类别 id 0。
    """
    name = obj.findtext("name")
    if not name:
        return None

    bbox = obj.find("bndbox")
    if bbox is None:
        return None

    xmin = float(bbox.findtext("xmin", default="0"))
    ymin = float(bbox.findtext("ymin", default="0"))
    xmax = float(bbox.findtext("xmax", default="0"))
    ymax = float(bbox.findtext("ymax", default="0"))

    # 简单合法性检查
    if xmax <= xmin or ymax <= ymin or width <= 0 or height <= 0:
        return None

    # VOC -> YOLO 归一化坐标
    x_c = ((xmin + xmax) / 2.0) / width
    y_c = ((ymin + ymax) / 2.0) / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height

    # 只要 window 一类
    cname = name.strip().lower()
    if cname != "window":
        return None

    # 在 RGB_window 数据集中，window 类别固定为 0
    return f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}"


def convert_rgb_window() -> Path:
    """
    将 VOC 格式的 RGB_window 窗户数据集转换为 YOLO 格式。

    输入目录结构假定为：
        Dataset/RGB_window/dataset/
          JPEGImages/
          Annotations/
          ImageSets/Main/train.txt
          ImageSets/Main/val.txt

    输出目录结构：
        datasets/YOLO_RGB_window/
          images/train/
          images/val/
          labels/train/
          labels/val/
    """
    images_dir = RGB_WINDOW_SRC / "JPEGImages"
    ann_dir = RGB_WINDOW_SRC / "Annotations"
    sets_dir = RGB_WINDOW_SRC / "ImageSets" / "Main"

    if not images_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError("RGB_window 数据集结构不完整，请确认 JPEGImages/ 和 Annotations/ 是否存在。")

    output = DATASETS_ROOT / "YOLO_RGB_window"
    reset_dir(output)

    print("[2/3] VOC -> YOLO: 正在转换 RGB_window (窗户 VOC 数据集, xml 中 name: window)...")

    for split in ("train", "val"):
        ids_file = sets_dir / f"{split}.txt"
        if not ids_file.exists():
            raise FileNotFoundError(f"RGB_window 缺少 ImageSets/Main/{split}.txt")

        ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]

        img_out_dir = output / "images" / split
        lbl_out_dir = output / "labels" / split
        img_out_dir.mkdir(parents=True, exist_ok=True)
        lbl_out_dir.mkdir(parents=True, exist_ok=True)

        for img_id in tqdm(ids, desc=f"[2/3] 处理 RGB_window {split}", unit="img"):
            xml_path = ann_dir / f"{img_id}.xml"
            if not xml_path.exists():
                continue

            tree = ET.parse(xml_path)
            root = tree.getroot()

            size = root.find("size")
            width = int(size.findtext("width", default="0")) if size is not None else 0
            height = int(size.findtext("height", default="0")) if size is not None else 0

            # 为该图片收集所有 window 框
            lines: List[str] = []
            for obj in root.findall("object"):
                line = parse_voc_box(obj, width, height)
                if line:
                    lines.append(line)

            if not lines:
                # 该图片没有 window, 跳过
                continue

            # 复制对应图片
            src_img_jpg = images_dir / f"{img_id}.jpg"
            src_img_png = images_dir / f"{img_id}.png"
            if src_img_jpg.exists():
                src_img = src_img_jpg
            elif src_img_png.exists():
                src_img = src_img_png
            else:
                # 没有对应图片，跳过
                continue

            dst_img = img_out_dir / src_img.name
            dst_label = lbl_out_dir / f"{img_id}.txt"

            shutil.copy2(src_img, dst_img)
            dst_label.write_text("\n".join(lines) + "\n")

    print(f"[2/3] 完成: RGB_window 已转换为 YOLO 格式 -> {output}")
    return output


# ====================== 步骤 3: 构建 Hard Negative 数据集 ======================

def gather_window_samples(dataset_root: Path) -> List[Sample]:
    """
    从 YOLO_RGB_window 中收集 Sample 列表。

    YOLO_RGB_window 的标签中，类别 id 固定为 0 (window)。
    """
    samples: List[Sample] = []
    for split in ("train", "val"):
        label_dir = dataset_root / "labels" / split
        image_dir = dataset_root / "images" / split
        if not label_dir.exists():
            continue

        label_files = list(label_dir.glob("*.txt"))
        for label_path in tqdm(label_files, desc=f"[3/3] 收集 window 样本 {split}", unit="file"):
            lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
            if not lines:
                continue

            # 从每一行中取类别 id（理论上都为 0）
            cls_set = {int(float(line.split()[0])) for line in lines}
            if not cls_set:
                continue

            # 对应的图片（优先 jpg, 然后 png）
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                png = image_dir / f"{label_path.stem}.png"
                if png.exists():
                    image_path = png
                else:
                    continue

            samples.append(Sample(
                image_path=image_path,
                label_lines=lines,
                classes=cls_set,
                tag=f"window_{split}",
            ))
    return samples


def gather_coco_samples(dataset_root: Path) -> Dict[int, List[Sample]]:
    """
    从 YOLO_RGB_carton 中只收集指定 COCO 类别（tv/laptop/microwave/oven/refrigerator），
    并映射为 HN_CLASSES 中的类别索引。

    输入 YOLO_RGB_carton 结构：
        images/train2017, labels/train2017
        images/val2017,   labels/val2017

    返回:
        by_class: {hn_class_id: [Sample, ...], ...}
    """
    by_class: Dict[int, List[Sample]] = {cls_id: [] for cls_id in COCO_TO_HN.values()}

    for subset in ("train2017", "val2017"):
        label_dir = dataset_root / "labels" / subset
        image_dir = dataset_root / "images" / subset
        if not label_dir.exists():
            continue

        label_files = list(label_dir.glob("*.txt"))
        for label_path in tqdm(label_files, desc=f"[3/3] 收集 COCO HN 样本 {subset}", unit="file"):
            raw_lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
            keep: List[str] = []
            classes: set[int] = set()

            for line in raw_lines:
                parts = line.split()
                cls_yolo = int(float(parts[0]))  # YOLO 中的 COCO 类别索引
                if cls_yolo in COCO_TO_HN:
                    new_cls = COCO_TO_HN[cls_yolo]  # 映射到 HN 内部类别 id
                    keep.append(" ".join([str(new_cls)] + parts[1:]))
                    classes.add(new_cls)

            if not keep:
                continue

            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                alt = image_dir / f"{label_path.stem}.png"
                if alt.exists():
                    image_path = alt
                else:
                    continue

            sample = Sample(
                image_path=image_path,
                label_lines=keep,
                classes=classes.copy(),
                tag=f"coco_{subset}",
            )
            for cls in classes:
                by_class.setdefault(cls, []).append(sample)

    return by_class


def split_balanced(per_class: Dict[int, List[Sample]]) -> Tuple[List[Sample], List[Sample]]:
    """
    在每个 HN 类别中选取 IMAGES_PER_CLASS 张“不同图片”的样本，
    并将它们划分为 train/val:

    - 每类总共 IMAGES_PER_CLASS 张 (200)
    - 各类 train 数量合计为 1000 张
    - 各类剩余进入 val，总数为 200 张

    规则：
    - 同一 image_path 不会被重复用于多个类别（用 used_images 去重）。
    """
    # 各类列表复制一份，避免原地修改
    per_class = {k: list(v) for k, v in per_class.items()}

    used_images: set[Path] = set()
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []

    # 按每类样本数从少到多排序；样本少的优先分配，减少因去重导致的失败概率
    order = sorted(per_class.keys(), key=lambda cls: len(per_class[cls]))

    # 目标总 train 数量为 1000，按类别平均分配，余数从前几类补齐
    base_train = 1000 // len(HN_CLASSES)  # 1000 // 6 = 166
    extra_train = 1000 % len(HN_CLASSES)  # 1000 % 6 = 4
    train_target: Dict[int, int] = {
        cls: base_train + (1 if idx < extra_train else 0)
        for idx, cls in enumerate(order)
    }

    for cls in order:
        candidates = per_class.get(cls, [])
        if not candidates:
            raise RuntimeError(f"类别 id {cls} 没有任何样本。")

        RNG.shuffle(candidates)
        selected: List[Sample] = []

        # 为该类别挑选 IMAGES_PER_CLASS 张“不同图片”的样本
        for sample in candidates:
            if sample.image_path in used_images:
                continue
            selected.append(sample)
            used_images.add(sample.image_path)
            if len(selected) == IMAGES_PER_CLASS:
                break

        if len(selected) < IMAGES_PER_CLASS:
            raise RuntimeError(
                f"类别 id {cls} 的去重后样本不足 {IMAGES_PER_CLASS} 张 "
                f"（当前仅 {len(selected)} 张）。"
            )

        RNG.shuffle(selected)
        cls_train_count = train_target[cls]
        train_samples.extend(selected[:cls_train_count])
        val_samples.extend(selected[cls_train_count:])

    RNG.shuffle(train_samples)
    RNG.shuffle(val_samples)
    return train_samples, val_samples


def export_split(samples: Iterable[Sample], split: str, output_root: Path) -> None:
    """
    将一组 Sample 导出到 YOLO_HN 的 images/{split}, labels/{split} 目录中。
    """
    image_dir = output_root / "images" / split
    label_dir = output_root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    samples_list = list(samples)
    for sample in tqdm(samples_list, desc=f"[3/3] 导出 YOLO_HN {split}", unit="img"):
        base = sample.base_name
        dst_img = image_dir / f"{base}{sample.image_path.suffix}"
        dst_label = label_dir / f"{base}.txt"

        shutil.copy2(sample.image_path, dst_img)
        dst_label.write_text("\n".join(sample.label_lines) + "\n")


def build_hn_dataset(window_root: Path, coco_root: Path) -> Path:
    """
    利用 YOLO_RGB_window + YOLO_RGB_carton 构建 Hard Negative YOLO 数据集 YOLO_HN。

    - 6 个类别: window / tv / laptop / microwave / oven / refrigerator
    - 每类 200 张（IMAGES_PER_CLASS）
    - train: 1000 张, val: 200 张
    - 目录结构: images/train, images/val, labels/train, labels/val
    """
    output = DATASETS_ROOT / "YOLO_HN"
    if output.exists():
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)

    print("[3/3] 构建 Hard Negative YOLO 数据集 (YOLO_HN)...")

    # 0 类: window（来自 RGB_window）
    window_samples = gather_window_samples(window_root)
    per_class: Dict[int, List[Sample]] = {0: window_samples}

    # 1~5 类: tv, laptop, microwave, oven, refrigerator（来自 COCO）
    coco_samples = gather_coco_samples(coco_root)
    for cls, samples in coco_samples.items():
        per_class.setdefault(cls, []).extend(samples)

    # 检查每类是否有足够样本
    for cls in range(len(HN_CLASSES)):
        num = len(per_class.get(cls, []))
        if num < IMAGES_PER_CLASS:
            raise RuntimeError(
                f"类别 '{HN_CLASSES[cls]}' 样本不足 {IMAGES_PER_CLASS} 张 "
                f"(当前 {num} 张)，无法构建 HN 数据集。"
            )

    # 采样 & 划分 train/val
    train_samples, val_samples = split_balanced(per_class)

    # 导出到 YOLO_HN
    export_split(train_samples, "train", output)
    export_split(val_samples, "val", output)

    print(
        f"[3/3] 完成: YOLO_HN 数据集构建完成 -> {output} "
        f"(train: {len(train_samples)} 张, val: {len(val_samples)} 张)"
    )
    return output


# ====================== 主入口 ======================

def main() -> None:
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    # 步骤 1: COCO -> YOLO (RGB_carton)
    rgb_carton = convert_rgb_carton()

    # 步骤 2: VOC -> YOLO (RGB_window)
    rgb_window = convert_rgb_window()

    # 步骤 3: 构建 HN 数据集
    build_hn_dataset(rgb_window, rgb_carton)


if __name__ == "__main__":
    main()
