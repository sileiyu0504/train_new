#!/usr/bin/env python3
"""
从 COCO 下载 tv/laptop/microwave/oven/refrigerator 五类数据，
与已有的 YOLO_RGB_window (window) 合并，构建 YOLO_HN 数据集。

改版说明：
- 不再逐张 HTTP 下载图片；
- 改为下载 COCO 官方的 train2017.zip、val2017.zip、annotations_trainval2017.zip；
- 从 zip 中按需解压只包含这 5 个类别的图片，再转为 YOLO 格式。

前置条件：
1) 你已经有 datasets/YOLO_RGB_window
2) 已安装: pycocotools, tqdm, requests
   pip install pycocotools tqdm requests
"""

from __future__ import annotations

import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

import requests
from pycocotools.coco import COCO
from tqdm.auto import tqdm

# ================================= 基本配置 =================================

DATASETS_ROOT = Path("datasets")

# 已有的 YOLO_RGB_window 路径
YOLO_RGB_WINDOW_ROOT = DATASETS_ROOT / "YOLO_RGB_window"

# COCO 原始压缩包及解压路径
COCO_ROOT = Path("datasets/COCO_5cls_raw")       # 原始 COCO ZIP + 按需解压后的图片 & 标注
YOLO_COCO_5CLS_ROOT = Path("datasets/YOLO_COCO_5cls")  # 转成 YOLO 的 5 类数据集

# Hard Negative 类别定义
HN_CLASSES = ["window", "tv", "laptop", "microwave", "oven", "refrigerator"]
IMAGES_PER_CLASS = 200
RNG = random.Random(42)

# COCO 原始 category_id -> 名称
# 官方 COCO category_id: 72 tv, 73 laptop, 78 microwave, 79 oven, 82 refrigerator
COCO_CAT_IDS = {
    72: "tv",
    73: "laptop",
    78: "microwave",
    79: "oven",
    82: "refrigerator",
}

# 在 YOLO 中的类别索引映射（统一到 HN_CLASSES 内部索引）
HN_NAME_TO_ID = {name: i for i, name in enumerate(HN_CLASSES)}
COCO_CAT_TO_HN: Dict[int, int] = {
    72: HN_NAME_TO_ID["tv"],
    73: HN_NAME_TO_ID["laptop"],
    78: HN_NAME_TO_ID["microwave"],
    79: HN_NAME_TO_ID["oven"],
    82: HN_NAME_TO_ID["refrigerator"],
}

# 三个 zip 的配置
COCO_ZIPS = {
    "train": ("http://images.cocodataset.org/zips/train2017.zip", COCO_ROOT / "train2017.zip"),
    "val":   ("http://images.cocodataset.org/zips/val2017.zip",   COCO_ROOT / "val2017.zip"),
    "ann":   ("http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
              COCO_ROOT / "annotations_trainval2017.zip"),
}

# 标注 json 路径
COCO_ANN_ROOT = COCO_ROOT / "annotations"
COCO_TRAIN_JSON = COCO_ANN_ROOT / "instances_train2017.json"
COCO_VAL_JSON   = COCO_ANN_ROOT / "instances_val2017.json"


# =========================== 小工具：下载文件 ===========================

def download_file(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[COCO] 已存在: {dst}, 跳过下载")
        return
    print(f"[COCO] 正在下载: {url} -> {dst}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        with open(dst, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dst.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


# ======================= 步骤 0：确保 COCO 压缩包 + 标注 =======================

def ensure_coco_zips_and_annotations() -> Tuple[Path, Path]:
    """
    确保 COCO 的 train/val 压缩包 和 标注 json 就位。
    返回 (train_json, val_json)
    """
    COCO_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) 下载三个 zip（若不存在）
    for key, (url, dst) in COCO_ZIPS.items():
        download_file(url, dst)

    # 2) 解压标注 zip（若 json 不存在）
    if not COCO_TRAIN_JSON.exists() or not COCO_VAL_JSON.exists():
        print("[COCO] 正在解压 annotations_trainval2017.zip ...")
        with zipfile.ZipFile(COCO_ZIPS["ann"][1], "r") as zf:
            zf.extractall(COCO_ROOT)

    if not COCO_TRAIN_JSON.exists() or not COCO_VAL_JSON.exists():
        raise FileNotFoundError("COCO 标注文件不存在或解压失败，请检查。")

    return COCO_TRAIN_JSON, COCO_VAL_JSON


# ================== 步骤 1：从 zip 中按需解压 5 类图片 ==================

def collect_and_extract_coco_images(train_json: Path, val_json: Path) -> None:
    """
    读取 COCO 标注，只挑选包含我们 5 个类别中任意一个的图片，
    然后从 train2017.zip / val2017.zip 中按需解压对应图像。

    图片解压目标:
        datasets/COCO_5cls_raw/images/train2017/
        datasets/COCO_5cls_raw/images/val2017/
    """
    img_root = COCO_ROOT / "images"

    for split, ann_path, zip_path in [
        ("train2017", train_json, COCO_ZIPS["train"][1]),
        ("val2017",   val_json,   COCO_ZIPS["val"][1]),
    ]:
        coco = COCO(str(ann_path))
        img_dir = img_root / split
        img_dir.mkdir(parents=True, exist_ok=True)

        # 找出五个类别的全部 image_id
        cat_ids = list(COCO_CAT_IDS.keys())   # [72,73,78,79,82]
        img_ids: Set[int] = set()
        for cid in cat_ids:
            img_ids.update(coco.getImgIds(catIds=[cid]))

        img_infos = coco.loadImgs(list(img_ids))
        print(f"[COCO] {split}: 5 类共有 {len(img_infos)} 张图片需要解压")

        # 从 zip 中按需解压
        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = set(zf.namelist())
            for img in tqdm(img_infos, desc=f"[COCO] 解压 {split} 图片", unit="img"):
                file_name = img["file_name"]  # 如 "000000123456.jpg"
                dst = img_dir / file_name
                if dst.exists():
                    continue
                arcname = f"{split}/{file_name}"  # zip 内部路径
                if arcname not in namelist:
                    # 理论上不应该发生，保险判断
                    continue
                # 解压到 img_root 下，这样会得到 images/train2017/xxx.jpg
                zf.extract(arcname, img_root)


# =========================== 步骤 2：COCO -> YOLO(5类) ===========================

@dataclass
class Sample:
    image_path: Path
    label_lines: List[str]
    classes: set[int]
    tag: str

    @property
    def base_name(self) -> str:
        return f"{self.tag}_{self.image_path.stem}"


def coco_to_yolo_5cls(train_json: Path, val_json: Path) -> Path:
    """
    将 COCO 标注转成只含 5 类的 YOLO 数据集。
    输出目录:
        datasets/YOLO_COCO_5cls/
          images/train2017, val2017
          labels/train2017, val2017
    """
    if YOLO_COCO_5CLS_ROOT.exists():
        shutil.rmtree(YOLO_COCO_5CLS_ROOT)

    (YOLO_COCO_5CLS_ROOT / "images").mkdir(parents=True, exist_ok=True)
    (YOLO_COCO_5CLS_ROOT / "labels").mkdir(parents=True, exist_ok=True)

    img_root = COCO_ROOT / "images"

    for split, ann_path in [("train2017", train_json), ("val2017", val_json)]:
        coco = COCO(str(ann_path))

        img_dir_src = img_root / split
        img_dir_dst = YOLO_COCO_5CLS_ROOT / "images" / split
        lbl_dir_dst = YOLO_COCO_5CLS_ROOT / "labels" / split
        img_dir_dst.mkdir(parents=True, exist_ok=True)
        lbl_dir_dst.mkdir(parents=True, exist_ok=True)

        # 获取 5 类的所有 image_id
        cat_ids = list(COCO_CAT_IDS.keys())
        img_ids: Set[int] = set()
        for cid in cat_ids:
            img_ids.update(coco.getImgIds(catIds=[cid]))

        img_ids = sorted(img_ids)

        for img_id in tqdm(img_ids, desc=f"[YOLO] 生成 {split} YOLO 标签 (5 类)", unit="img"):
            img_info = coco.loadImgs([img_id])[0]
            file_name = img_info["file_name"]
            src_img = img_dir_src / file_name
            if not src_img.exists():
                # 如果图片不存在（解压失败等），跳过
                continue

            # 获取该图像的所有相关标注
            ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)
            if not anns:
                continue

            w, h = img_info["width"], img_info["height"]
            lines: List[str] = []
            for ann in anns:
                cid = ann["category_id"]
                if cid not in COCO_CAT_TO_HN:
                    continue
                cls_id = COCO_CAT_TO_HN[cid]
                bbox = ann["bbox"]  # [x_min, y_min, width, height] in pixel
                x, y, bw, bh = bbox
                # 转 YOLO 格式 (归一化 xc, yc, w, h)
                x_c = (x + bw / 2.0) / w
                y_c = (y + bh / 2.0) / h
                ww = bw / w
                hh = bh / h
                lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}")

            if not lines:
                continue

            # 复制图片 & 写标签
            dst_img = img_dir_dst / file_name
            dst_lbl = lbl_dir_dst / (Path(file_name).stem + ".txt")
            shutil.copyfile(src_img, dst_img)
            dst_lbl.write_text("\n".join(lines) + "\n")

    print(f"[YOLO] 完成: COCO 5 类已转换为 YOLO 格式 -> {YOLO_COCO_5CLS_ROOT}")
    return YOLO_COCO_5CLS_ROOT


# ======================== 步骤 3：构建 YOLO_HN 数据集 ========================

def gather_window_samples(window_root: Path) -> List[Sample]:
    samples: List[Sample] = []
    for split in ("train", "val"):
        label_dir = window_root / "labels" / split
        image_dir = window_root / "images" / split
        if not label_dir.exists():
            continue

        for label_path in tqdm(list(label_dir.glob("*.txt")),
                               desc=f"[HN] 收集 window 样本 {split}", unit="file"):
            lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
            if not lines:
                continue
            cls_set = {int(float(line.split()[0])) for line in lines}
            if not cls_set:
                continue
            image_path = image_dir / f"{label_path.stem}.jpg"
            if not image_path.exists():
                png = image_dir / f"{label_path.stem}.png"
                if png.exists():
                    image_path = png
                else:
                    continue
            samples.append(Sample(image_path=image_path,
                                  label_lines=lines,
                                  classes=cls_set,
                                  tag=f"window_{split}"))
    return samples


def gather_coco_5cls_samples(dataset_root: Path) -> Dict[int, List[Sample]]:
    by_class: Dict[int, List[Sample]] = {HN_NAME_TO_ID[name]: [] for name in
                                         ["tv", "laptop", "microwave", "oven", "refrigerator"]}

    for subset in ("train2017", "val2017"):
        label_dir = dataset_root / "labels" / subset
        image_dir = dataset_root / "images" / subset
        if not label_dir.exists():
            continue

        for label_path in tqdm(list(label_dir.glob("*.txt")),
                               desc=f"[HN] 收集 COCO 5 类样本 {subset}", unit="file"):
            raw_lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
            keep: List[str] = []
            classes: set[int] = set()
            for line in raw_lines:
                parts = line.split()
                cls_id = int(float(parts[0]))
                if cls_id in by_class:
                    keep.append(line)
                    classes.add(cls_id)
            if not keep:
                continue

            image_path = image_dir / (label_path.stem + ".jpg")
            if not image_path.exists():
                alt = image_dir / (label_path.stem + ".png")
                if alt.exists():
                    image_path = alt
                else:
                    continue

            sample = Sample(image_path=image_path,
                            label_lines=keep,
                            classes=classes.copy(),
                            tag=f"coco_{subset}")
            for cls in classes:
                by_class[cls].append(sample)

    return by_class


def split_balanced(per_class: Dict[int, List[Sample]]) -> Tuple[List[Sample], List[Sample]]:
    per_class = {k: list(v) for k, v in per_class.items()}

    used_images: Set[Path] = set()
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []

    order = sorted(per_class.keys(), key=lambda c: len(per_class[c]))

    base_train = 1000 // len(HN_CLASSES)
    extra_train = 1000 % len(HN_CLASSES)
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

        for sample in candidates:
            if sample.image_path in used_images:
                continue
            selected.append(sample)
            used_images.add(sample.image_path)
            if len(selected) == IMAGES_PER_CLASS:
                break

        if len(selected) < IMAGES_PER_CLASS:
            raise RuntimeError(
                f"类别 id {cls} 去重后样本不足 {IMAGES_PER_CLASS} 张 (当前 {len(selected)} 张)。"
            )

        RNG.shuffle(selected)
        cls_train = train_target[cls]
        train_samples.extend(selected[:cls_train])
        val_samples.extend(selected[cls_train:])

    RNG.shuffle(train_samples)
    RNG.shuffle(val_samples)
    return train_samples, val_samples


def export_split(samples: Iterable[Sample], split: str, output_root: Path) -> None:
    img_dir = output_root / "images" / split
    lbl_dir = output_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    samples_list = list(samples)
    for s in tqdm(samples_list, desc=f"[HN] 导出 YOLO_HN {split}", unit="img"):
        base = s.base_name
        dst_img = img_dir / f"{base}{s.image_path.suffix}"
        dst_lbl = lbl_dir / f"{base}.txt"
        shutil.copyfile(s.image_path, dst_img)
        lbl_text = "\n".join(s.label_lines) + "\n"
        dst_lbl.write_text(lbl_text)


def build_yolo_hn(window_root: Path, coco_yolo_root: Path) -> Path:
    output = DATASETS_ROOT / "YOLO_HN"
    if output.exists():
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)

    print("[HN] 构建 YOLO_HN 数据集...")

    # window 类（0）
    window_samples = gather_window_samples(window_root)
    per_class: Dict[int, List[Sample]] = {HN_NAME_TO_ID["window"]: window_samples}

    # 5 个 COCO 类
    coco_per_class = gather_coco_5cls_samples(coco_yolo_root)
    for cls, s_list in coco_per_class.items():
        per_class.setdefault(cls, []).extend(s_list)

    # 检查数量
    for cls in range(len(HN_CLASSES)):
        num = len(per_class.get(cls, []))
        if num < IMAGES_PER_CLASS:
            raise RuntimeError(
                f"类别 '{HN_CLASSES[cls]}' 样本不足 {IMAGES_PER_CLASS} 张 (当前 {num} 张)。"
            )

    train_s, val_s = split_balanced(per_class)
    export_split(train_s, "train", output)
    export_split(val_s, "val", output)

    print(f"[HN] 完成: YOLO_HN -> {output} (train={len(train_s)}, val={len(val_s)})")
    return output


# =============================== 主流程 ===============================

def main():
    DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

    if not YOLO_RGB_WINDOW_ROOT.exists():
        raise FileNotFoundError(f"未找到 YOLO_RGB_window: {YOLO_RGB_WINDOW_ROOT}")

    # 0) 确保 COCO 压缩包与标注就绪
    train_json, val_json = ensure_coco_zips_and_annotations()

    # 1) 从 zip 中按需解压 5 类图片
    collect_and_extract_coco_images(train_json, val_json)

    # 2) COCO -> YOLO(5类)
    coco_yolo_root = coco_to_yolo_5cls(train_json, val_json)

    # 3) 与 YOLO_RGB_window 构建 YOLO_HN
    build_yolo_hn(YOLO_RGB_WINDOW_ROOT, coco_yolo_root)


if __name__ == "__main__":
    main()
