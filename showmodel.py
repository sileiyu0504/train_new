#!/usr/bin/env python3
from pathlib import Path

import torch
from torchview import draw_graph

from ultralytics.nn.tasks import DetectionModel

# 4 通道输入的 RGBD 模型
model = DetectionModel(cfg="ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml", ch=4)
model.eval()

# Dummy 输入：BCHW = 1x4x640x640
sample = torch.zeros(1, 4, 640, 640)

graph = draw_graph(
    model,
    input_data=sample,
    expand_nested=True,
    depth=3,  # 视图展开深度可调
    graph_name="yolo11_rgbd",
)

# 保存 dot 源文件，便于后续在任意机器上用 graphviz 渲染
dot_path = Path("yolo11_rgbd_arch.dot")
dot_path.write_text(graph.visual_graph.source)

# 尝试渲染为 SVG/PNG；若系统缺少 graphviz 的 dot 可执行，会打印提示但保留 dot 文件
for fmt in ("svg", "png"):
    try:
        graph.visual_graph.render(f"yolo11_rgbd_arch", format=fmt, cleanup=True)
        print(f"Saved yolo11_rgbd_arch.{fmt}")
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to render {fmt}: {exc}. Dot saved at {dot_path}")
