# RGB-D 模型改造说明

## 主要改动

1. **数据通道扩展**
   - `ultralytics/ultralytics/data/dataset.py`：针对 YOLODataset 新增深度图索引、读取与 4 通道拼接逻辑，并在 `collate_fn`、`get_image_and_label` 中透传 `depth` 标志。
   - `ultralytics/ultralytics/data/augment.py`：`RandomHSV` 针对前三个 RGB 通道变换；`Format` 拆分 `RGB+Depth`，输出 `depth` 张量。
   - `ultralytics/ultralytics/models/yolo/detect/train.py`、`val.py`、`engine/trainer.py`、`engine/validator.py`：预处理、推理阶段同步归一化/缩放深度图，并在编译模式或验证模式下传入 `(img, depth)`。

2. **RGB-D 模块**
   - 新增 `ultralytics/ultralytics/nn/modules/rgbd.py`，包含 `DepthEncoder`（多尺度深度编码）与 `DepthGuidedFusion`（非注意力的引导加法）。
   - `ultralytics/ultralytics/nn/tasks.py`：检测模型支持 `rgbd` 配置，构建深度分支及颈部融合，`loss` 和 `_predict_once` 可接受 `(img, depth)`。

3. **模型配置**
   - 新建 `ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml`，在标准 YOLO11 结构基础上添加 `rgbd` 配置块，可通过 `model=yolo11s-rgbd.yaml` 等方式启用。

4. **数据集参数**
   - `ultralytics/ultralytics/data/build.py` 支持 `depth_*` 配置（如 `depth_train`、`depth_val` 或 `depth: {train: depth/train, ...}`），并传递给数据集以匹配深度图片。

## 数据集准备

1. 使用 `prepare_dataset.py` 生成
   ```
   prepared_dataset/
     images/{train,val}
     labels/{train,val}
     depth/{train,val}
   ```
   RGB/Depth 文件需同名（可通过 `depth_replace: ["_color_", "_depth_"]` 定义替换规则）。

2. 数据配置示例
   ```yaml
   path: prepared_dataset
   train: images/train
   val: images/val
   depth:
     train: depth/train
     val: depth/val
   depth_replace: ["_color_", "_depth_"]
   depth_scale: 65535.0  # 若深度为 16bit png
   names:
     0: carton
   ```

## 训练策略（建议）

1. **阶段一（纯 RGB 预训练）**
   - 使用原始 `yolo11s.yaml` 或更小尺寸，加载 7k 纸箱 RGB 数据训练，得到 `best_rgb.pt`。

2. **阶段二（RGBD 微调）**
   - 切换模型为 `ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml`。
   - `yolo task=detect mode=train model=... data=rgbd.yaml pretrained=best_rgb.pt freeze=10`，前期冻结大部分 backbone，仅训练颈部和检测头，引导深度分支稳定。
   - 5~10 epoch 后解冻更多层，并保持 `depth` 分支较大的学习率（可用 `lr0`/`lrf` 或自定义 `param_group`）。
   - 增强策略需保证 RGB 与 Depth 同步（关闭 Mosaic、CopyPaste，保留尺度/仿射/翻转）。
   - 若 RGBD 样本有限，可使用伪深度（MiDaS/Zoedepth）先热身深度编码器，再用真实深度微调。

3. **验证指标**
   - 标准 `mAP50`/`mAP50-95`。
   - 额外统计窗户/玻璃等 Hard Negative 上的误检率（FPPI），以突出深度引导对误检的改善。
   - 对比相同精度下所需 RGBD 数据量（例如仅用 50% RGBD 数据即可达到 RGB-only 基线的 96%+ mAP）。

## 使用示例

```bash
# 阶段一：RGB 预训练
yolo task=detect mode=train model=ultralytics/ultralytics/cfg/models/11/yolo11s.yaml \
     data=rgb_carton.yaml epochs=100 imgsz=640 project=runs/rgb_pretrain name=y11s_rgb

# 阶段二：RGBD 微调
yolo task=detect mode=train model=ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml \
     data=rgbd_carton.yaml epochs=60 imgsz=640 project=runs/rgbd_ft name=y11s_rgbd \
     pretrained=runs/rgb_pretrain/y11s_rgb/weights/best.pt freeze=12 optimizer=AdamW cos_lr=1
```

## 文件列表

- 新建：`ultralytics/ultralytics/nn/modules/rgbd.py`、`ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml`、`RGBD_CHANGES.md`
- 关键修改：`ultralytics/ultralytics/data/{build.py,dataset.py,augment.py}`、`ultralytics/ultralytics/models/yolo/detect/{train.py,val.py}`、`ultralytics/ultralytics/engine/{trainer.py,validator.py}`、`ultralytics/ultralytics/nn/tasks.py`

如需推理纯 RGB 数据，可直接加载 `yolo11-rgbd` 权重并仅传入 RGB 图，深度分支自动退化为无效输入，不影响原有流程。
