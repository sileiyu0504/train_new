# RGB-D 改造详细变更（Changes_1）

> 说明：以下内容以接近 `git diff` 的形式列出了此次 RGB-D 相关开发涉及的所有新建与修改文件，便于审查与后续排查。

---

## 🆕 新增文件

1. `ultralytics/ultralytics/nn/modules/rgbd.py`  
   - 定义 `DepthEncoder` 与 `DepthGuidedFusion` 两个模块，负责深度支路特征提取和颈部融合。
   ```diff
   +class DepthEncoder(nn.Module):
   +    def __init__(self, in_channels=1, channels=(32, 64, 128, 256)):
   +        ...
   +    def forward(self, depth):
   +        return {"p3": p3, "p4": p4, "p5": p5}
   +
   +class DepthGuidedFusion(nn.Module):
   +    def __init__(self, rgb_channels, depth_channels):
   +        ...
   +    def forward(self, rgb, depth):
   +        guide = torch.tanh(self.align(depth_resized))
   +        return self.mixer(rgb + guide)
   ```

2. `ultralytics/ultralytics/cfg/models/11/yolo11-rgbd.yaml`  
   - 在标准 YOLO11 结构上新增 `rgbd` 配置块（启用开关、深度编码通道、融合层映射）。
   ```diff
   +rgbd:
   +  enabled: true
   +  depth_channels: 1
   +  encoder_channels: [48, 96, 192, 256]
   +  fusion_layers:
   +    p3: 16
   +    p4: 19
   +    p5: 22
   ```

3. `RGBD_CHANGES.md`  
   - 顶层说明文档，记录 RGBD 增强的背景、数据规范、训练策略以及新增文件清单。

4. `Changes_1.md`（本文）  
   - 作为变更总览备忘，便于查阅。

---

## ✏️ 已修改文件（节选 diff）

### 1. `ultralytics/ultralytics/data/build.py`
```diff
@@ def build_yolo_dataset(...):
-    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
-    return dataset(..., data=data, fraction=cfg.fraction if mode == "train" else 1.0)
+    dataset = YOLOMultiModalDataset if multi_modal else YOLODataset
+
+    def _resolve_depth_path() -> str | None:
+        ...
+        return str(path)
+
+    depth_path = _resolve_depth_path()
+    return dataset(
+        ...,
+        data=data,
+        fraction=cfg.fraction if mode == "train" else 1.0,
+        split=mode,
+        depth_path=depth_path,
+    )
```

### 2. `ultralytics/ultralytics/data/dataset.py`
```diff
@@ class YOLODataset(...):
-    def __init__(..., data=None, task="detect", **kwargs):
-        ...
-        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)
+    def __init__(..., data=None, task="detect", split="train", depth_path=None, **kwargs):
+        self.depth_dir = Path(depth_path) if depth_path else None
+        self.depth_enabled = self.depth_dir is not None
+        self.depth_replace = tuple(self.data.get("depth_replace", ("_color_", "_depth_")))
+        self.depth_scale = float(self.data.get("depth_scale", 65535.0))
+        self.depth_fill = float(self.data.get("depth_fill_value", 0.0))
+        self.depth_fill_byte = np.uint8(np.clip(self.depth_fill, 0.0, 1.0) * 255)
+        super().__init__(*args, channels=self.data.get("channels", 3), **kwargs)
+        self._build_depth_index()

+    def _build_depth_index(self):
+        lookup = {Path(p).stem: p for p in self.get_img_files(str(self.depth_dir))}
+        ...
+        self.depth_files = depth_list
+
+    def _load_depth_map(self, index, target_shape):
+        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
+        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
+        depth = np.clip(depth.astype(np.float32) / scale, 0.0, 1.0)
+        return depth
+
@@
-    def get_image_and_label(self, index):
-        label = super().get_image_and_label(index)
-        return label
+    def get_image_and_label(self, index):
+        label = super().get_image_and_label(index)
+        if not self.depth_enabled:
+            label["use_depth"] = False
+            return label
+        depth = self._load_depth_map(index, (h, w))
+        ...
+        label["img"] = np.concatenate((img, depth_channel), axis=2)
+        label["use_depth"] = True
+        return label

@@ YOLODataset.collate_fn
-        if k in {"img", "text_feats"}:
+        if k in {"img", "text_feats", "depth"}:
             value = torch.stack(value, 0)
```

### 3. `ultralytics/ultralytics/data/augment.py`
```diff
@@ class RandomHSV:
-        img = labels["img"]
-        if img.shape[-1] != 3:
+        img = labels["img"]
+        if img.shape[-1] < 3:
             return labels
-        if self.hgain or self.sgain or self.vgain:
-            dtype = img.dtype
-            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
-            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)
+        rgb = img[..., :3]
+        ...
+            hue, sat, val = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV))
+            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=rgb)
+            img[..., :3] = rgb

@@ class Format.__call__:
-        img = labels.pop("img")
+        img = labels.pop("img")
+        depth = None
+        if img.ndim == 3 and img.shape[2] > 3:
+            depth = img[..., 3:]
+            img = img[..., :3]
         ...
-        labels["img"] = self._format_img(img)
+        labels["img"] = self._format_img(img)
+        if depth is not None:
+            labels["depth"] = self._format_depth(depth)
+        elif labels.pop("use_depth", False):
+            labels.setdefault("depth", torch.zeros(1, h, w))

+    def _format_depth(...):
+        depth = depth.transpose(2, 0, 1)
+        return torch.from_numpy(depth)
```

### 4. `ultralytics/ultralytics/models/yolo/detect/train.py`
```diff
@@ DetectionTrainer.preprocess_batch
         for k, v in batch.items():
             if isinstance(v, torch.Tensor):
                 batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
         batch["img"] = batch["img"].float() / 255
+        if "depth" in batch:
+            batch["depth"] = batch["depth"].float() / 255
         if self.args.multi_scale:
             ...
                 imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
+                if "depth" in batch:
+                    batch["depth"] = nn.functional.interpolate(batch["depth"], size=ns, mode="nearest")
```

### 5. `ultralytics/ultralytics/models/yolo/detect/val.py`
```diff
@@ DetectionValidator.preprocess
         for k, v in batch.items():
             if isinstance(v, torch.Tensor):
                 batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
         batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
+        if "depth" in batch:
+            dtype = batch["img"].dtype if self.args.half else torch.float32
+            batch["depth"] = batch["depth"].to(dtype) / 255
         return batch
```

### 6. `ultralytics/ultralytics/engine/trainer.py`
```diff
@@ within training loop
-                    if self.args.compile:
-                        preds = self.model(batch["img"])
-                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
+                    if self.args.compile:
+                        model_inputs = (batch["img"], batch["depth"]) if "depth" in batch else batch["img"]
+                        preds = self.model(model_inputs)
+                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
```

### 7. `ultralytics/ultralytics/engine/validator.py`
```diff
@@ BaseValidator.__call__
-            with dt[1]:
-                preds = model(batch["img"], augment=augment)
+            with dt[1]:
+                model_inputs = (batch["img"], batch["depth"]) if "depth" in batch else batch["img"]
+                preds = model(model_inputs, augment=augment)
```

### 8. `ultralytics/ultralytics/nn/modules/__init__.py`
```diff
@@
-from .conv import (..., ConvTranspose, ...)
-from .head import (...)
+from .conv import (..., ConvTranspose, ...)
+from .head import (...)
+from .rgbd import DepthEncoder, DepthGuidedFusion
@@ __all__
-    "ConvTranspose",
+    "ConvTranspose",
+    "DepthEncoder",
+    "DepthGuidedFusion",
```

### 9. `ultralytics/ultralytics/nn/tasks.py`
```diff
@@ DetectionModel.__init__
         self.model, self.save = parse_model(...)
         ...
+        self.rgbd_cfg = self.yaml.get("rgbd", {})
+        self.rgbd_enabled = bool(self.rgbd_cfg.get("enabled"))
+        if self.rgbd_enabled:
+            self.depth_encoder = DepthEncoder(...)
+            self.depth_fusions = nn.ModuleDict({
+                level: DepthGuidedFusion(rgb_channels, depth_channels)
+                for level, layer_idx in fusion_layers.items()
+            })
+
@@ DetectionModel._predict_augment
-        img_size = x.shape[-2:]
+        if isinstance(x, (tuple, list)):
+            x = x[0]
+        img_size = x.shape[-2:]

@@ DetectionModel.loss
-        if preds is None:
-            preds = self.predict(batch["img"])
+        depth = batch.get("depth")
+        if preds is None:
+            preds = self.predict((batch["img"], depth))

@@ DetectionModel._predict_once
-        for m in self.model:
-            if m.f != -1:
-                x = y[m.f] if isinstance(m.f, int) else [...]
-            if profile:
-                self._profile_one_layer(m, x, dt)
-            x = m(x)
+        depth = None
+        if isinstance(x, (tuple, list)):
+            depth = x[1] if len(x) > 1 else None
+            x = x[0]
+        depth_feats = self.depth_encoder(depth) if self.rgbd_enabled and depth is not None else {}
+        for m in self.model:
+            ...
+            if self.rgbd_enabled and fusion_key in self.rgbd_layer_map:
+                depth_tensor = depth_feats.get(fusion_key)
+                x_in = self.depth_fusions[fusion_key](x_in, depth_tensor)
+            if profile:
+                self._profile_one_layer(m, x_in, dt)
+            x = m(x_in)
```

### 10. `ultralytics/ultralytics/data/augment.py`（已在第 3 条说明）  
> 该文件改动较多，已在上文展示关键 diff，未再重复。

---

如需进一步查看完整差异，可在仓库根目录执行 `git diff -- ultralytics/...`（若取消 `.gitignore` 对 `ultralytics/` 的忽略）或直接比对本文件所列片段。若有新增/删减计划，请以此文档为基线。***
