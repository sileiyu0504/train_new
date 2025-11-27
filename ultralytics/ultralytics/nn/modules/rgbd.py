"""
RGB-D 辅助模块：包含深度分支特征提取器与颈部融合块。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2
from .conv import Conv


class DepthEncoder(nn.Module):
    """轻量深度编码器，用于生成与 P3/P4/P5 对齐的深度特征。"""

    def __init__(self, in_channels: int = 1, channels: tuple[int, int, int, int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels 参数需包含 4 个阶段的通道数")
        c1, c2, c3, c4 = channels
        self.stem = Conv(in_channels, c1, 3, 2)
        self.stage1 = nn.Sequential(Conv(c1, c1, 3, 2), C3k2(c1, c1, 1, False))
        self.stage2 = nn.Sequential(Conv(c1, c2, 3, 2), C3k2(c2, c2, 1, False))
        self.stage3 = nn.Sequential(Conv(c2, c3, 3, 2), C3k2(c3, c3, 1, True))
        self.stage4 = nn.Sequential(Conv(c3, c4, 3, 2), C3k2(c4, c4, 1, True))
        self.out_channels = {"p3": c2, "p4": c3, "p5": c4}

    def forward(self, depth: torch.Tensor | None) -> dict[str, torch.Tensor] | dict:
        if depth is None:
            return {}
        x = self.stem(depth)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return {"p3": p3, "p4": p4, "p5": p5}


class DepthGuidedFusion(nn.Module):
    """利用深度特征生成引导增益，与 RGB 语义特征做逐元素融合。"""

    def __init__(self, rgb_channels: int, depth_channels: int) -> None:
        super().__init__()
        self.align = Conv(depth_channels, rgb_channels, 1, 1, act=True)
        self.mixer = Conv(rgb_channels, rgb_channels, 1, 1, act=True)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor | None) -> torch.Tensor:
        if depth is None:
            return rgb
        if depth.shape[2:] != rgb.shape[2:]:
            depth = F.interpolate(depth, size=rgb.shape[2:], mode="bilinear", align_corners=False)
        guide = torch.tanh(self.align(depth))
        return self.mixer(rgb + guide)
