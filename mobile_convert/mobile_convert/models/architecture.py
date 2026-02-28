from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MultiScaleCSRDecoder(nn.Module):
    def __init__(self, in_channels_list: list[int]) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        mid_ch = 128
        self.reduce_32 = nn.Sequential(nn.Conv2d(in_channels_list[-1], mid_ch, 1), nn.ReLU(inplace=True))
        self.reduce_16 = nn.Sequential(nn.Conv2d(in_channels_list[-2], mid_ch, 1), nn.ReLU(inplace=True))
        self.backend = nn.Sequential(
            nn.Conv2d(mid_ch * 2 + 32, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, feats_16: torch.Tensor, feats_32: torch.Tensor, meta_map_16: torch.Tensor) -> torch.Tensor:
        f32_up = self.up(self.reduce_32(feats_32))
        f16 = self.reduce_16(feats_16)
        if f32_up.shape != f16.shape:
            f32_up = F.interpolate(f32_up, size=f16.shape[2:], mode="bilinear", align_corners=True)
        combined = torch.cat([f16, f32_up, meta_map_16], dim=1)
        return self.backend(combined)


class UltimateSpecialist(nn.Module):
    def __init__(self, model_name: str, pretrained_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained_backbone, features_only=True)
        ch_list = self.backbone.feature_info.channels()

        self.meta_proj = nn.Sequential(nn.Linear(3, 32), nn.LayerNorm(32), nn.GELU())
        self.count_heads = nn.ModuleList([MultiScaleCSRDecoder(ch_list) for _ in range(9)])
        self.measure_head = nn.Sequential(
            nn.Linear(ch_list[-1] + 32, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6),
        )

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, n_tiles, c, h, w = x.shape
        x_flat = x.view(bsz * n_tiles, c, h, w)

        feats = self.backbone(x_flat)
        f16 = feats[-2]
        f32 = feats[-1]

        m = self.meta_proj(meta)
        m_flat = m.repeat_interleave(n_tiles, dim=0)

        bh16, bw16 = f16.shape[2:]
        m_map16 = m_flat.view(bsz * n_tiles, 32, 1, 1).expand(-1, -1, bh16, bw16)

        tile_counts = []
        for head in self.count_heads:
            densities = torch.relu(head(f16, f32, m_map16))
            tile_sum = densities.sum(dim=(1, 2, 3)).view(bsz, n_tiles)
            tile_counts.append(tile_sum.sum(dim=1).unsqueeze(1))

        counts = torch.cat(tile_counts, dim=1)

        m_map32 = m_flat.view(bsz * n_tiles, 32, 1, 1).expand(-1, -1, f32.shape[2], f32.shape[3])
        combined32 = torch.cat([f32, m_map32], dim=1)
        pooled = torch.nn.functional.adaptive_avg_pool2d(combined32, 1).view(bsz, n_tiles, -1).mean(dim=1)
        measures = self.measure_head(pooled)
        return counts, measures


class OnnxTileModel(nn.Module):
    def __init__(self, model: UltimateSpecialist) -> None:
        super().__init__()
        self.model = model

    def forward(self, stack: torch.Tensor, meta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(stack, meta)
