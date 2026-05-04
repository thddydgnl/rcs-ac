from __future__ import annotations

import torch
from torch import nn


class RGBBEVHead(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 4, height: int = 128, width: int = 128) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.encoder = nn.Sequential(
            nn.Conv2d(int(in_channels), 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 48, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, int(out_channels), kernel_size=4, stride=2, padding=1),
        )

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.decoder(self.encoder(rgb.to(dtype=torch.float32)))
        if int(logits.shape[-2]) != self.height or int(logits.shape[-1]) != self.width:
            logits = torch.nn.functional.interpolate(
                logits,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
        return {
            "rgb_bev_logits": logits,
            "rgb_bev": torch.sigmoid(logits),
        }
