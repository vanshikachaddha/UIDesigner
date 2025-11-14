#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoder.py

Simple CNN encoder for pix2code-style model.
"""

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Simple CNN encoder:
    - 3 conv blocks (conv + ReLU + maxpool)
    - Global average pool at the end
    """

    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112x112

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56x56

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28x28
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) -> (B, out_dim)
        """
        x = self.features(x)
        x = self.global_pool(x)         # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)       # (B, 128)
        x = self.fc(x)                  # (B, out_dim)
        return x
