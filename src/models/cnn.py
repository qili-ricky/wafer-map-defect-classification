"""
CNN model architectures for wafer map defect classification.

TODO:
- SimpleCNN: 3-4 conv layers baseline
- ResNet18: transfer learning / from scratch
- Vision Transformer: small ViT for comparison
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Baseline CNN for wafer map classification."""

    def __init__(self, num_classes=9, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
