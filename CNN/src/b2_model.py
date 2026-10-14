import torch
from torch import nn


class VGGLike(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(
                1,
                8,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.Conv2d(
                8,
                8,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(8),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 2
            nn.Conv2d(
                8,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            # Block 3
            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                32,
                1,
                bias=True,
            ),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x.squeeze(1)