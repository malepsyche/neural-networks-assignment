import torch
from torch import nn


class AlexNetLike(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1, 16,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                16, 32,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                64 * 4 * 4,
                64,
                bias=True,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(
                64,
                32,
                bias=True,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),

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