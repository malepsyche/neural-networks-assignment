import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.relu2 = nn.ReLU()

        # projection shortcut is required when spatial dimensions or channels change
        if (
            stride != 1
            or in_channels != out_channels
        ):
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
            )
        else:
            self.shortcut = nn.Identity()

        self.output_relu = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = out + identity
        out = self.output_relu(out)

        return out


class ResNetLike(nn.Module):
    def __init__(self):
        super().__init__()

        # input: 1 x 64 x 64
        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # 64 x 64 -> 32 x 32
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        # 16 x 32 x 32
        self.block1 = ResidualBlock(
            in_channels=16,
            out_channels=16,
            stride=1,
        )

        # 16 x 32 x 32 -> 32 x 16 x 16
        self.block2 = ResidualBlock(
            in_channels=16,
            out_channels=32,
            stride=2,
        )

        # 32 x 16 x 16 -> 64 x 8 x 8
        self.block3 = ResidualBlock(
            in_channels=32,
            out_channels=64,
            stride=2,
        )

        # 64 x 8 x 8 -> 64 x 1 x 1
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.classifier = nn.Linear(
            64,
            1,
            bias=True,
        )

    def forward(self, x):
        x = self.stem(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avg_pool(x)

        x = torch.flatten(
            x,
            start_dim=1,
        )

        x = self.classifier(x)

        return x.squeeze(1)