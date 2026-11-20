
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import setup_logger


class Avg2MaxPooling(nn.Module):
    """
    Custom pooling operation combining average and maximum pooling.

    Output = Average - 2 * Maximum

    Preserves the original notebook implementation.
    """

    def __init__(self, pool_size=3, stride=2):
        super().__init__()

        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        h, w = x.shape[-2:]

        out_h = (h + self.stride - 1) // self.stride
        out_w = (w + self.stride - 1) // self.stride

        pad_h = max(
            (out_h - 1) * self.stride + self.pool_size - h, 0
        )

        pad_w = max(
            (out_w - 1) * self.stride + self.pool_size - w, 0
        )

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padding = (pad_left, pad_right, pad_top, pad_bottom)

        x_avg = F.pad(x, padding, value=0)

        mask = torch.ones_like(x)
        mask = F.pad(mask, padding, value=0)

        avg_sum = F.avg_pool2d(
            x_avg,
            kernel_size=self.pool_size,
            stride=self.stride,
            divisor_override=1
        )

        valid_count = F.avg_pool2d(
            mask,
            kernel_size=self.pool_size,
            stride=self.stride,
            divisor_override=1
        )

        avg = avg_sum / valid_count.clamp_min(1)

        x_max = F.pad(
            x,
            padding,
            value=float("-inf")
        )

        max_val = F.max_pool2d(
            x_max,
            kernel_size=self.pool_size,
            stride=self.stride
        )

        return avg - (max_val + max_val)


class ConvBlock(nn.Module):
    """Standard convolutional block."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1
        )

        self.bn = nn.BatchNorm2d(
            out_channel,
            eps=0.001,
            momentum=0.01
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class DepthWiseSerparableConv(nn.Module):
    """Depthwise separable convolutional block."""

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size=3,
            padding=1,
            groups=in_channel
        )

        self.pointwise = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=1
        )

        self.bn = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class FibonacciNet(nn.Module):
    """
    FibonacciNet architecture for binary thyroid image classification.

    Default configuration:
        Input channels: 3
        Output classes: 1
    """

    def __init__(self, input_channels=3, num_classes=1):
        super().__init__()

        # Block 1: 21 filters
        self.block1 = ConvBlock(
            input_channels, 21
        )

        # Block 2: 34 filters
        self.block2 = nn.Conv2d(
            21, 34, kernel_size=3, padding=1
        )

        self.block2_bn = nn.BatchNorm2d(
            34, eps=0.001, momentum=0.01
        )

        # Block 3: 55 filters
        self.block3 = nn.Conv2d(
            34, 55, kernel_size=3, padding=1
        )

        self.block3_bn = nn.BatchNorm2d(
            55, eps=0.001, momentum=0.01
        )

        # PCB1
        self.pcb1_conv1 = nn.Conv2d(
            34, 24, kernel_size=3, padding=1
        )

        self.pcb1_pool1 = Avg2MaxPooling()

        self.pcb1_conv2 = nn.Conv2d(
            24, 24, kernel_size=3, padding=1
        )

        self.pcb1_pool2 = Avg2MaxPooling()

        # Block 4
        self.block4 = ConvBlock(
            55, 89
        )

        # PCB2
        self.pcb2_conv1 = nn.Conv2d(
            55, 24, kernel_size=3, padding=1
        )

        self.pcb2_pool1 = Avg2MaxPooling()

        self.pcb2_conv2 = nn.Conv2d(
            24, 24, kernel_size=3, padding=1
        )

        self.pcb2_pool2 = Avg2MaxPooling()

        # Block 5
        self.block5 = ConvBlock(
            113, 144
        )

        # Block 6
        self.block6 = DepthWiseSerparableConv(
            168, 233
        )

        # Block 7
        self.block7 = DepthWiseSerparableConv(
            233, 377
        )

        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            377, num_classes
        )

    def forward(self, x):

        # Block 1
        x = self.block1(x)
        x = F.max_pool2d(x, 2)

        # Block 2
        x = self.block2(x)
        x = self.block2_bn(x)

        x2 = F.relu(x)

        x = F.max_pool2d(x, 2)

        # Block 3
        x = self.block3(x)
        x = self.block3_bn(x)

        x3 = F.relu(x)

        x = F.max_pool2d(x, 2)

        # PCB1
        pcb1 = self.pcb1_conv1(x2)
        pcb1 = self.pcb1_pool1(pcb1)

        pcb1 = self.pcb1_conv2(pcb1)
        pcb1 = self.pcb1_pool2(pcb1)

        # Block 4
        x = self.block4(x)
        x = F.max_pool2d(x, 2)

        # Feature concatenation 1
        pcb1 = F.interpolate(
            pcb1,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, pcb1], dim=1)

        # PCB2
        pcb2 = self.pcb2_conv1(x3)
        pcb2 = self.pcb2_pool1(pcb2)

        pcb2 = self.pcb2_conv2(pcb2)
        pcb2 = self.pcb2_pool2(pcb2)

        # Block 5
        x = self.block5(x)
        x = F.max_pool2d(x, 2)

        # Feature concatenation 2
        pcb2 = F.interpolate(
            pcb2,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat([x, pcb2], dim=1)

        # Block 6
        x = self.block6(x)

        # Block 7
        x = self.block7(x)

        # Classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
