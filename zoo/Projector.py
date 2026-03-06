import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Projector(nn.Module):
    def __init__(self, in_channels=2, out_channels=16):
        super(Projector, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv_1 = ConvBlock(in_channels, out_channels // 2)
        self.conv_2 = ConvBlock(out_channels // 2, out_channels)
        self.final = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, input):
        out = self.conv_1(input)
        out = self.pool(out)
        out = self.conv_2(out)
        out = self.pool(out)
        out = self.final(out)
        return out
