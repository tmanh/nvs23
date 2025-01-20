import torch.nn as nn


class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, x, out_shape):
        h, w = out_shape
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x[:, :, :h, :w]