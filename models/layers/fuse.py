import torch.nn as nn

from models.layers.gruunet import GRUUNet

from models.layers.legacy_fuse import *
from models.layers.minGRU.mingru import minGRU
from .osa_utils import *


class SNetDS2BNBase8(nn.Module):
    """2D U-Net style network with batch normalization and dilated convolutions."""

    def __init__(self, in_dim, base_filter=8):
        super(SNetDS2BNBase8, self).__init__()

        # Initial Convolutions
        self.sconv0_0 = self.conv_bn(in_dim, base_filter, dilation=1)
        self.sconv0_1 = self.conv_bn(base_filter, base_filter * 2, dilation=1)
        self.sconv0_2 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=2)
        self.sconv0_3 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=1, relu=True)

        # Branch 1
        self.sconv1_2 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=3)
        self.sconv1_3 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=1, relu=True)

        # Branch 2
        self.sconv2_2 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=4)
        self.sconv2_3 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=1, relu=True)

        # Concatenation & Final Conv
        self.sconv3_0 = nn.Conv2d(base_filter * 2 * 3, base_filter * 2, kernel_size=3, padding=1, bias=False)

    def conv_bn(self, in_channels, out_channels, dilation=1, relu=True):
        """Helper function for convolution + batch normalization."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Convolutions
        x0_0 = self.sconv0_0(x)
        x0_1 = self.sconv0_1(x0_0)
        x0_2 = self.sconv0_2(x0_1)
        x0_3 = self.sconv0_3(x0_2)

        # Branch 1
        x1_2 = self.sconv1_2(x0_2)
        x1_3 = self.sconv1_3(x1_2)

        # Branch 2
        x2_2 = self.sconv2_2(x0_2)
        x2_3 = self.sconv2_3(x2_2)

        # Concatenation & Final Convolution
        x_concat = torch.cat([x0_3, x1_3, x2_3], dim=1)
        out = self.sconv3_0(x_concat)

        return out

class LocalFusion(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

        self.shallow = SNetDS2BNBase8(3)
        self.gru = minGRU(16)
        self.gru_back = minGRU(16)

        self.alpha = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.out = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def forward(self, prj_colors):
        B, V, C, H, W = prj_colors.shape

        prj_colors = prj_colors.view(B * V, C, H, W)
        fs = self.shallow(prj_colors)

        fs = fs.view(B, V, -1, H, W).permute(0, 3, 4, 1, 2)
        fs = self.gru(fs)
        fs = self.gru_back(torch.flip(fs, dims=[1]))
        fs = torch.flip(fs, dims=[1])
        fs = fs.view(B, H, W, V, -1).permute(0, 3, 4, 1, 2)

        fs = fs.contiguous().view(B * V, -1, H, W)
        out = self.out(fs).view(B, V, 3, H, W)
        alpha = self.alpha(fs).view(B, V, 1, H, W)

        return torch.sum(out * torch.softmax(alpha, dim=1), dim=1)


class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fuse = GRUUNet(
            [64, 64, 256, 512, 1024],
            [64, 256, 512, 1024, 2048],
            [2048, 2048, 1024, 512, 256],
            [2048, 1024, 512, 256, 64]
        )

    def forward(self, prj_feats, prj_depths):
        return self.fuse(prj_feats, prj_depths)
    
    # def create_fuse_layer(self, in_dim, out_dim):
    #     return nn.Sequential(
    #         nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, padding=0),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(out_dim * 4, out_dim * 4, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(out_dim * 4, out_dim, kernel_size=3, padding=1),
    #     )

    # def __init__(self) -> None:
    #     super().__init__()
    #     self.fuse5 = FusionInner(2048)
    #     self.fuse4 = FusionOuter(1024, 2048)
    #     self.fuse3 = FusionOuter(512, 1024)
    #     self.fuse2 = FusionOuter(256, 512)
    #     self.fuse1 = Merger(64, 256)

    # def forward(self, prj_feats, prj_src_feats, prj_depths):
    #     f5 = self.fuse5(prj_feats[-1], prj_src_feats[-1], prj_depths[-1])
    #     f4 = self.fuse4(f5, prj_feats[-2], prj_src_feats[-2], prj_depths[-2])
    #     f3 = self.fuse3(f4, prj_feats[-3], prj_src_feats[-3], prj_depths[-3])
    #     f2 = self.fuse2(f3, prj_feats[-4], prj_src_feats[-4], prj_depths[-4])
    #     f1 = self.fuse1(f2, prj_feats[-5], prj_src_feats[-5], prj_depths[-5])
    #     return f1