import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.osa import VFBlock
from models.layers.vivim import ViLViewFuseLayer, ViLViewLayer
from models.layers.weight_init import trunc_normal_
from models.layers.gruunet import GRUUNet


from .osa_utils import *


class Fusion(nn.Module):
    def create_fuse_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 4, out_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 4, out_dim, kernel_size=3, padding=1),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def __init__x(self) -> None:
        super().__init__()

        self.fuse5 = ViLViewLayer(2816 + 1)
        
        self.up4 = nn.Sequential(
            nn.Conv2d(2816 + 1, (1408 + 1) * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
        )
        self.fuse4 = ViLViewLayer(1408 + 1)

        self.up3 = nn.Sequential(
            nn.Conv2d(1408 + 1, (704 + 1) * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
        )
        self.fuse3 = ViLViewLayer(704 + 1)

        self.up2 = nn.Sequential(
            nn.Conv2d(704 + 1, (352 + 1) * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
        )
        self.fuse2 = ViLViewLayer(352 + 1)

        self.up1 = nn.Sequential(
            nn.Conv2d(352 + 1, 128 * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2),
        )

        self.apply(self._init_weights)

    def forwardx(self, prjs):
        # torch.Size([1, 2, 64, 256, 192])
        # torch.Size([1, 2, 256, 128, 96])
        # torch.Size([1, 2, 512, 64, 48])
        # torch.Size([1, 2, 1024, 32, 24])
        # torch.Size([1, 2, 2048, 16, 12])
        prev_prj = self.up4(
            self.fuse5(prjs[3])
        ).unsqueeze(1)  # N, V, C, H, W
        
        prev_prj = self.up3(
            self.fuse4(
                torch.cat([prev_prj, prjs[2]], dim=1)
            )
        ).unsqueeze(1)  # N, V, C, H, W
        
        prev_prj = self.up2(
            self.fuse3(
                torch.cat([prev_prj, prjs[1]], dim=1)
            )
        ).unsqueeze(1)  # N, V, C, H, W

        prev_prj = self.up1(
            self.fuse2(
                torch.cat([prev_prj, prjs[0]], dim=1)
            )
        )

        # torch.mean(prjs[0], dim=1)[:, :-1]
        return prev_prj
    
    def __init__(self) -> None:
        super().__init__()

        self.gru = GRUUNet()
        self.fuse = VFBlock(67)

    def forward(self, prjs):
        d = prjs[0][:, :, -1:]
        # out, _ = self.gru(prjs)
        out = self.fuse(prjs[0], d)
        return out