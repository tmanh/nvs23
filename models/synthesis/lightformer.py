import math
import torch.nn as nn

from models.layers.fuse import Fusion
from models.layers.weight_init import trunc_normal_
from models.synthesis.base import BaseModule


class Decoder(nn.Module):
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.out = nn.Conv2d(64, 3, 3, 1, 1)
        self.up = nn.Sequential(
            nn.Conv2d(67, 64 * 4, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
            nn.GELU(),
        )

        self.apply(self._init_weights)

    def forward(self, x):
        return self.out(self.up(x))


class LightFormer(BaseModule):
    def freeze(self):
        self.freeze_shallow_color_encoder()

    def init_fusion_module(self):
        self.merge_net = Fusion()
        self.out = Decoder()
