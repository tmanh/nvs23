import torch.nn as nn

from models.layers.fuse import Fusion
from models.synthesis.base import BaseModule


class LightFormer(BaseModule):
    def freeze(self):
        self.freeze_shallow_color_encoder()

    def init_fusion_module(self):
        self.merge_net = Fusion()
        self.up1 = nn.Conv2d(96, 64, 3, 1, 1)
        self.up2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)
