import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.cat import CrossAttention
from models.layers.weight_init import trunc_normal_
from models.layers.gruunet import GRUUNet
from models.layers.basic import PixelShuffleUpsample


from models.layers.legacy_fuse import *
from .osa_utils import *


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