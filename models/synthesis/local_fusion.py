import math

import torch
import torch.nn as nn

from models.layers.fuse import Fusion, LocalFusion
from models.layers.weight_init import trunc_normal_
from models.synthesis.base import BaseModule
from models.synthesis.encoder import RadioEncoder


class LocalGRU(BaseModule):
    def init_color_encoder(self):
        pass

    def init_fusion_module(self):
        self.merge_net = LocalFusion()

    def extract_src_feats(self, colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs):
        ori_shape = colors.shape[-2:]

        prj_fs, prj_pts = self.warp_all_views(
            colors, depths, ori_shape,
            self.compute_K(K, ori_shape, colors.shape[-2:]),
            src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
            radius=self.opt.model.radius,
            max_alpha=False
        )

        return prj_fs, prj_pts

    def forward(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, visualize=False):
        prj_feats, prj_depths = self.extract_src_feats(colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs)

        final = self.merge_net(prj_feats)

        if visualize:
            mask = (torch.sum(prj_depths, dim=1) > 0).float().detach()
            return final, mask, prj_feats  # self.out(merged_fs), warped
        return final
    
    def freeze(self):
        pass