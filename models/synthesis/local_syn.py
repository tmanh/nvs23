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

        prj_colors, prj_depths, prj_weights = self.list_warp_all_views(
            colors, depths, ori_shape,
            self.compute_K(K, ori_shape, colors.shape[-2:]),
            src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
            radius=self.opt.model.radius,
            max_alpha=False
        )

        return prj_colors, prj_depths, prj_weights

    def forward(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, visualize=False, depth=False):
        with torch.no_grad():
            prj_colors, prj_depths, prj_weights = self.extract_src_feats(colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs)
            prj_fs = torch.cat([prj_colors, prj_depths, prj_weights], dim=2)

        final = self.merge_net(prj_fs)[:, :3]
        mask = (torch.sum(prj_weights, dim=1) > 0).float().detach()

        final = final * mask

        if depth and visualize:
            return final, mask, prj_colors, prj_depths

        if visualize:
            return final, mask, prj_colors  # self.out(merged_fs), warped
        
        return final
    
    def freeze(self):
        pass


class LocalSimGRU(BaseModule):
    def init_color_encoder(self):
        pass

    def init_fusion_module(self):
        self.merge_net = LocalFusion(input=4)

    def extract_src_feats(self, colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs):
        ori_shape = colors.shape[-2:]

        prj_colors, prj_depths = self.warp_all_views(
            colors, depths, ori_shape,
            self.compute_K(K, ori_shape, colors.shape[-2:]),
            src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
            radius=self.opt.model.radius,
            max_alpha=False
        )

        return prj_colors, prj_depths

    def forward(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, visualize=False, depth=False):
        with torch.no_grad():
            prj_colors, prj_depths = self.extract_src_feats(colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs)
            prj_fs = torch.cat([prj_colors, prj_depths], dim=2)

        final = self.merge_net(prj_fs)[:, :3]
        mask = (torch.sum(prj_depths, dim=1) > 0).float().detach()

        final = final * mask

        if depth and visualize:
            return final, mask, prj_colors, prj_depths

        if visualize:
            return final, mask, prj_colors  # self.out(merged_fs), warped
        
        return final
    
    def freeze(self):
        pass