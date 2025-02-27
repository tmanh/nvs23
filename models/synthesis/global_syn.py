import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.fuse import Fusion, GlobalFusion, LocalFusion
from models.layers.weight_init import trunc_normal_
from models.synthesis.base import BaseModule
from models.synthesis.encoder import MultiScaleSwin, RadioEncoder


class GlobalGRU(BaseModule):
    def init_color_encoder(self):
        self.encoder = RadioEncoder()
        self.freeze()

    def init_fusion_module(self):
        self.merge_net = GlobalFusion(1024)

    def forward(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, visualize=False):
        feats, prj_colors, prj_feats, prj_depths = self.list_extract_src_feats(colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs)

        final = self.merge_net(feats, prj_feats, prj_depths.shape[-2:])

        if visualize:
            mask = (torch.sum(prj_depths, dim=1) > 0).float().detach()
            return final, mask, prj_colors
        return final
    
    def list_extract_src_feats(self, colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs):
        ori_shape = colors.shape[-2:]

        prj_colors, prj_depths, prj_weights = self.list_warp_all_views(
            colors, depths, ori_shape,
            self.compute_K(K, ori_shape, colors.shape[-2:]),
            src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
            radius=self.opt.model.radius,
            max_alpha=False
        )

        feats = self.encoder(prj_colors)

        src_feats = self.encoder(colors)
        prj_feats, _ = self.warp_all_views(
            src_feats, depths, ori_shape,
            self.compute_K(K, ori_shape, src_feats.shape[-2:]),
            src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
            radius=self.opt.model.radius,
            max_alpha=False
        )

        # import cv2
        # import numpy as np
        # print('xxxx')
        # s = 2
        # nc = F.interpolate(
        #     colors.view(-1, *colors.shape[2:]),
        #     size=(ori_shape[0] // s, ori_shape[1] // s),
        #     mode='bilinear',
        #     align_corners=False,
        # )
        # nc = nc.view(*colors.shape[:3], ori_shape[0] // s, ori_shape[1] // s)
        # prj, _ = self.warp_all_views(
        #     nc, depths, ori_shape,
        #     self.compute_K(K, ori_shape, nc.shape[-2:]),
        #     src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
        #     radius=self.opt.model.radius,
        #     max_alpha=False
        # )
        # print(prj.shape)
        # lw = (prj * 255.0).clamp(0, 255.0)
        # for k in range(lw.shape[1]):
        #     out = lw[0, k].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        #     cv2.imwrite(f'output/out_s_{k}.png', out)
        # exit()
 
        return feats, prj_colors, prj_feats, prj_depths
    
    def extract_src_feats(self, colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs):
        ori_shape = colors.shape[-2:]

        with torch.no_grad():
            prj_colors, prj_depths, prj_weights = self.list_warp_all_views(
                colors, depths, ori_shape,
                self.compute_K(K, ori_shape, colors.shape[-2:]),
                src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
                radius=self.opt.model.radius,
                max_alpha=False
            )

        prj_feats, _ = self.warp_all_views(
            src_feats, depths, ori_shape,
            self.compute_K(K, ori_shape, src_feats.shape[-2:]),
            src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
            radius=self.opt.model.radius,
            max_alpha=False
        )
        # feats = self.encoder(prj_colors)

        src_feats = self.encoder(colors)
        

        return prj_colors, prj_depths, prj_weights
    
    def freeze(self):
        pass