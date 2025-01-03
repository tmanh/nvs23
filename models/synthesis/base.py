import numpy as np

import math
import cv2
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import one_hot

from models.layers.weight_init import trunc_normal_
from models.losses.multi_view import *

from models.projection.z_buffer_manipulator import Screen_PtsManipulator

from models.synthesis.encoder import ColorFeats

import kornia as K


def fill_holes(images, depths):
    kernel = torch.ones(3, 3, device=images.device)
    N, V, _, H, W = images.shape
    dilated_images = K.morphology.dilation(images.view(N * V, -1, H, W), kernel)
    dilated_depths = K.morphology.dilation(depths.view(N * V, -1, H, W), kernel)

    masks = (depths > 0).float()

    inpainted_imags = images.view(N, V, -1, H, W) + (1 - masks) * dilated_images
    inpainted_depths = depths.view(N, V, -1, H, W) + (1 - masks) * dilated_depths

    return inpainted_imags, inpainted_depths


def merge_reprojected_views(images, depths, bins=100):
    """
    Merges multi-view reprojected images based on mode of depth values,
    handling pixels with zero depth (invalid pixels).
    
    Args:
        images (torch.Tensor): Tensor of reprojected images, shape [B, V, C, H, W].
        depths (torch.Tensor): Tensor of reprojected depths, shape [B, V, 1, H, W].
        bins (int): Number of bins for depth discretization to compute the mode.
        
    Returns:
        torch.Tensor: Merged image, shape [B, C, H, W].
    """
    # Validate input shapes
    assert images.shape[0] == depths.shape[0], "Batch sizes must match"
    assert images.shape[1] == depths.shape[1], "Number of views must match"
    assert images.shape[-2:] == depths.shape[-2:], "Image and depth resolutions must match"
    
    batch_size, n_views, _, height, width = depths.shape
    
    # Step 1: Mask invalid depths
    valid_mask = (depths > 0).float()  # Mask where depth > 0
    masked_depths = depths * valid_mask  # Zero-out invalid depths
    
    # Step 2: Discretize valid depths into bins
    depth_min = masked_depths[valid_mask > 0].min()  # Minimum valid depth
    depth_max = masked_depths[valid_mask > 0].max()  # Maximum valid depth
    depth_bins = torch.floor((masked_depths - depth_min) / (depth_max - depth_min + 1e-8) * (bins - 1)).long()
    depth_bins = depth_bins * valid_mask.long()  # Retain invalid pixels as zeros
    
    # Step 3: Compute the mode bin
    depths_reshaped = depth_bins.view(batch_size, n_views, -1)  # Flatten spatial dimensions
    one_hot_bins = one_hot(depths_reshaped, num_classes=bins).float() * valid_mask.view(batch_size, n_views, -1, 1)
    one_hot_bins = one_hot_bins.view(batch_size, n_views, 1, height, width, bins)
    mode_bin = one_hot_bins.sum(dim=-1).argmax(dim=1)

    # Map the mode bin back to depth values
    mode_depth = mode_bin.float() / (bins - 1) * (depth_max - depth_min) + depth_min
    
    # Step 4: Calculate weights based on proximity to the mode, excluding invalid pixels
    depth_diff = torch.abs(masked_depths - mode_depth)
    weights = torch.exp(-depth_diff) * valid_mask  # Exclude invalid pixels from weights
    
    # Normalize weights
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    # masked_depths[masked_depths <= 0] = float('inf')
    # weights = (masked_depths == torch.min(masked_depths, dim=1, keepdim=True)[0]).float()
    
    # Step 5: Perform weighted merge of images
    weighted_images = images * weights  # Broadcast weights to match image channels
    merged_image = weighted_images.sum(dim=1)  # Merge views by summing
    
    return merged_image


class BaseModule(nn.Module):
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

    def __init__(self, opt):
        super().__init__()

        self.init_hyper_params(opt)
        self.init_models()
        self.freeze()

    def to_train(self):
        self.train()
        self.freeze()
        return self

    def to_eval(self):
        self.eval()
        self.freeze()
        return self

    @staticmethod
    def allocated():
        current = torch.cuda.memory_allocated(0)/1024/1024/1024
        print(f'allocated: {current} GB')
        return current

    @staticmethod
    def gain(previous):
        current = torch.cuda.memory_allocated(0)/1024/1024/1024
        print(f'allocated: {current} GB')
        print(f'gain: {current - previous} GB')

    def init_hyper_params(self, opt):
        ##### LOAD PARAMETERS
        opt.decode_in_dim = 1
        self.opt = opt

        # Use H if specifid in opt or H = W
        self.H = opt.model.H
        self.W = opt.model.W

    def init_models(self):
        self.init_color_encoder()
        self.init_fusion_module()
        self.init_renderer()
        self.init_decoder()

    def init_color_encoder(self):
        self.encoder = ColorFeats()
        self.encoder.freeze()
    
    def init_renderer(self):
        width = self.opt.model.W
        height = self.opt.model.H

        self.pts_transformer = Screen_PtsManipulator(W=width, H=height, opt=self.opt)

    def init_decoder(self):
        pass

    def init_fusion_module(self):
        pass

    def to_cuda(self, *args):
        if torch.cuda.is_available():
            new_args = [args[i].cuda() for i in range(len(args))]
        return new_args
    
    ##### FORWARD ###################################

    def compute_K(self, K, ori_shape, fs_shape):
        hc, wc = ori_shape
        hf, wf = fs_shape

        sh, sw = hf / hc, wf / wc
        sK = K.clone()

        sK[:, 0, :] = sw * sK[:, 0, :]
        sK[:, 1, :] = sh * sK[:, 1, :]

        return sK

    def forward(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, visualize=False):
        ori_shape = colors.shape[-2:]
        B, V = colors.shape[:2]

        warped = []
        with torch.no_grad():
            feats = self.encoder(colors)
            prj_feats = []
            prj_depths = []
            for fs in feats:
                fs = torch.cat(
                    [
                        fs,
                        F.interpolate(
                            colors.view(B * V, -1, *ori_shape),
                            size=fs.shape[-2:], mode='bilinear',
                            align_corners=True, antialias=True
                        ).view(B,  V, -1, *fs.shape[-2:])
                    ],
                    dim=2
                )
                prj_fs, prj_pts = self.project(
                    fs, depths, ori_shape,
                    self.compute_K(K, ori_shape, fs.shape[-2:]),
                    src_RTinvs, src_RTs, dst_RTinvs, dst_RTs
                )

                prj_feats.append(prj_fs)     # N, V, C, H, W
                prj_depths.append(prj_pts)   # N, V, C, H, W

                if visualize:
                    warped.append(prj_fs[:, :, -3:])

            # warped = None
            prj_clr, prj_pts = self.project(
                colors, depths, ori_shape,
                K,
                src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
                radius=3.0
            )
            warped.append(prj_clr)
            # warped = (warped, merge_reprojected_views(prj_clr, prj_pts))

            # N, C, V, H, W
            prjs = [torch.cat([vf, df], dim=2) for vf, df in zip(prj_feats, prj_depths)]
        
        merged_fs = self.merge_net(prjs)

        mask = (torch.sum(prj_pts, dim=1) > 0).float()

        final = self.out(merged_fs[:, :-1])
        merged_clr = merged_fs[:, -4:-1]
        merged_dpt = merged_fs[:, -1:]

        return final, mask, merged_clr, merged_dpt, warped  # self.out(merged_fs), warped

    def view_render(
            self, src_feats, src_pts,
            K, K_inv,
            src_RTs, src_RTinvs, dst_RTs, dst_RTinvs,
            H, W, radius
        ):
        prj_feats = []
        prj_depths = []

        _, V, _, _ = src_feats.shape
        for i in range(V):
            pts_3D_nv = self.pts_transformer.view_to_world_coord(
                src_pts[:, i], K, K_inv, src_RTs[:, i], src_RTinvs[:, i], H, W
            )

            src_fs = src_feats[:, i:i + 1]
            sampler = self.pts_transformer.world_to_view(
                pts_3D_nv,
                K,
                K_inv,
                dst_RTs.view(-1, 4, 4),
                dst_RTinvs.view(-1, 4, 4),
                H, W
            )
            pointcloud = sampler.permute(0, 2, 1).contiguous()
            src_fs = src_fs.view(-1, *src_fs.shape[2:])

            prj_fs, prj_ds = self.pts_transformer.splatter(
                pointcloud, src_fs, image_size=(H, W), depth=True, radius=radius
            )

            prj_depths.append(prj_ds)
            prj_feats.append(prj_fs)

        return torch.stack(prj_feats, 1), torch.stack(prj_depths, 1)

    def project(
            self, feats, depths, ori_shape, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, radius=None,
        ):
        bs, nv, c, hf, wf = feats.shape
        hc, wc = ori_shape

        depths = F.interpolate(
            depths.view(bs * nv, 1, hc, wc), size=(hf, wf), mode='nearest'
        )

        feats = feats.contiguous().view(bs, nv, c, -1)
        depths = depths.contiguous().view(bs, nv, 1, -1)

        prj_feats, prj_depths = self.view_render(
            feats, depths,
            K, torch.inverse(K),
            src_RTs, src_RTinvs, dst_RTs, dst_RTinvs,
            hf, wf,
            radius
        )

        return prj_feats, prj_depths

    def freeze(self):
        self.freeze_shallow_color_encoder()

    def freeze_shallow_color_encoder(self):
        self.encoder.freeze()

    def scale_intrinsic(self, K, oh, ow, sh, sw):
        with torch.no_grad():
            lk = K.clone()
            sx, sy = sw / ow, sh / oh

            lk[:, 0, :] = sx * lk[:, 0, :]
            lk[:, 1, :] = sy * lk[:, 1, :]

            lkinv = torch.inverse(lk)
        
        return lk, lkinv
