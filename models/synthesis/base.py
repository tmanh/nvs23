import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from models.losses.synthesis import SynthesisLoss
from models.losses.multi_view import *

from models.projection.z_buffer_manipulator import Screen_PtsManipulator

from models.synthesis.encoder import SwinColorFeats


class BaseModule(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.init_hyper_params(opt)
        self.init_models()
        self.init_loss()
        self.freeze()

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
        self.H = opt.H
        self.W = opt.W

    def init_models(self):
        self.init_color_encoder()
        self.init_fusion_module()
        self.init_renderer()

    def init_color_encoder(self):
        self.encoder = SwinColorFeats()
        self.encoder.freeze()
    
    def init_renderer(self):
        width = self.opt.W
        height = self.opt.H

        self.pts_transformer = Screen_PtsManipulator(W=width, H=height, opt=self.opt)

    def init_fusion_module(self):
        pass

    def init_loss(self):
        self.loss_function = SynthesisLoss(opt=self.opt)

    def to_cuda(self, *args):
        if torch.cuda.is_available():
            new_args = [args[i].cuda() for i in range(len(args))]
        return new_args
    
    ##### FORWARD ###################################

    def forward(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs):
        fs = self.encoder(colors)

        prj_fs, warped, prj_depths = self.project(
            colors, depths, fs, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs
        )
        prj_depths = prj_depths.permute(1, 0, 2, 3, 4)
        prj_fs = prj_fs.permute(1, 0, 2, 3, 4)
        
        refined_fs = self.merge_net(
            prj_fs, prj_depths
        )

        out = self.up1(refined_fs)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.up2(out)
        out = F.interpolate(out, size=colors.shape[-2:], mode='nearest')
        out = self.out(out)

        return out, warped

    def view_render(
            self, src_feats, src_colors, pred_pts,
            K, K_inv,
            src_RTs, src_RTinvs, dst_RTs, dst_RTinvs,
            H, W
        ):
        num_inputs = self.opt.input_view_num
        num_outputs = dst_RTs.shape[1]
        results = []
        warped = []
        projected_depths = []
        _, _, _, N = src_feats.shape

        for i in range(num_inputs):
            pts_3D_nv = self.pts_transformer.view_to_world_coord(
                pred_pts[:, i], K, K_inv, src_RTs[:, i], src_RTinvs[:, i], H, W
            )

            modified_src = src_feats[:, i:i + 1]
            src_co = src_colors[:, i:i + 1]

            sampler = self.pts_transformer.world_to_view(
                pts_3D_nv.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, N),
                K.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, 4),
                K_inv.unsqueeze(1).expand(-1, num_outputs, -1, -1).view(-1, 4, 4),
                dst_RTs.view(-1, 4, 4),
                dst_RTinvs.view(-1, 4, 4)
            )
            pointcloud = sampler.permute(0, 2, 1).contiguous()
            modified_src = modified_src.view(-1, *modified_src.shape[2:])
            src_co = src_co.view(modified_src.shape[0], 3, H * W)

            modified_src = torch.cat([modified_src, src_co * 0.5 + 0.5], dim=1)

            result, depth = self.pts_transformer.splatter(pointcloud, modified_src, depth=True)

            projected_depths.append(depth)
            results.append(result[:, :-3])
            warped.append(result[:, -3:])

        return torch.stack(results, 0), torch.stack(warped, 0), torch.stack(projected_depths, 0)

    def project(self, colors, depths, feats, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs):
        bs, nv, c, hf, wf = feats.shape
        _, _, _, hc, wc = colors.shape

        sh, sw = hf / hc, wf / wc
        sK = K.clone()

        sK[:, 0, :] = sw * sK[:, 0, :]
        sK[:, 1, :] = sh * sK[:, 1, :]

        colors = F.interpolate(
            colors.view(bs * nv, 3, hc, wc), size=(hf, wf),
            mode='bilinear', align_corners=False, antialias=True
        )
        depths = F.interpolate(
            depths.view(bs * nv, 1, hc, wc), size=(hf, wf), mode='nearest'
        )

        feats = feats.contiguous().view(bs, nv, c, -1)
        colors = colors.contiguous().view(bs, nv, 3, -1)
        depths = depths.contiguous().view(bs, nv, 1, -1)

        prj_fs, warped, prj_depths = self.view_render(
            feats, colors, depths,
            sK, torch.inverse(sK),
            src_RTs, src_RTinvs, dst_RTs, dst_RTinvs,
            hf, wf
        )
        
        return prj_fs, warped, prj_depths

    ##### DATA AUGMENTATION ###################################

    def augment(self, output_imgs, warped, rotation=False):
        if rotation:
            rotate_index = np.random.randint(0, 4)
            warped = torch.rot90(warped, k=rotate_index, dims=[-2, -1])
            output_imgs = torch.rot90(output_imgs, k=rotate_index, dims=[-2, -1])

        flip_index = np.random.randint(0, 1)
        warped = self.augment_flip(warped, flip_index)
        output_imgs = self.augment_flip(output_imgs, flip_index)

        return output_imgs, warped

    def augment_with_feats(self, output_imgs, gen_fs, warped, rotation=False):
        if rotation:
            rotate_index = np.random.randint(0, 4)
            gen_fs = torch.rot90(gen_fs, k=rotate_index, dims=[-2, -1])
            warped = torch.rot90(warped, k=rotate_index, dims=[-2, -1])
            output_imgs = torch.rot90(output_imgs, k=rotate_index, dims=[-2, -1])

        flip_index = np.random.randint(0, 1)
        warped = self.augment_flip(warped, flip_index)
        output_imgs = self.augment_flip(output_imgs, flip_index)
        gen_fs = self.augment_flip(gen_fs, flip_index)

        return output_imgs, gen_fs, warped
    
    def augment_flip(self, tensor, flip_index):
        if flip_index == 1:
            tensor = torch.flip(tensor, dims=[-1])
        elif flip_index == 2:
            tensor = torch.flip(tensor, dims=[-2])
        elif flip_index == 3:
            tensor = torch.flip(tensor, dims=[-2, -1])
        
        return tensor
    
    ##### DATA AUGMENTATION ###################################
    
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

    def apply_basic_loss(self, gt_depth, regressed_pts):
        depth_loss = 0

        with torch.no_grad():
            valid = gt_depth > 0.0

        depth_loss += F.l1_loss(regressed_pts[valid], gt_depth[valid])
        depth_loss += F.mse_loss(regressed_pts[valid], gt_depth[valid])

        return depth_loss