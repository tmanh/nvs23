import numpy as np

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.multi_view import *

from models.projection.z_buffer_manipulator import Screen_PtsManipulator

from models.synthesis.encoder import SwinColorFeats


class BaseModule(nn.Module):
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

    def init_color_encoder(self):
        self.encoder = SwinColorFeats()
        self.encoder.freeze()
    
    def init_renderer(self):
        width = self.opt.model.W
        height = self.opt.model.H

        self.pts_transformer = Screen_PtsManipulator(W=width, H=height, opt=self.opt)

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

    def forward(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, visualize=False, py=-1, px=-1, ps=-1):
        ori_shape = colors.shape[-2:]

        with torch.no_grad():
            feats = self.encoder(colors)
            
            prj_feats = []
            prj_depths = []
            for fs in feats:
                prj_fs, prj_pts = self.project(
                    fs, depths, ori_shape,
                    self.compute_K(K, ori_shape, colors.shape[-2:]),
                    src_RTinvs, src_RTs, dst_RTinvs, dst_RTs
                )
                prj_feats.append(prj_fs)     # V, N, C, H, W
                prj_depths.append(prj_pts)   # V, N, C, H, W

            prjs = [torch.cat([vf, df], dim=2) for vf, df in zip(prj_feats, prj_depths)]
        
        refined_fs = self.merge_net(
            prjs
        )

        out = self.up1(refined_fs)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.up2(out)
        out = F.interpolate(out, size=shape, mode='nearest')
        out = self.out(out)

        return out, warped
    
    def forward_train(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, py=-1, px=-1, ps=-1):
        shape = colors.shape[-2:]
        fs = self.encoder(colors)

        prj_fs, warped, prj_depths = self.project(
            colors, depths, fs, K,
            src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
            False
        )
        prj_depths = prj_depths.permute(1, 0, 2, 3, 4)
        prj_fs = prj_fs.permute(1, 0, 2, 3, 4)
        
        if ps > 0:
            prj_depths = prj_depths[
                ...,
                py:py + ps,
                px:px + ps
            ]
            prj_fs = prj_fs[
                ...,
                py:py + ps,
                px:px + ps
            ]

        refined_fs = self.merge_net(
            prj_fs, prj_depths
        )

        N, V, _, H, W = fs.shape
        fs = fs[:, :, :96].view(N * V, 96, H, W)
        fs = fs[
            ...,
            py:py + ps,
            px:px + ps
        ]

        out = self.decode(ps, refined_fs)
        raw = self.decode(ps, fs).view(N, V, 3, ps, ps)
        return out, raw
    
    def forward_stage1(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, py=-1, px=-1, ps=-1):
        fs = self.encoder(colors)

        N, V, _, H, W = fs.shape
        fs = fs[:, :, :96].view(N * V, 96, H, W)
        if ps != -1:
            fs = fs[
                ...,
                py:py + ps,
                px:px + ps
            ]
        else:
            ps = (H, W)

        return self.decode(ps, fs)

    def decode(self, shape, refined_fs):
        out = self.up1(refined_fs)
        out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.up2(out)
        out = F.interpolate(out, size=shape, mode='nearest')
        out = self.out(out)
        return out

    def view_render(
            self, src_feats, src_pts,
            K, K_inv,
            src_RTs, src_RTinvs, dst_RTs, dst_RTinvs,
            H, W
        ):
        prj_feats = []
        prj_depths = []
        _, V, _, N = src_feats.shape
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
                dst_RTinvs.view(-1, 4, 4)
            )
            pointcloud = sampler.permute(0, 2, 1).contiguous()
            src_fs = src_fs.view(-1, *src_fs.shape[2:])

            prj_fs, prj_ds = self.pts_transformer.splatter(
                pointcloud, src_fs, image_size=(H, W), depth=True
            )

            prj_depths.append(prj_ds)
            prj_feats.append(prj_fs)

        return torch.stack(prj_feats, 0), torch.stack(prj_depths, 0)

    def project(
            self, feats, depths, ori_shape, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs,
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
            hf, wf
        )

        return prj_feats, prj_depths

    ##### DATA AUGMENTATION ###################################

    def augment(self, feats, depths, rotation=False):
        if rotation:
            rotate_index = np.random.randint(0, 4)
            depths = torch.rot90(depths, k=rotate_index, dims=[-2, -1])
            feats = torch.rot90(feats, k=rotate_index, dims=[-2, -1])

        flip_index = np.random.randint(0, 1)
        feats = self.augment_flip(feats, flip_index)
        depths = self.augment_flip(depths, flip_index)

        return feats, depths

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
