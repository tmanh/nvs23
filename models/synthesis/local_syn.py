import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.layers.fuse import Fusion, LocalFusion
from models.layers.weight_init import trunc_normal_
from models.losses.sds_loss import Zero123
from models.losses.sds_utils import CustomCamera
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
        self.merge_net = LocalFusion(in_dim=4)

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


    def fit_depth(self, tgt_colors, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, visualize=False):
        import cv2
        from models.losses.sds_loss import StableDiffusion

        intrinsic = K.detach().cpu().numpy()

        # sds = StableDiffusion(colors.device)
        # sds.get_text_embeds('high quality image', 'artifact,noise')

        guidance_zero123 = Zero123(colors.device, model_key='ashawkey/zero123-xl-diffusers')
        with torch.no_grad():
            small = colors.view(-1, *colors.shape[2:]).detach()
            small = F.interpolate(small, size=(256, 256), mode='bilinear')
            guidance_zero123.get_img_embeds(small)

        source_RTs = [
            {
                "c2w": src_RTs[0, i].detach().cpu().numpy(),
                "focal_length": np.array((intrinsic[0, 0, 0], intrinsic[0, 1, 1])),
            } for i in range(src_RTs.shape[1]) 
        ]

        target_RT = {
            "c2w": dst_RTs[0, 0].detach().cpu().numpy(),
            "focal_length": np.array((intrinsic[0, 0, 0], intrinsic[0, 1, 1])),
        }

        # Normalize the depth map for better visualization
        depth_vis = depths[0, 0, 0].detach().cpu().numpy()
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())  # Normalize to [0,1]
        depth_vis = (depth_vis * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
      
        # Apply a colormap
        depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

        # Show the image
        cv2.imwrite(f"output/depth_raw.png", depth_colormap)

        # Convert depths into a trainable parameter
        depths = nn.Parameter(depths, requires_grad=True)
        
        # Define optimizer (Adam works well for depth optimization)
        optimizer = optim.Adam([depths], lr=1e-3)
        
        loss_history = []

        for i in range(1000):
            # Extract projected colors and projected depths
            prj_colors, prj_depths = self.extract_src_feats(colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs)

            # Compute color reconstruction loss
            # sds_loss = sds.compute_sds_loss(prj_colors.view(-1, *prj_colors.shape[2:])) 
            sds_loss = guidance_zero123.compute_sds_loss(
                prj_colors,
                target_RT, source_RTs,
                step_ratio=i / 1000)
            loss = 0.0001 * sds_loss # F.mse_loss(prj_colors, tgt_colors)  # L1 loss to compare colors

            # Optional: Add depth smoothness loss (TV regularization)
            eps = 1e-6  # Small value to prevent division-by-zero or NaN propagation

            # Compute depth gradients along x and y
            depth_grad_x = torch.abs(depths[:, :, :, :, :-1] - depths[:, :, :, :, 1:]) / (torch.abs(depths[:, :, :, :, :-1]) + eps)
            depth_grad_y = torch.abs(depths[:, :, :, :-1, :] - depths[:, :, :, 1:, :]) / (torch.abs(depths[:, :, :, :-1, :]) + eps)

            # Compute mean smoothness loss
            depth_smoothness_loss = depth_grad_x.mean() + depth_grad_y.mean()
            total_loss = loss #+ 0.05 * depth_smoothness_loss  # Weight the smoothness loss
            print(f"Iteration {i}: Loss = {total_loss.item():.6f}")
    
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            loss_history.append(total_loss.item())

            # Optional: Visualize progress every 100 iterations
            if visualize and i % 25 == 0:
                # Normalize the depth map for better visualization
                depth_vis = depths[0, 0, 0].detach().cpu().numpy()
                depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())  # Normalize to [0,1]
                depth_vis = (depth_vis * 255).astype(np.uint8)  # Convert to uint8 for OpenCV
                
                # Apply a colormap
                depth_colormap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)

                # Show the image
                cv2.imwrite(f"output/depth.png", depth_colormap)

                prj_colors
                lw = (prj_colors * 255.0).clamp(0, 255.0)
                for k in range(lw.shape[1]):
                    out = lw[0, k].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                    cv2.imwrite(f'output/d_{k}.png', out)
                
        print("Optimization complete!")
        
        return depths.detach()  # Return optimized depth map