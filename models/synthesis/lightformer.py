import time

import torch
import torch.nn as nn
import torch.nn.functional as functional

from models.synthesis.base import BaseModule


class SwinColorFeats(nn.Module):
    def forward(self, colors, cfeats):
        norm_colors = self.normalize(colors * 0.5 + 0.5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        shallows = self.encs[0](norm_colors)
        x4 = self.encs[1](shallows)

        u0 = cfeats[-1] + self.c0(cfeats[-1])

        u10 = functional.interpolate(u0, size=cfeats[-2].shape[-2:], mode='bilinear', align_corners=False)
        u11 = torch.cat([cfeats[-2], u10], dim=1)
        u12 = self.c11(u11)
        u13 = self.c12(u12)

        u20 = functional.interpolate(u13, size=cfeats[-3].shape[-2:], mode='bilinear', align_corners=False)
        u21 = torch.cat([cfeats[-3], u20], dim=1)
        u22 = self.c21(u21)
        u23 = self.c22(u22)

        u30 = functional.interpolate(u23, size=cfeats[-4].shape[-2:], mode='bilinear', align_corners=False)
        u31 = torch.cat([cfeats[-4], u30], dim=1)
        u32 = self.c31(u31)
        u33 = self.c32(u32)

        u40 = functional.interpolate(u33, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        u41 = torch.cat([x4, u40], dim=1)
        u42 = self.c41(u41)
        u43 = self.c42(u42)

        u50 = functional.interpolate(u43, size=shallows.shape[-2:], mode='bilinear', align_corners=False)
        u51 = torch.cat([shallows, u50], dim=1)
        u52 = self.c51(u51)
        return self.c52(u52)

    @staticmethod
    def normalize(tensor, mean, std, inplace=False):
        if not torch.is_tensor(tensor):
            raise TypeError(f'tensor should be a torch tensor. Got {type(tensor)}.')

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

        if (std == 0).any():
            raise ValueError(f'std evaluated to zero after conversion to {dtype}, leading to division by zero.')

        if mean.ndim == 1 and tensor.ndim == 3:
            mean = mean[:, None, None]
        if mean.ndim == 1 and tensor.ndim == 4:
            mean = mean[None, :, None, None]
        
        if std.ndim == 1 and tensor.ndim == 3:
            std = std[:, None, None]
        if std.ndim == 1 and tensor.ndim == 4:
            std = std[None, :, None, None]
        
        tensor.sub_(mean).div_(std)
        return tensor


class LightFormer(BaseModule):
    def view_render(self, src_feats, src_colors, pred_pts, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs):
        """
        We construct point cloud for each input and render them to target views individually.
        Inputs: 
            -- src: shape: BS x num_inputs x C x N. Input points features.
            -- pred_pts: BS x num_inputs x 1 x N. Input points depths.
            -- K: BS x 4 x 4. Intrinsic matrix. 
            -- src_RTs: BS x num_inputs x 4 x 4. Input camera matrixes. 
            -- dst_RTs: BS x num_outputs x 4 x 4. Target camera matrixes. 
        Outputs:
            -- rendered_images
            -- rendered_depths or None
        """
        num_inputs = self.opt.input_view_num
        num_outputs = dst_RTs.shape[1]
        results = []
        warped = []
        projected_depths = []
        _, _, _, N = src_feats.shape

        for i in range(num_inputs):
            pts_3D_nv = self.pts_transformer.view_to_world_coord(pred_pts[:, i], K, K_inv, src_RTs[:, i], src_RTinvs[:, i])

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
            src_co = src_co.view(modified_src.shape[0], 3, self.pts_transformer.H * self.pts_transformer.W)

            modified_src = torch.cat([modified_src, src_co * 0.5 + 0.5], dim=1)

            result, depth = self.pts_transformer.splatter(pointcloud, modified_src, depth=True)

            projected_depths.append(depth)
            results.append(result[:, :-3])
            warped.append(result[:, -3:])

        return torch.stack(results, 0), torch.stack(warped, 0), torch.stack(projected_depths, 0)

    def rendering(self, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, fs, colors, regressed_pts):
        bs, nv, c, _, _ = fs.shape
        fs = fs.contiguous().view(bs, nv, c, -1)
        colors = colors.contiguous().view(bs, nv, 3, -1)
        regressed_pts = regressed_pts.contiguous().view(bs, nv, 1, -1)

        gen_fs, warped, gen_depth = self.view_render(fs, colors, regressed_pts, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs)
        return gen_fs, warped, gen_depth
    
    def _eval_one_step(self, input_imgs, output_imgs, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, raw_depth):
        start = time.time()
        raw_depth, input_imgs, completed, estimated, fs = self.depth_regression(raw_depth, input_imgs)

        input_imgs, completed, gen_fs, warped, projected_depths = self.project(
            input_imgs, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, completed, fs)
        projected_masks = (projected_depths > 0).float()

        projected_depths, refine_color = self.generate_novel_view(output_imgs, gen_fs, warped, projected_depths)
        elapsed = time.time() - start

        n_view, n_batch, _, height, width = gen_fs.shape
        estimated = estimated.view(n_view, n_batch, 1, height, width).permute(1, 0, 2, 3, 4)
        completed = completed.view(n_view, n_batch, 1, height, width).permute(1, 0, 2, 3, 4)
        input_imgs = input_imgs.view(n_batch, n_view, 3, height, width)

        torch.cuda.empty_cache()
        output_dict = {
            "InputImg": input_imgs,
            "OutputImg": output_imgs.view(-1, *output_imgs.shape[2:]),
            "PredImg": refine_color,
            "ProjectedImg": refine_color,
            'Completed': estimated,
            'Estimated': estimated,
            "Warped": warped.permute(1, 0, 2, 3, 4) * 2 - 1,
            "ProjectedMasks": projected_masks.permute(1, 0, 2, 3, 4),
            "ProjectedDepths": projected_depths,
        }

        if raw_depth is not None:
            output_dict['InputDepth'] = raw_depth.view(n_view, n_batch, 1, height, width).permute(1, 0, 2, 3, 4)

        return elapsed, output_dict

    def eval_one_step(self, batch):
        _, input_imgs, output_imgs, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs = self.data_process(batch)
        raw_depth, _, _, _ = self.get_raw_depth(batch, height=input_imgs.shape[-2], width=input_imgs.shape[-1])

        return self._eval_one_step(input_imgs, output_imgs, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, raw_depth)

    def generate_novel_view(self, output_imgs, gen_fs, warped, projected_depths):
        projected_feats = torch.cat([gen_fs, warped], dim=2).permute(1, 0, 2, 3, 4)
        projected_depths = projected_depths.permute(1, 0, 2, 3, 4)
        refine_color = self.compute_enhanced_images(projected_feats, projected_depths, projected_feats.shape[1], output_imgs.shape[-2:])
        return projected_depths, refine_color

    def compute_out_color(self, colors, alphas):
        colors = torch.stack(colors)
        alphas = torch.stack(alphas)

        alphas = torch.softmax(alphas, dim=0)
        return (alphas * colors).sum(dim=0)

    def estimate_view_color(self, x, out_colors, alphas, out_size):
        x = functional.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        x = self.up_conv(x)

        out_colors.append(self.rgb_conv(x))
        alphas.append(self.alpha_conv(x))

    def compute_enhanced_images(self, projected_features, projected_depths, n_views, out_size):
        c_hs = None
        d_hs = None
        out_colors = []
        alphas = []
        for vidx in range(n_views):
            y, c_hs, d_hs = self.merge_net(projected_features[:, vidx], projected_depths[:, vidx], c_hs, d_hs)
            self.estimate_view_color(y, out_colors, alphas, out_size)


        return self.compute_out_color(out_colors, alphas)

    def project(self, input_imgs, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, completed, fs):
        bs, nv, c, _, _ = fs.shape
        
        fs = fs.contiguous().view(bs, nv, c, -1)
        input_imgs = input_imgs.contiguous().view(bs, nv, 3, -1)
        completed = completed.contiguous().view(bs, nv, 1, -1)

        gen_fs, warped, projected_depths = self.view_render(fs, input_imgs, completed, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs)
        
        return input_imgs, completed, gen_fs, warped, projected_depths

    def depth_regression(self, gt_depth, input_imgs):
        B, N, C, H, W = input_imgs.shape
        input_imgs = input_imgs.view(-1, C, H, W) # BS * num_input, C, H, W

        if gt_depth is not None:
            gt_depth = gt_depth.view(-1, 1, H, W)
                
        completed, estimated, c_feats = self.com_depth_light(gt_depth, input_imgs)

        visual_feats = self.encoder(input_imgs, c_feats)
        visual_feats = visual_feats.view(B, N, -1, H, W)

        return gt_depth, input_imgs, completed, estimated, visual_feats