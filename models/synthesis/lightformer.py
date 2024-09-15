import torch
import torch.nn as nn
import torch.nn.functional as functional

from models.layers.fuse import Fusion
from models.synthesis.base import BaseModule
from models.layers.swin import SwinTransformerV2


class SwinColorFeats(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = SwinTransformerV2(window_size=8)
        self.backbone.load_pretrained()
        self.backbone.eval()

        self.pre_conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(768, 64, 3, 1, 1),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 32, 3, 1, 1),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(192, 64, 3, 1, 1),
                    nn.GELU(),
                ),
                None
            ]
        )

    def forward(self, colors):
        B, V, C, H, W = colors.shape
        with torch.no_grad():
            feats = self.backbone(colors.view(-1, C, H, W))

        hf, wf = feats[0].shape[-2:]
        merge = []
        for i, f in enumerate(feats[::-1]):
            if self.pre_conv[i] is not None:
                f = self.pre_conv[i](f)
                f = functional.interpolate(f, size=(hf, wf), mode='nearest')
            merge.append(f)
        
        return torch.cat(merge, dim=1).view(B, V, -1, hf, wf)

class LightFormer(BaseModule):
    def freeze(self):
        self.freeze_shallow_color_encoder()

    def init_color_encoder(self):
        self.encoder = SwinColorFeats()

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()

        self.merge_net = Fusion()
        self.up1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.up2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)

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

    def forward(self, batch):
        depth_loss = 0

        _, src_colors, dst_colors, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs = self.data_process(batch)
        src_depths, dst_depths, augmented_mask = self.get_raw_depth(batch, src_colors.shape[-2], src_colors.shape[-1]) # list of depths  with the shape BxHxW.
        
        B, N, C, H, W = src_colors.shape
        src_colors = src_colors.view(-1, C, H, W) # BS * num_input, C, H, W

        if src_depths is not None:
            src_depths = src_depths.view(-1, 1, H, W)
                
        if augmented_mask is not None:
            augmented_mask = augmented_mask.view(-1, 1, H, W)

        if self.opt.train_depth_only:
            src_coms, src_ests, c_feats = self.com_depth_light(src_depths * augmented_mask, src_colors)
            depth_loss = self.add_depth_loss(
                src_depths, dst_depths, src_colors,  dst_colors,
                K, src_RTs, dst_RTs, src_coms, src_ests)

            loss = {"Total Loss": depth_loss, "depth_loss": depth_loss}

            return (loss, {
                    "InputImg": src_colors,
                    "OutputImg": dst_colors.view(-1, *dst_colors.shape[2:]),
                    "PredImg": dst_colors.view(-1, *dst_colors.shape[2:]),
                    "ProjectedImg": dst_colors.view(-1, *dst_colors.shape[2:])
                },
            )
        elif self.opt.train_depth:
            src_coms, src_ests, c_feats = self.com_depth_light(src_depths * augmented_mask, src_colors)
            depth_loss = self.add_depth_loss(
                src_depths, dst_depths, src_colors,  dst_colors,
                K, src_RTs, dst_RTs, src_coms, src_ests)
        else:
            with torch.no_grad():
                src_coms, src_ests, c_feats = self.com_depth_light(src_depths * augmented_mask, src_colors)

        fs = self.encoder(src_colors, c_feats)
        fs = fs.view(B, N, -1, H, W)

        # gen_fs, warped, projected_depths = self.rendering(K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, fs, input_imgs, gt_depth)
        gen_fs, warped, projected_depths = self.rendering(K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, fs, src_colors, src_coms)
        dst_colors, gen_fs, warped = self.augment_with_feats(dst_colors, gen_fs, warped)

        projected_feats = torch.cat([gen_fs, warped], dim=2).permute(1, 0, 2, 3, 4)
        projected_depths = projected_depths.permute(1, 0, 2, 3, 4)
        refine_color = self.compute_enhanced_images(projected_feats, projected_depths, projected_feats.shape[1], dst_colors.shape[-2:])
        loss = self.loss_function(refine_color, dst_colors.view(-1, *dst_colors.shape[2:]))

        # # """
        # loss["Total Loss"] += depth_loss
        # loss["depth_loss"] = depth_loss
        # # """

        return (
            loss, {
                "InputImg": src_colors,
                "OutputImg": dst_colors.view(-1, *dst_colors.shape[2:]),
                "PredImg": refine_color.clamp(-1, 1),
                "ProjectedImg": refine_color.clamp(-1, 1)
            },
        )

    def rendering(self, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, fs, colors, regressed_pts):
        bs, nv, c, _, _ = fs.shape
        fs = fs.contiguous().view(bs, nv, c, -1)
        colors = colors.contiguous().view(bs, nv, 3, -1)
        regressed_pts = regressed_pts.contiguous().view(bs, nv, 1, -1)

        gen_fs, warped, gen_depth = self.view_render(fs, colors, regressed_pts, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs)
        return gen_fs, warped, gen_depth
    
    def eval_one(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs):
        fs = self.encoder(colors)
        shape = colors.shape
        colors, depths, prj_fs, warped, prj_depths = self.project(
            colors, depths, fs, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs
        )
        prj_depths = prj_depths.permute(1, 0, 2, 3, 4)
        prj_fs = prj_fs.permute(1, 0, 2, 3, 4)
        refined_fs = self.merge_net(
            prj_fs, prj_depths, shape[-2:]
        )

        out = self.up1(refined_fs)
        out = functional.interpolate(out, scale_factor=2, mode='nearest')
        out = self.up2(out)
        out = functional.interpolate(out, size=shape[-2:], mode='nearest')
        out = self.out(out)

        return out, warped

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
            # macs, _ = profile(self.merge_net, inputs=(projected_features[:, vidx], projected_depths[:, vidx], c_hs, d_hs))
            # print('Fusion Module 0 FLOPs: ', macs * 3)
            # exit()

            y, c_hs, d_hs = self.merge_net(
                projected_features[:, vidx], projected_depths[:, vidx], c_hs, d_hs)
            self.estimate_view_color(y, out_colors, alphas, out_size)


        return self.compute_out_color(out_colors, alphas)

    def project(self, colors, depths, feats, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs):
        bs, nv, c, hf, wf = feats.shape
        _, _, _, hc, wc = colors.shape

        sh, sw = hf / hc, wf / wc
        sK = K.clone()

        sK[:, 0, :] = sw * sK[:, 0, :]
        sK[:, 1, :] = sh * sK[:, 1, :]

        colors = functional.interpolate(
            colors.view(bs * nv, 3, hc, wc), size=(hf, wf),
            mode='bilinear', align_corners=False, antialias=True
        )
        depths = functional.interpolate(
            depths.view(bs * nv, 1, hc, wc), size=(hf, wf), mode='nearest'
        )

        feats = feats.contiguous().view(bs, nv, c, -1)
        colors = colors.contiguous().view(bs, nv, 3, -1)
        depths = depths.contiguous().view(bs, nv, 1, -1)

        prj_fs, warped, prj_depths = self.view_render(
            feats, colors, depths,
            sK, torch.inverse(sK),
            src_RTs, src_RTinvs, dst_RTs, dst_RTinvs
        )
        
        return colors, depths, prj_fs, warped, prj_depths
