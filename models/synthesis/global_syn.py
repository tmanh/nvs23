import torch

from models.layers.fuse import GlobalFusion, SNetDS2BNBase8
from models.synthesis.base import BaseModule
from models.synthesis.encoder import MultiScaleSwin
from models.synthesis.local_syn import LocalGRU


class GlobalGRU(BaseModule):
    def init_color_encoder(self):
        self.shallow = SNetDS2BNBase8(3)
        self.encoder = MultiScaleSwin()

    def init_fusion_module(self):
        self.local = LocalGRU(self.opt)
        sd = torch.load('weights/local.pt', weights_only=False)
        self.local.load_state_dict(sd)

        self.merge_net = GlobalFusion()

    def forward(self, depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, visualize=False):
        shallow, feats, prj_colors, prj_feats, prj_depths, mask, src_feats = self.list_extract_src_feats(colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs)

        final = self.merge_net(shallow, feats, prj_feats, prj_depths.shape[-2:])

        if visualize:
            mask = (torch.sum(prj_depths, dim=1) > 0).float().detach()

            if self.training:
                shallow = self.shallow(colors.view(-1, *colors.shape[2:]))
                return final, mask, prj_colors, self.merge_net.forward_diff(
                    shallow, src_feats, prj_depths.shape[-2:]
                )
            return final, mask, prj_colors
        
        return final
    
    def list_extract_src_feats(self, colors, depths, K, src_RTinvs, src_RTs, dst_RTinvs, dst_RTs):
        B, V, C, H, W = colors.shape

        with torch.no_grad():
            src_feats = self.encoder(colors.view(B * V, -1, H, W), B, V)
            
            prj_feats = []
            for i in range(len(src_feats)):
                prj_fs, _, _ = self.list_warp_all_views(
                    src_feats[i].view(B, V, *src_feats[i].shape[1:]),
                    depths, (H, W),
                    self.compute_K(K, (H, W), src_feats[i].shape[-2:]),
                    src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
                    radius=self.opt.model.radius,
                    max_alpha=False,
                    top_k=4
                ) 
                prj_feats.append(prj_fs)

            merge_color, mask, prj_colors, prj_depths = self.local(depths, colors, K, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, True, True)

        shallow = self.shallow(merge_color)
        feats = self.encoder(merge_color.view(B, -1, H, W), B, 1)

        return shallow, feats, prj_colors, prj_feats, prj_depths, mask, src_feats

    def freeze(self):
        for param in self.local.parameters():
            param.requires_grad = False
        self.local.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()