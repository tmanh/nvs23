import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fwd.transformer import Image_Fusion_Transformer
from models.fwd.architectures import ResNetEncoder
from models.fwd.decoder import Decoder

from types import SimpleNamespace

from models.synthesis.base import BaseModule


class FWD(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        opt = SimpleNamespace()
        opt.ngf = 64
        opt.decode_in_dim = 64
        opt.norm_G = 'sync:spectral_batch'
        opt.refine_model_type ='resnet_256W8customup'
        opt.atten_n_head = 4
        opt.atten_k_dim = 16
        opt.atten_v_dim = 64
        opt.decoder_norm = 'instance'
        opt.model = cfg.model

        self.opt = opt

        self.encoder = ResNetEncoder(opt, channels_in=3, channels_out=64, downsample=False, norm=opt.norm_G)

        out_dim = 64
        in_dim = 32 + 64
        self.vd_1 = nn.Sequential(nn.Linear(4, 16), nn.ReLU(inplace=True), nn.Linear(16, 32))
        self.vd_2 = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, out_dim))
        
        self.fusion_module = Image_Fusion_Transformer(opt)
        self.decoder = Decoder(opt, norm=opt.decoder_norm)

    def forward(
        self, depths, colors, K,
        src_RTs, src_RTinvs,
        dst_RTs, dst_RTinvs,
        visualize=None
    ):
        B, N, C, H, W = colors.shape
        colors = colors.view(-1, C, H, W)

        self.opt.input_view_num = N

        fs = self.encoder(colors)
        fs = torch.cat([fs, colors * 0.5 + 0.5], axis=1)
        fs = fs.view(B, N, -1, H, W)

        ori_shape = H, W
        prj_fs, prj_pts = self.warp_all_views(
            fs, depths, ori_shape,
            self.compute_K(K, ori_shape, fs.shape[-2:]),
            src_RTinvs, src_RTs, dst_RTinvs, dst_RTs,
            radius=self.opt.model.radius,
            max_alpha=self.opt.model.fradius # False if i == 0 else self.opt.model.fradius
        )

        B, V, C, H, W = prj_fs.shape
        gen_fs = self.fusion_module(prj_fs.permute(1, 0, 2, 3, 4))
        out = self.decoder(gen_fs)

        return out
