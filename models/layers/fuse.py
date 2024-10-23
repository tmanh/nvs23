import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.vivim import MambaLayer


class Fusion(nn.Module):
    def create_fuse_layer(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 4, out_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim * 4, out_dim, kernel_size=3, padding=1),
        )

    def __init__(self) -> None:
        super().__init__()

        self.enc1 = MambaLayer(96 + 1)
        self.enc2 = MambaLayer(192 + 1)
        self.enc3 = MambaLayer(384 + 1)
        self.enc4 = MambaLayer(768 + 1)

        self.fuse2 = self.create_fuse_layer(96 + 1 + 192 + 1, 96 + 1)
        self.fuse3 = self.create_fuse_layer(192 + 1 + 384 + 1, 192 + 1)
        self.fuse4 = self.create_fuse_layer(384 + 1 + 768 + 1, 384 + 1)

        self.fuses = nn.ModuleList([self.fuse4, self.fuse3, self.fuse2])
        self.encs = nn.ModuleList([self.enc3, self.enc2, self.enc1])

    def forward(self, prjs):
        print(prjs[-1].shape)
        prev_prj = self.enc4(prjs[-1])  # N, C, V, H, W
        for prj, fuse, enc in zip(prjs[::-1][1:], self.fuses, self.encs):
            prev_prj = F.interpolate(
                prev_prj, size=prj.shape[-2:],
                align_corners=True, mode='bilinear'
            )

            n, _, v, h, w = prev_prj.shape
            mf = torch.cat(
                [prev_prj, prj], dim=1).permute(0, 2, 1, 3, 4
            ).view(n * v, -1, h, w)
            prj = fuse(
                mf
            ).view(n, v, -1, h, w).permute(0, 2, 1, 3, 4)

            prev_prj = enc(
                prj
            )
        print(prev_prj.shape)
        exit()

        return out

    def split(self, prj_feats, B, V, H, W):
        fs1 = prj_feats[:, :, :96]
        fs2 = prj_feats[:, :, 96:160].view(
            B, V, 64, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B * V, 256, H // 2, W // 2)
        fs3 = prj_feats[:, :, 160:192].view(
            B, V, 32, H // 4, 4, W // 4, 4).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B * V, 512, H // 4, W // 4)
        fs4 = prj_feats[:, :, 192:].view(
            B, V, 64, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B * V, 4096, H // 8, W // 8)
            
        fs2 = self.prj_2(fs2).view(B, V, 192, H // 2, W // 2)
        fs3 = self.prj_3(fs3).view(B, V, 384, H // 4, W // 4)
        fs4 = self.prj_4(fs4).view(B, V, 768, H // 8, W // 8)

        return fs1, fs2, fs3, fs4
