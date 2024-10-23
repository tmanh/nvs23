import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.vivim import MambaLayer


class MultiViewAttentionFusion(nn.Module):
    def __init__(self, input_channels, n_heads=8):
        super(MultiViewAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_channels, num_heads=n_heads)

    def forward(self, x):
        N, C, V, H, W = x.shape

        x = x.view(N, V, C, -1).permute(1, 0, 3, 2)    # Reshape to (V, N, HW, C)
        x = x.reshape(V, N * H * W, C)                 # Flatten the spatial dimensions
        attention_out, _ = self.attention(x, x, x)     # Self-attention over viewpoints
        attention_out = attention_out.mean(dim=0).view(N, C, H, W)  # Merge viewpoints
        
        return attention_out


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

        self.view_fuse = MultiViewAttentionFusion(96)

    def forward(self, prjs):
        prev_prj = self.enc4(prjs[-1])  # N, C, V, H, W

        list_fuses = []
        for prj, fuse, enc in zip(prjs[::-1][1:], self.fuses, self.encs):
            n, _, v, h, w = prev_prj.shape
            prev_prj = prev_prj.permute(0, 2, 1, 3, 4).view(n * v, -1, h, w)
            
            n, _, v, h, w = prj.shape
            prj = prj.permute(0, 2, 1, 3, 4).view(n * v, -1, h, w)
            prev_prj = F.interpolate(
                prev_prj, size=prj.shape[-2:],
                align_corners=True, mode='bilinear'
            )

            mf = torch.cat([prev_prj, prj], dim=1)
            prj = fuse(
                mf
            ).view(n, v, -1, h, w).permute(0, 2, 1, 3, 4)

            prev_prj = enc(
                prj
            )

            list_fuses.append(prev_prj[:, :-1])

        vfeats = list_fuses[-1]
        fused = self.view_fuse(vfeats)

        return fused, list_fuses
