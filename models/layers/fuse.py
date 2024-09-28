from .osa import *
from .cat import ICATBlock


class FusionBlock(nn.Module):
    def __init__(self, in_dim, window_size) -> None:
        w1 = window_size

        self.cross = ICATBlock(in_dim, (in_dim + 1) * 2, 1, window_size=w1, num_heads=8)
        self.spatial = LOSABlock(in_dim, window_size=w1)

    def forward(self, vf, df):
        x = vf[:, 0]
        y = vf[:, 1:]
        xm = df[:, 0]
        ym = df[:, 1:]

        x = self.cross(x, y, xm, ym)
        x = self.spatial(x)

        return x


class Fusion(nn.Module):
    def __init__(self, in_dim=256, depth_dim=16, window_size=8, dropout=0.0, with_pe=False) -> None:
        super().__init__()

        w1 = window_size

        self.prj_2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(256, 192, 3, 1, 1)
        )

        self.prj_3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(512, 384, 3, 1, 1)
        )

        self.prj_4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4096, 768, 3, 1, 1)
        )

        ########### STAGE BASE ATTENTION
        self.cross = ICATBlock(in_dim, (in_dim + 1) * 2, 1, window_size=w1, num_heads=8)
        self.spatial = LOSABlock(in_dim, window_size=w1)

    def forward(self, prj_feats, prj_depths):
        B, V, C, H, W = prj_feats.shape

        fs1, fs2, fs3, fs4 = self.split(prj_feats, B, V, H, W)

        print(prj_feats.shape, fs1.shape, fs2.shape, fs3.shape, fs4.shape)
        exit()

        ########### TO FEATURE
        d = prj_depths.contiguous().view(-1, 1, H, W)
        vf = prj_feats.contiguous().view(-1, C, H, W)

        ########### CROSS/SPACE
        vf = vf.contiguous().view(B, V, C, H, W)
        df = d.contiguous().view(B, V, 1, H, W)

        x = vf[:, 0]
        y = vf[:, 1:]
        xm = df[:, 0]
        ym = df[:, 1:]

        for _ in range(5):
            x = self.cross(x, y, xm, ym)
            x = self.spatial(x)

        return x[:, :C]

    def split(self, prj_feats, B, V, H, W):
        fs1 = prj_feats[:, :, :96]
        fs2 = prj_feats[:, :, 96:160].view(
            B, V, 64, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B * V, 256, H // 2, W // 2)
        fs3 = prj_feats[:, :, 160:192].view(
            B, V, 32, H // 4, 4, W // 4, 4).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B * V, 512, H // 4, W // 4)
        fs4 = prj_feats[:, :, 192:].view(
            B, V, 64, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B * V, 4096, H // 8, W // 8)
            
        fs2 = self.prj_2(fs2).view(B, V, 192, H // 2, W // 2)
        fs3 = self.prj_3(fs2).view(B, V, 384, H // 2, W // 2)
        fs4 = self.prj_4(fs2).view(B, V, 786, H // 2, W // 2)

        return fs1, fs2, fs3, fs4
