from .osa import *
from .cat import ICATBlock


class FusionBlock(nn.Module):
    def __init__(self, in_dim, window_size) -> None:
        super().__init__()

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
    def __init__(self, window_size=8) -> None:
        super().__init__()

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
        self.enc1 = FusionBlock(96, window_size=window_size)
        self.enc2 = FusionBlock(192, window_size=window_size)
        self.enc3 = FusionBlock(384, window_size=window_size)
        self.enc4 = FusionBlock(768, window_size=window_size)

        self.down1 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(96, 192, 3, 2, 1)
        )
        self.down2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(192, 384, 3, 2, 1)
        )
        self.down3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(384, 768, 3, 2, 1)
        )

        self.up1 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(768 * 2, 384, 3, 1, 1)
        )
        self.up2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(384 * 2, 192, 3, 1, 1)
        )
        self.up3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(192 * 2, 96, 3, 1, 1)
        )

        self.out = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(96 * 2, 96, 3, 1, 1)
        )

    def depth_split(self, d):
        N, V, _, H, W = d.shape
        d1 = d.view(N * V, -1, H, W)
        d2 = F.interpolate(d1, size=(H // 2, W // 2), mode='nearest')
        d3 = F.interpolate(d1, size=(H // 4, W // 4), mode='nearest')
        d4 = F.interpolate(d1, size=(H // 8, W // 8), mode='nearest')

        return (
            d1.view(N, V, -1, H, W),
            d2.view(N, V, -1, H // 2, W // 2),
            d3.view(N, V, -1, H // 4, W // 4),
            d4.view(N, V, -1, H // 8, W // 8),
        )

    def forward(self, prj_feats, prj_depths):
        B, V, _, H, W = prj_feats.shape

        fs1, fs2, fs3, fs4 = self.split(prj_feats, B, V, H, W)
        ds1, ds2, ds3, ds4 = self.depth_split(prj_depths)

        fs1 = self.enc1(fs1, ds1)
        fs2 = self.enc2(fs2, ds2)
        fs3 = self.enc3(fs3, ds3)
        fs4 = self.enc4(fs4, ds4)

        dfs2 = self.down1(fs1)
        dfs3 = self.down2(dfs2)
        dfs4 = self.down3(dfs3)
        
        dfs3 = self.up1(torch.cat([dfs4, fs4], dim=1))
        dfs3 = F.interpolate(dfs3, size=fs3.shape[-2:], mode='nearest')

        dfs2 = self.up2(torch.cat([dfs3, fs3], dim=1))
        dfs2 = F.interpolate(dfs2, size=fs2.shape[-2:], mode='nearest')

        dfs1 = self.up3(torch.cat([dfs2, fs2], dim=1))
        dfs1 = F.interpolate(dfs1, size=fs1.shape[-2:], mode='nearest')

        out = fs1 + self.out(torch.cat([dfs1, fs1], dim=1))

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
