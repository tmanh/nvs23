from .osa import *
from .cat import ICATBlock


class Fusion(nn.Module):
    def __init__(self, in_dim=256, depth_dim=16, window_size=8, dropout=0.0, with_pe=False) -> None:
        super().__init__()

        w1 = window_size

        ########### STAGE BASE ATTENTION
        self.in_view1 = OSA_Block(in_dim, window_size=w1, dropout=dropout, with_pe=with_pe)

        self.depth_dim = depth_dim
        self.dconv = nn.Conv2d(1, depth_dim, 3, 1, 1)
        self.in_pts1 = OSA_Block(depth_dim, window_size=w1, dropout=dropout, with_pe=with_pe)

        self.cross = ICATBlock(in_dim, (in_dim + depth_dim) * 2, depth_dim, window_size=w1, num_heads=8)
        self.spatial = LOSABlock(in_dim, window_size=w1)

    def forward(self, prj_feats, prj_depths):
        B, V, C, H, W = prj_feats.shape

        fs1 = prj_feats[:, :, :96]
        fs2 = prj_feats[:, :, 96:160].view(
            B, V, 64, H // 2, 2, W // 2, 2).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B, V, 256, H // 2, W // 2)
        fs3 = prj_feats[:, :, 160:192].view(
            B, V, 32, H // 4, 4, W // 4, 4).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B, V, 512, H // 4, W // 4)
        fs4 = prj_feats[:, :, 192:].view(
            B, V, 64, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 6, 3, 5).contiguous().view(B, V, 1024, H // 8, W // 8)

        print(prj_feats.shape, fs1.shape, fs2.shape, fs3.shape, fs4.shape)
        exit()

        ########### TO FEATURE
        d = prj_depths.contiguous().view(-1, 1, H, W)
        df = self.dconv(d)
        vf = prj_feats.contiguous().view(-1, C, H, W)

        ########### BASE ATTENTION
        vf = self.in_view1(vf)
        df = self.in_pts1(df)

        ########### CROSS/SPACE
        vf = vf.contiguous().view(B, V, C, H, W)
        df = df.contiguous().view(B, V, self.depth_dim, H, W)
        f = torch.cat([vf, df], dim=2)

        x = vf[:, 0]
        y = vf[:, 1:]
        xm = df[:, 0]
        ym = df[:, 1:]

        for _ in range(5):
            x = self.cross(x, y, xm, ym)
            x = self.spatial(x)

        return x[:, :C]
