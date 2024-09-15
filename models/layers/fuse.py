from .osa import *
from .cat import ICATBlock


class Fusion(nn.Module):
    def __init__(self, in_dim=256, depth_dim=16, window_size=8, dropout=0.0, with_pe=False) -> None:
        super().__init__()

        w1 = window_size
        w2 = (2 * window_size) // 3

        ########### STAGE BASE ATTENTION
        self.in_view1 = OSA_Block(in_dim, window_size=w1, dropout=dropout, with_pe=with_pe)

        self.depth_dim = depth_dim
        self.dconv = nn.Conv2d(1, depth_dim, 3, 1, 1)
        self.in_pts1 = OSA_Block(depth_dim, window_size=w1, dropout=dropout, with_pe=with_pe)

        ########### STAGE CROSS-1/SPACE-1
        self.cross1 = ICATBlock(in_dim + depth_dim, window_size=w1, num_heads=8)
        self.spatial1 = OSA_Block(in_dim + depth_dim, window_size=w1)

        ########### STAGE CROSS-2/SPACE-2
        self.cross2 = ICATBlock(in_dim + depth_dim, window_size=w2, num_heads=8)
        self.spatial2 = OSA_Block(in_dim + depth_dim, window_size=w2)

        ########### STAGE CROSS-3/SPACE-3
        self.cross3 = ICATBlock(in_dim + depth_dim, window_size=w1, num_heads=8)
        self.spatial3 = OSA_Block(in_dim + depth_dim, window_size=w1)

        ########### STAGE CROSS-4/SPACE-4
        self.cross4 = ICATBlock(in_dim + depth_dim, window_size=w2, num_heads=8)
        self.spatial4 = OSA_Block(in_dim + depth_dim, window_size=w2)

        ########### STAGE CROSS-5/SPACE-5
        self.cross5 = ICATBlock(in_dim + depth_dim, window_size=w1, num_heads=8)
        self.spatial5 = OSA_Block(in_dim + depth_dim, window_size=w1)


    def forward(self, prj_feats, prj_depths):
        B, V, C, H, W = prj_feats.shape

        ########### TO FEATURE
        d = prj_depths.view(-1, 1, H, W)
        df = self.dconv(d)
        vf = prj_feats.view(-1, C, H, W)

        ########### BASE ATTENTION
        vf = self.in_view1(vf)
        df = self.in_pts1(df)

        ########### CROSS-1/SPACE-1
        vf = vf.view(B, V, C, H, W)
        df = df.view(B, V, self.depth_dim, H, W)
        f = torch.cat([vf, df], dim=2)

        x = f[:, 0]
        y = f[:, 1:]
        x = self.cross1(x, y)
        x = self.spatial1(x)

        ########### CROSS-2/SPACE-2
        x = self.cross2(x, y)
        x = self.spatial2(x)

        ########### CROSS-3/SPACE-3
        x = self.cross3(x, y)
        x = self.spatial3(x)

        ########### CROSS-4/SPACE-4
        x = self.cross4(x, y)
        x = self.spatial4(x)

        ########### STAGE CROSS-5/SPACE-5
        x = self.cross5(x, y)
        x = self.spatial5(x)

        return x[:, :C]