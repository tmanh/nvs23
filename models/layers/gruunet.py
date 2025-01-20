import torch
import torch.nn as nn
import torch.nn.functional as F


def heinsen_associative_scan_log(log_coeffs, log_values):
    a_star = log_coeffs.cumsum(dim = 1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim = 1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()


def g(x):
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def exists(v):
    return v is not None


class ConvGRU2d(nn.Module):
    def __init__(
        self,
        channels_x,
        channels_out,
        kernel_size=3,
        padding=1,
        bias=True,
    ):
        super().__init__()
        self.channels_x = channels_x
        self.channels_out = channels_out

        self.conv_gate = nn.Conv2d(
            in_channels=channels_x,
            out_channels=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_hidden = nn.Conv2d(
            in_channels=channels_x,
            out_channels=channels_out,
            kernel_size=1,
            padding=0,
            bias=bias,
        )

    def forward(self, x, prev_h=None):
        gate = self.conv_gate(x)
        hidden = self.conv_hidden(x)

        # handle sequential (TODO: handle parallel)
        hidden = g(hidden)
        gate = gate.sigmoid()
        out = torch.lerp(prev_h, hidden, gate) if exists(prev_h) else (hidden * gate)

        # TODO: adding projection layer
        return out


class DepthConvGRU2d(nn.Module):
    def __init__(
        self,
        channels_x,
        channels_out,
        kernel_size=3,
        padding=1,
        bias=True,
    ):
        super().__init__()
        self.channels_x = channels_x
        self.channels_out = channels_out

        self.d_conv_gate = nn.Conv2d(
            in_channels=channels_x + 1,
            out_channels=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.d_conv_hidden = nn.Conv2d(
            in_channels=1,
            out_channels=channels_out,
            kernel_size=1,
            padding=0,
            bias=bias,
        )
        self.conv_hidden = nn.Conv2d(
            in_channels=channels_x,
            out_channels=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(self, x, d, prev_h=None, prev_d_h=None):
        combined = torch.cat([x, d], dim=1)

        gate = self.d_conv_gate(combined)
        d_hidden = self.d_conv_hidden(d)
        hidden = self.conv_hidden(x)

        d_hidden = g(d_hidden)
        hidden = g(hidden)
        gate = gate.sigmoid()
        
        hidden = torch.lerp(prev_h, hidden, gate) if exists(prev_h) else (hidden * gate)
        d_hidden = torch.lerp(prev_d_h, d_hidden, gate) if exists(prev_d_h) else (d_hidden * gate)

        return hidden, d_hidden


class FuseConvGRU2d(nn.Module):
    def __init__(
        self,
        channels_x,
        channels_out,
        kernel_size=3,
        padding=1,
        bias=True,
    ):
        super().__init__()
        self.channels_x = channels_x
        self.channels_out = channels_out

        self.conv_gru = DepthConvGRU2d(
            channels_x=channels_x,
            channels_out=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, d, h=None, d_h=None):
        return self.conv_gru(x, d, h, d_h)


class GRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, fuse=False):
        super().__init__()
        self.gru = ConvGRU2d(in_channels, out_channels)
        if stride == 2:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.ReLU(inplace=True),
            )
        self.fuse = None
        if fuse:
            self.fuse = nn.Sequential(
                nn.Conv2d(in_channels + out_channels, in_channels, kernel_size=1, padding=0, stride=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups=in_channels),
                nn.ReLU(inplace=True),
            )
        self.stride = stride

    def forward(self, x, h=None, prev=None):
        if self.stride == 2:
            x = self.down(x)
        if self.fuse is not None:
            x = F.interpolate(x, size=prev.shape[-2:], mode="nearest")
            x = self.fuse(torch.cat([x, prev], dim=1))
        return self.gru(x, h)


class FuseGRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.gru = FuseConvGRU2d(out_channels, out_channels)
        if stride == 2:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
                nn.ReLU(inplace=True),
            )
        self.stride = stride

    def forward(self, x, d, h=None, dh=None):
        if self.stride == 2:
            x = self.down(x)
            d = F.interpolate(d, size=x.shape[-2:], mode='nearest')
        return self.gru(x, d, h, dh)


class GRUUNet(nn.Module):
    def __init__(self, e_in_channels, e_out_channels, d_in_channels, d_out_channels):
        super().__init__()

        # Encoder
        self.enc_blocks = nn.ModuleList()
        for idx, (in_dim, out_dim) in enumerate(zip(e_in_channels, e_out_channels)):
            self.enc_blocks.append(
                FuseGRUBlock(in_dim, out_dim, stride=2 if idx > 0 else 1)
            )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(e_out_channels[-1], e_out_channels[-1], kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(e_out_channels[-1], e_out_channels[-1], kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.dec_blocks = nn.ModuleList()
        for in_dim, out_dim in zip(d_in_channels, d_out_channels):
            self.dec_blocks.append(
                GRUBlock(in_dim, out_dim, fuse=True)
            )

        self.conf = nn.Conv2d(e_in_channels[0], e_in_channels[0], kernel_size=3, padding=1, stride=1, bias=False)

    def forward(self, xs, ds):
        B, V, C, H, W = xs.shape

        ve_hs = [None for _ in self.enc_blocks]
        vd_hs = [None for _ in self.enc_blocks]
        dd_hs = [None for _ in self.enc_blocks]
        
        views = []
        confs = []
        for view in range(V):
            x = xs[:, view]
            d = ds[:, view]
            for l in range(len(self.enc_blocks)):
                x, d = self.enc_blocks[l](x, ds[:, view], ve_hs[l], dd_hs[l])
                ve_hs[l] = x
                dd_hs[l] = d

            # Bottleneck
            x = self.bottleneck(x)

            # Decoder
            for l in range(len(self.dec_blocks)):
                x = self.dec_blocks[l](x, vd_hs[l], ve_hs[len(self.dec_blocks) - l - 1])
                vd_hs[l] = x

            views.append(x)
            confs.append(self.conf(x))

        views = torch.stack(views, dim=1)
        confs = torch.softmax(torch.stack(confs, dim=1), dim=1)
        return torch.sum(views * confs, dim=1)
