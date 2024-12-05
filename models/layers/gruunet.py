import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRU2d(nn.Module):
    def __init__(
        self,
        channels_x,
        channels_out,
        kernel_size=3,
        padding=1,
        nonlinearity="tanh",
        bias=True,
    ):
        super().__init__()
        self.channels_x = channels_x
        self.channels_out = channels_out

        self.conv_gates = nn.Conv2d(
            in_channels=channels_x + channels_out,
            out_channels=2 * channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_can = nn.Conv2d(
            in_channels=channels_x + channels_out,
            out_channels=channels_out,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        if nonlinearity == "tanh":
            self.nonlinearity = nn.Tanh()
        elif nonlinearity == "relu":
            self.nonlinearity = nn.ReLU()
        else:
            raise Exception("invalid nonlinearity")

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(
                (x.shape[0], self.channels_out, x.shape[2], x.shape[3]),
                dtype=x.dtype,
                device=x.device,
            )
        combined = torch.cat([x, h], dim=1)
        combined_conv = torch.sigmoid(self.conv_gates(combined))
        del combined
        r = combined_conv[:, : self.channels_out]
        z = combined_conv[:, self.channels_out :]

        combined = torch.cat([x, r * h], dim=1)
        n = self.nonlinearity(self.conv_can(combined))
        del combined

        h = z * h + (1 - z) * n
        return h


class GRUUNet(nn.Module):
    def __init__(
        self,
        in_channels=[2048 + 4, 1024 + 4, 512 + 4, 256 + 4, 64 + 4],
        dec_channels=[1024 + 4, 512 + 4, 256 + 4, 64 + 4],
        n_dec_convs=2,
        gru_all=False,
        gru_nonlinearity="relu",
    ):
        super().__init__()
        self.n_rnn = 0
        self.gru_nonlinearity = gru_nonlinearity

        cin = in_channels[0] + in_channels[1]
        decs = []
        for idx, cout in enumerate(dec_channels):
            decs.append(
                self._dec(cin, cout, n_convs=n_dec_convs, gru_all=gru_all)
            )
            cin = cout + in_channels[min(idx + 2, len(in_channels) - 1)]
        self.decs = nn.ModuleList(decs)

    def _dec(self, channels_in, channels_out, n_convs=2, gru_all=False):
        mods = []
        for idx in range(n_convs):
            if gru_all or idx == n_convs - 1:
                self.n_rnn += 1
                mods.append(
                    ConvGRU2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                        nonlinearity=self.gru_nonlinearity,
                    )
                )
            else:
                mods.append(
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    )
                )
                mods.append(nn.ReLU())
            channels_in = channels_out
        return nn.Sequential(*mods)

    def forward(self, xs, hs=None):
        if hs is None:
            hs = [None for _ in range(self.n_rnn)]

        feats = []
        hidx = 0
        for dec in self.decs:
            x0 = xs.pop()
            x1 = xs.pop()
            
            _, _, _, h0, w0 = x0.shape
            B, V, _, h1, w1 = x1.shape

            x0 = x0.view(B * V, -1, h0, w0)
            x1 = x1.view(B * V, -1, h1, w1)

            x0 = F.interpolate(
                x0, size=(h1, w1), mode="nearest"
            )
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            for mod in dec:
                if isinstance(mod, ConvGRU2d):
                    x = mod(x, hs[hidx])
                    hs[hidx] = x
                    hidx += 1
                else:
                    x = mod(x)
            xs.append(x.view(B, V, -1, h1, w1))

        x = xs.pop()

        return x, hs