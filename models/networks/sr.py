import torch
import torch.nn as nn
import torch.nn.functional as functional

from tma_model_zoo.universal import Resnet


class SuperResolution(nn.Module):
    def __init__(self, in_dim=3, n_resblock=4, act=nn.GELU(), mode='neural'):
        super().__init__()

        self.encoder = Resnet(in_dim=in_dim, n_feats=48, kernel_size=3, n_resblock=n_resblock, out_dim=48, tail=False)
        self.reduce_dim = nn.Sequential(nn.Conv2d(in_channels=48, out_channels=24, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        self.extend_dim = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=24, kernel_size=1, stride=1, padding=0), nn.ReLU(inplace=True))
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1, stride=1, padding=0), act,
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=1, stride=1, padding=0))
        self.decoder = Resnet(in_dim=48, n_feats=48, kernel_size=3, n_resblock=n_resblock, out_dim=in_dim, tail=True)

        self.map = None
        self.fracion = None
        self.mode = mode

    def project_bilinear(self, x, out_shape):
        return functional.interpolate(x, size=out_shape, mode='bilinear')

    def project(self, x, out_shape):
        if self.map is None or self.map.shape != out_shape:
            self.compute_mapping(x.shape[-2:], out_shape, x.device)

        x_reduced = self.reduce_dim(x)

        f_extended = self.extend_dim(self.fracion)

        upscale_x_reduced = x_reduced[:, :, self.map[0], self.map[1]]
        upscale_x_reduced = torch.cat([upscale_x_reduced, f_extended.repeat(upscale_x_reduced.shape[0], 1, 1, 1)], dim=1)

        modification = self.projector(upscale_x_reduced)

        return modification * x[:, :, self.map[0], self.map[1]]

    def compute_mapping(self, in_shape, out_shape, device):
        y, x = torch.meshgrid([torch.arange(out_shape[0]), torch.arange(out_shape[1])], indexing='ij')

        scale_y, scale_x = out_shape[0] / in_shape[0], out_shape[1] / in_shape[1]

        y = y / scale_y
        x = x / scale_x

        y_int = y.long()
        x_int = x.long()

        y_frac = y - y_int
        x_frac = x - x_int
            
        self.map = torch.stack([y_int, x_int]).to(device)
        self.fracion = torch.stack([y_frac, x_frac]).unsqueeze(0).to(device)

    def forward(self, x, out_shape):  # sourcery skip: class-extract-method
        x = self.encoder(x)
        if self.mode == 'neural':
            x = self.project(x, out_shape)
        else:
            x = self.project_bilinear(x, out_shape)
        return self.decoder(x)

    def test(self, x, out_shape):
        with torch.no_grad():
            x = self.encoder(x)
            if self.mode == 'neural':
                x = self.project(x, out_shape)
            else:
                x = self.project_bilinear(x, out_shape)
            return self.decoder(x)
