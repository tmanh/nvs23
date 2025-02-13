import torch
import torch.nn as nn
import torch.nn.functional as F

from models.synthesis.base import BaseModule


class DeepBlending(nn.Module):
    def __init__(self, in_dim=64, n_view=2) -> None:
        super().__init__()

        self.conv0   = nn.Conv2d(in_dim + 1, 128, kernel_size=3, stride=2, padding=1)
        self.conv1   = nn.Conv2d(128 * n_view, 256, kernel_size=3, stride=2, padding=1)
        self.conv2   = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv3   = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv4   = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1)
        self.neck    = nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1)
        self.uconv1  = nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1)
        self.uconv2  = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.uconv3  = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.uconv4  = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.uconv5  = nn.Conv2d(128, n_view, kernel_size=3, stride=1, padding=1)

    def forward(self, prj_fs, prj_depths):
        fuse = torch.cat([prj_fs, prj_depths], dim=2)

        N, V, C, H, W = fuse.shape
        fuse = fuse.view(N * V, C, H, W)

        x = self.conv0(fuse)
        x = x.view(N, -1, *x.shape[-2:])
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.neck(x)
        x = self.uconv1(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.uconv2(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.uconv3(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.uconv4(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.uconv5(F.interpolate(x, size=(H, W), mode='nearest'))

        alpha = F.softmax(x, dim=1).view(N, V, 1, H, W)

        return torch.sum(prj_fs * alpha, dim=1)


class DeepBlendingPlus(BaseModule):
    def freeze(self):
        self.encoder.freeze()

    def init_fusion_module(self):
        self.merge_net = DeepBlending()
        self.up1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.up2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)
