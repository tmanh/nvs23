import torch.nn as nn

from models.layers.adaptive_conv_cuda.adaptive_conv import AdaptiveConv
from models.layers.cat import ViewCrossAttention
from models.layers.gruunet import GRUUNet

from models.layers.legacy_fuse import *
from models.layers.minGRU.mingru import minGRU
from models.layers.upsampler import PixelShuffleUpsampler
from .osa_utils import *

import random


def create_irregular_mask(shape, device, threshold=0.5, smooth=True, kernel_size=5):
    """
    Creates a binary mask with irregular (blob-like) shapes.
    
    Args:
        shape (tuple): Shape of the tensor, e.g., (N, C, H, W)
        threshold (float): Threshold value to binarize the mask.
        smooth (bool): Whether to smooth the random tensor to get irregular shapes.
        kernel_size (int): Size of the smoothing kernel.
    
    Returns:
        mask (torch.Tensor): A binary mask of the given shape.
    """
    # Step 1: Generate an initial random tensor with values in [0,1)
    rand_tensor = torch.rand(shape, device=device)
    
    # Optionally smooth the random tensor to create smoother, irregular blobs.
    if smooth:
        # Create a simple mean filter kernel
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / (kernel_size ** 2)
        padding = kernel_size // 2
        
        # Reshape to combine batch and channel dimensions for group convolution
        N, C, H, W = rand_tensor.shape
        rand_tensor_reshaped = rand_tensor.view(N * C, 1, H, W)
        
        # Apply the convolution (smoothing)
        smoothed = F.conv2d(rand_tensor_reshaped, kernel, padding=padding)
        smoothed = smoothed.view(N, C, H, W)
    else:
        smoothed = rand_tensor
    
    # Create a binary mask using the threshold.
    mask = (smoothed > threshold).float()
    return mask


def random_noise_function(shape, device, max_value, mask_threshold=0.5, smooth_mask=True, kernel_size=5):
    """
    Generates a final noise tensor by combining an irregular mask and a noise map.
    
    Args:
        shape (tuple): Shape of the tensors (e.g., (N, C, H, W)).
        mask_threshold (float): Threshold for binarizing the mask.
        smooth_mask (bool): Whether to apply smoothing to the random mask.
        kernel_size (int): Kernel size used for smoothing the mask.
    
    Returns:
        final_noise (torch.Tensor): The resulting noise tensor.
        mask (torch.Tensor): The binary mask used.
        noise_map (torch.Tensor): The raw noise map sampled from a normal distribution.
    """
    with torch.no_grad():
        # Step 1: Create an irregular random mask.
        mask = create_irregular_mask(shape, device, threshold=mask_threshold, smooth=smooth_mask, kernel_size=kernel_size)
        
        # Step 2: Create a random noise map (e.g., diffusion model noise)
        noise_map = torch.randn(shape, device=device) * random.uniform(0.0, max_value) 
        
        # Step 3: Compute the final noise tensor using the mask.
        # Here, the noise is applied only where mask == 1.
        final_noise = mask * noise_map
    
    return final_noise, mask, noise_map


class SNetDS2BNBase8(nn.Module):
    """2D U-Net style network with batch normalization and dilated convolutions."""

    def __init__(self, in_dim, base_filter=8):
        super(SNetDS2BNBase8, self).__init__()

        # Initial Convolutions
        self.sconv0_0 = self.conv_bn(in_dim, base_filter, dilation=1)
        self.sconv0_1 = self.conv_bn(base_filter, base_filter * 2, dilation=1)
        self.sconv0_2 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=2)
        self.sconv0_3 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=1, relu=True)

        # Branch 1
        self.sconv1_2 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=3)
        self.sconv1_3 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=1, relu=True)

        # Branch 2
        self.sconv2_2 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=4)
        self.sconv2_3 = self.conv_bn(base_filter * 2, base_filter * 2, dilation=1, relu=True)

        # Concatenation & Final Conv
        self.sconv3_0 = nn.Conv2d(base_filter * 2 * 3, base_filter * 2, kernel_size=3, padding=1, bias=False)

    def conv_bn(self, in_channels, out_channels, dilation=1, relu=True):
        """Helper function for convolution + batch normalization."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial Convolutions
        x0_0 = self.sconv0_0(x)
        x0_1 = self.sconv0_1(x0_0)
        x0_2 = self.sconv0_2(x0_1)
        x0_3 = self.sconv0_3(x0_2)

        # Branch 1
        x1_2 = self.sconv1_2(x0_2)
        x1_3 = self.sconv1_3(x1_2)

        # Branch 2
        x2_2 = self.sconv2_2(x0_2)
        x2_3 = self.sconv2_3(x2_2)

        # Concatenation & Final Convolution
        x_concat = torch.cat([x0_3, x1_3, x2_3], dim=1)
        out = self.sconv3_0(x_concat)

        return out


class LocalFusion(nn.Module):
    def __init__(self, in_dim=5) -> None:
        super().__init__()

        self.radius = 3
        self.repeats = 3
        self.diameter = self.radius * 2 + 1

        self.shallow = SNetDS2BNBase8(in_dim)
        self.gru = minGRU(16)
        self.gru_back = minGRU(16)

        self.alpha = nn.Conv2d(16, 1, kernel_size=1, padding=0)


    def forward(self, prj_feats):
        B, V, C, H, W = prj_feats.shape

        # Step 1: extract features - 2D
        prj_feats = prj_feats.view(B * V, C, H, W)
        fs = self.shallow(prj_feats)

        # Step 2: extract features along different points
        fs = fs.view(B, V, -1, H, W).permute(0, 3, 4, 1, 2)
        fs = self.gru(fs)
        fs = self.gru_back(torch.flip(fs, dims=[1]))
        fs = torch.flip(fs, dims=[1])
        fs = fs.view(B, H, W, V, -1).permute(0, 3, 4, 1, 2)

        # Step 3: predict weights
        fs = fs.contiguous().view(B * V, -1, H, W)
        alpha = self.alpha(fs).view(B, V, 1, H, W)
        prj_feats = prj_feats.view(B, V, -1, H, W)
        return torch.sum(prj_feats * torch.softmax(alpha, dim=1), dim=1)


class GlobalFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fuses = nn.ModuleList(
            [
                ViewCrossAttention(96),
                ViewCrossAttention(192),
                ViewCrossAttention(384),
                ViewCrossAttention(768)
            ]
        )

        self.ups = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(768, 384 * 4, kernel_size=1, stride=1, padding=0),
                    nn.PixelShuffle(2),
                ),
                nn.Sequential(
                    nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(384, 192 * 4, kernel_size=1, stride=1, padding=0),
                    nn.PixelShuffle(2),
                ),
                nn.Sequential(
                    nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                    nn.GELU(),
                    nn.Conv2d(192, 96 * 4, kernel_size=1, stride=1, padding=0),
                    nn.PixelShuffle(2),
                ),
                PixelShuffleUpsampler(96)
            ]
        )

        self.sr = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 64 * 4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32 * 4, kernel_size=1, stride=1, padding=0),
            nn.PixelShuffle(2),
        )

        self.out = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.merge_shallow = nn.Conv2d(48, 32, kernel_size=3, padding=1)

    def forward_diff(self, shallow, feats, original_shape):
        final_noise, _, _ = random_noise_function(shallow.shape, shallow.device, max_value=0.1, kernel_size=13, mask_threshold=0.4)
        shallow = shallow + final_noise

        final_noise, _, _ = random_noise_function(feats[3].shape, feats[3].device, max_value=0.1, kernel_size=9)
        up = feats[3] + final_noise

        up = feats[2] + self.ups[0](up)[:, :, :feats[2].shape[-2], :feats[2].shape[-1]]
        final_noise, _, _ = random_noise_function(up.shape, up.device, max_value=0.1, kernel_size=5)
        up = up + final_noise

        up = feats[1] + self.ups[1](up)[:, :, :feats[1].shape[-2], :feats[1].shape[-1]]
        final_noise, _, _ = random_noise_function(up.shape, up.device, max_value=0.1, kernel_size=3, mask_threshold=0.75)
        up = up + final_noise

        up = feats[0] + self.ups[2](up)[:, :, :feats[0].shape[-2], :feats[0].shape[-1]]

        up = self.sr(up)

        if up.shape[-2:] != original_shape:
            up = F.interpolate(up, size=original_shape, mode='nearest')
    
        up = up + self.merge_shallow(torch.cat([up, shallow], dim=1))

        return self.out(up)
        

    def forward(self, shallow, feats, prj_feats, original_shape):
        new_fs = []
        for i in range(len(feats)):
            B, V, C, H, W = prj_feats[i].shape
            fs = feats[i].unsqueeze(1).permute(0, 3, 4, 1, 2).view(B * H * W, 1, C)
            pfs = prj_feats[i].permute(0, 3, 4, 1, 2).reshape(B * H * W, V, C)
            fs = (fs + self.fuses[i](fs, pfs, pfs)).view(B, H, W, C).permute(0, 3, 1, 2)
            new_fs.append(fs)

        up = self.ups[0](new_fs[3])[:, :, :new_fs[2].shape[-2], :new_fs[2].shape[-1]]
        print(up.shape, new_fs[2].shape)
        up = new_fs[2] + up
        up = new_fs[1] + self.ups[1](up)[:, :, :new_fs[1].shape[-2], :new_fs[1].shape[-1]]
        up = new_fs[0] + self.ups[2](up)[:, :, :new_fs[0].shape[-2], :new_fs[0].shape[-1]]
        up = self.sr(up)

        if up.shape[-2:] != original_shape:
            up = F.interpolate(up, size=original_shape, mode='nearest')

        up = up + self.merge_shallow(torch.cat([up, shallow], dim=1))
        out = self.out(up)

        return out


class Fusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fuse = GRUUNet(
            [64, 64, 256, 512, 1024],
            [64, 256, 512, 1024, 2048],
            [2048, 2048, 1024, 512, 256],
            [2048, 1024, 512, 256, 64]
        )

    def forward(self, prj_feats, prj_depths):
        return self.fuse(prj_feats, prj_depths)
    
    # def create_fuse_layer(self, in_dim, out_dim):
    #     return nn.Sequential(
    #         nn.Conv2d(in_dim, out_dim * 4, kernel_size=1, padding=0),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(out_dim * 4, out_dim * 4, kernel_size=3, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.Conv2d(out_dim * 4, out_dim, kernel_size=3, padding=1),
    #     )

    # def __init__(self) -> None:
    #     super().__init__()
    #     self.fuse5 = FusionInner(2048)
    #     self.fuse4 = FusionOuter(1024, 2048)
    #     self.fuse3 = FusionOuter(512, 1024)
    #     self.fuse2 = FusionOuter(256, 512)
    #     self.fuse1 = Merger(64, 256)

    # def forward(self, prj_feats, prj_src_feats, prj_depths):
    #     f5 = self.fuse5(prj_feats[-1], prj_src_feats[-1], prj_depths[-1])
    #     f4 = self.fuse4(f5, prj_feats[-2], prj_src_feats[-2], prj_depths[-2])
    #     f3 = self.fuse3(f4, prj_feats[-3], prj_src_feats[-3], prj_depths[-3])
    #     f2 = self.fuse2(f3, prj_feats[-4], prj_src_feats[-4], prj_depths[-4])
    #     f1 = self.fuse1(f2, prj_feats[-5], prj_src_feats[-5], prj_depths[-5])
    #     return f1