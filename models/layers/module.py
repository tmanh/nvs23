import torch
import torch.nn as nn
import torch.nn.functional as functional


class UpRefine(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpRefine, self).__init__()
        self.convA = ConvGELU(skip_input, output_features, kernel_size=1, stride=1, padding=0)
        self.convB = ConvGELU(output_features, output_features, kernel_size=3, stride=1, padding=1)

    def forward(self, up_x, concat_with):
        if up_x.shape[-2:] != concat_with.shape[-2:]:
            up_x = functional.interpolate(up_x, size=concat_with.shape[-2:], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))


class UpResidual(nn.Sequential):
    def __init__(self, skip_input, up_channels, output_features):
        super(UpResidual, self).__init__()
        self.skip_input = skip_input
        self.up_channels = up_channels
        self.convA = ConvGELU(skip_input, up_channels, kernel_size=3, stride=1, padding=1)
        self.convB = ConvGELU(up_channels, output_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x, concat_with):
        up_x = functional.interpolate(x, size=concat_with.shape[-2:], mode='bilinear', align_corners=True)
        up_x = self.convA(up_x)
        return self.convB(concat_with + up_x)


class ConvBnReLU3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
    ) -> None:
        """initialization method for convolution3D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            padding: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.relu(self.bn(self.conv(x)), inplace=True)


class ConvReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            bias=False, groups=groups,
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))
    

class ConvGELU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        """initialization method for convolution2D + batch normalization + relu module
        Args:
            in_channels: input channel number of convolution layer
            out_channels: output channel number of convolution layer
            kernel_size: kernel size of convolution layer
            stride: stride of convolution layer
            pad: pad of convolution layer
            dilation: dilation of convolution layer
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            bias=False, groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return functional.gelu(self.conv(x))
    

class NeckBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.residual = in_feats == out_feats
        self.convs = nn.Sequential(*[
            nn.Conv2d(in_feats, out_feats, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1, groups=out_feats),
            nn.GELU(),
            nn.Conv2d(out_feats, out_feats, kernel_size=1), nn.GELU()]
        )
    
    def forward(self, x):
        return x + self.convs(x) if self.residual else self.convs(x)
    

class SimpleNeck(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.convs = nn.ModuleList([NeckBlock(in_feats, out_feats) for in_feats, out_feats in zip(in_features, out_features)])
    
    def forward(self, xs):
        return [conv(x) for conv, x in zip(self.convs, xs)]
    

class SimilarityNet(nn.Module):
    """Similarity Net, used in Evaluation module (adaptive evaluation step)
    1. Do 1x1x1 convolution on aggregated cost [B, G, Ndepth, H, W] among all the source views,
        where G is the number of groups
    2. Perform adaptive spatial cost aggregation to get final cost (scores)
    """

    def __init__(self, G: int, num_depth: int) -> None:
        """Initialize method

        Args:
            G: the feature channels of input will be divided evenly into G groups
        """
        super(SimilarityNet, self).__init__()

        self.conv0 = ConvGELU(in_channels=G * num_depth, out_channels=4 * num_depth, kernel_size=1, stride=1, padding=0)
        self.similarity = nn.Conv2d(in_channels=4 * num_depth, out_channels=num_depth, kernel_size=1, stride=1, padding=0)

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Forward method for SimilarityNet

        Args:
            x1: [B, G, Ndepth, H, W], where G is the number of groups, aggregated cost among all the source views with
                pixel-wise view weight
            grid: position of sampling points in adaptive spatial cost aggregation, (B, evaluate_neighbors*H, W, 2)
            weight: weight of sampling points in adaptive spatial cost aggregation, combination of
                feature weight and depth weight, [B,Ndepth,1,H,W]

        Returns:
            final cost: in the shape of [B,Ndepth,H,W]
        """
        # [B,Ndepth,num_neighbors,H,W]
        return self.similarity(self.conv0(x1))
    

class PixelwiseNet(nn.Module):
    """Pixelwise Net: A simple pixel-wise view weight network, composed of 1x1x1 convolution layers
    and sigmoid nonlinearities, takes the initial set of similarities to output a number between 0 and 1 per
    pixel as estimated pixel-wise view weight.

    1. The Pixelwise Net is used in adaptive evaluation step
    2. The similarity is calculated by ref_feature and other source_features warped by differentiable_warping
    3. The learned pixel-wise view weight is estimated in the first iteration of Patchmatch and kept fixed in the
    matching cost computation.
    """

    def __init__(self, G: int) -> None:
        """Initialize method

        Args:
            G: the feature channels of input will be divided evenly into G groups
        """
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=G, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Forward method for PixelwiseNet

        Args:
            x1: pixel-wise view weight, [B, G, Ndepth, H, W], where G is the number of groups
        """
        # [B,1,H,W]
        return self.output(self.conv1(self.conv0(x1)))


def depth_regression(p: torch.Tensor, depth_values: torch.Tensor) -> torch.Tensor:
    """Implements per-pixel depth regression based upon a probability distribution per-pixel.

    The regressed depth value D(p) at pixel p is found as the expectation w.r.t. P of the hypotheses.

    Args:
        p: probability volume [B, D, H, W]
        depth_values: discrete depth values [B, D]
    Returns:
        result depth: expected value, soft argmin [B, 1, H, W]
    """

    return torch.sum(p * depth_values.view(depth_values.shape[0], 1, 1), dim=1).unsqueeze(1)


def single_warping(c_src_fea, d_src_fea, K, K_inv, src_proj, ref_proj, depth):
    b, _, h, w = c_src_fea.shape
    
    with torch.no_grad():
        proj = torch.inverse(K @ ref_proj)

        xyz = get_xyz(b, h, w, c_src_fea.device)

        # [B, 3, Ndepth, H*W]
        xyz = torch.cat([xyz * depth.view(b, 1, -1), torch.ones((b, 1, h * w), device=depth.device)], dim=1)
        xyz = K @ (src_proj @ (proj @ xyz))

        # avoid negative depth
        mask = xyz[:, 2:3] <= 1e-3
        xyz[:, 0:1][mask] = float(w)
        xyz[:, 1:2][mask] = float(h)
        xyz[:, 2:3][mask] = 1.0

        grid = xyz[:, :2, :] / xyz[:, 2:3, :]  # [B, 2, H*W]

        proj_x_normalized = grid[:, 0, :] / ((w - 1) / 2) - 1  # [B, Ndepth, H*W]
        proj_y_normalized = grid[:, 1, :] / ((h - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=-1)  # [B, Ndepth, H*W, 2]
        grid = grid.view(b, h, w, 2)

    wc_src_fea = functional.grid_sample(
        c_src_fea,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(b, -1, h, w)

    pd = xyz[:, 2:3].view(b, -1, h, w)
    pd = functional.grid_sample(
        pd,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return wc_src_fea, pd


def differentiable_warping(
    src_fea, K, K_inv, src_proj, ref_proj, depth_samples, scale=(1.0, 1.0),
):
    """Differentiable homography-based warping, implemented in Pytorch.

    Args:
        src_fea: [B, C, H, W] source features, for each source view in batch
        src_proj: [B, 4, 4] source camera projection matrix, for each source view in batch
        ref_proj: [B, 4, 4] reference camera projection matrix, for each ref view in batch
        depth_samples: [B, Ndepth, H, W] virtual depth layers
    Returns:
        warped_src_fea: [B, C, Ndepth, H, W] features on depths after perspective transformation
    """
    batch, n_depth, height, width = depth_samples.shape

    if scale[0] != 1.0 or scale[1] != 1.0:
        sK = K.clone()
        sK[:, 0:1, :3] = sK[:, 0:1, :3] * scale[0]
        sK[:, 1:2, :3] = sK[:, 1:2, :3] * scale[1]
        sK_inv = torch.inverse(sK)
    else:
        sK = K
        sK_inv = K_inv

    with torch.no_grad():
        proj = src_proj @ torch.inverse(ref_proj)
        rot = proj[:, :3, :3]     # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        xyz = get_xyz(batch, height, width, src_fea.device)

        # [B, 3, Ndepth, H*W]
        
        xyz = (rot @ (sK_inv[:, :3, :3] @ xyz)).unsqueeze(2).repeat(1, 1, n_depth, 1)
        xyz = xyz * depth_samples.view(batch, 1, n_depth, height * width)  
        xyz = xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        xyz = xyz.permute(0, 2, 1, 3)
        xyz = sK[:, :3, :3].unsqueeze(1) @ xyz
        xyz = xyz.permute(0, 2, 1, 3)

        # avoid negative depth
        mask = xyz[:, 2:] <= 1e-3
        xyz[:, 0:1][mask] = float(width)
        xyz[:, 1:2][mask] = float(height)
        xyz[:, 2:3][mask] = 1.0

        grid = xyz[:, :2, :, :] / xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        
        proj_x_normalized = grid[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
        proj_y_normalized = grid[:, 1, :, :] / ((height - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]

    return functional.grid_sample(
        src_fea,
        grid.view(batch, n_depth * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch, -1, n_depth, height, width)


def get_xyz(batch, height, width, device):
    y, x = torch.meshgrid(
        [
            torch.arange(0, height, dtype=torch.float32, device=device),
            torch.arange(0, width, dtype=torch.float32, device=device),
        ], indexing='ij'
    )
    y, x = y.contiguous().view(height * width), x.contiguous().view(height * width)
    return torch.unsqueeze(torch.stack((x, y, torch.ones_like(x))), 0).repeat(batch, 1, 1)  # [B, 3, H*W]
