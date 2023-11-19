import math

import torch
import torch.nn as nn
import torch.nn.functional as functional


def convbn(in_channels, out_channels, kernel_size=3,stride=1, padding=1):
    return nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
		nn.BatchNorm2d(out_channels)
	)


def average_by_kernel(kernel, weight):
    kernel_size = int(math.sqrt(kernel.size()[1]))
    return functional.conv2d(kernel, weight, stride=1, padding=(kernel_size-1) // 2)


def generate_average_kernel(kernel_size=3):
    encoder = torch.zeros(kernel_size**2, kernel_size**2, kernel_size, kernel_size).cuda()
    kernel_range_list = list(range(kernel_size - 1, -1, -1))
    
    ls = []
    for _ in range(kernel_size):
        ls.extend(kernel_range_list)

    index = [list(range(kernel_size**2 - 1, -1, -1)), list(range(kernel_size**2)), [val for val in kernel_range_list for j in range(kernel_size)], ls]
    encoder[index] = 1
    return nn.Parameter(encoder, requires_grad=False)



class ColorDiff(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1), nn.GELU())
        self.conv2x1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1), nn.GELU())
        self.conv2x2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2, dilation=2), nn.GELU())
        self.conv2x4 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=4, dilation=4), nn.GELU())

        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0), nn.GELU())
    
    def forward(self, x):
        x = self.conv1(x)

        x1 = self.conv2x1(x)
        x2 = self.conv2x2(x)
        x4 = self.conv2x4(x)

        return self.conv3(x1 + x2 + x4)


class CSPNGuidanceAccelerate(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.generate = convbn(in_channels, self.kernel_size * self.kernel_size, kernel_size=3, stride=1, padding=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature):
        return self.softmax(self.generate(feature))


class CSPNAccelerate(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=1, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, guidance_kernel, modified_data, source_data):  # with standard CSPN, an addition input0 port is added
        bs, c, h, w = modified_data.shape

        # STEP 1: reshape
        input_im2col = functional.unfold(source_data, self.kernel_size, self.dilation, self.padding, self.stride)   # N x (K x K) x (H x W)
        input_im2col = input_im2col.view(bs, c, self.kernel_size ** 2, -1)
        guidance_kernel = guidance_kernel.reshape(bs, 1, self.kernel_size * self.kernel_size, h * w)                      # N x (K x K) x (H x W)
        modified_data = modified_data.view(bs, c, 1, h * w)                                                                 # N x 1 x (H x W)

        # STEP 2: reinforce the source data back to the modified data
        mid_index = int((self.kernel_size * self.kernel_size - 1)/ 2)
        input_im2col[:, :, mid_index:mid_index+1, :] = modified_data

        # STEP 3: weighted average based on guidance kernel
        output = torch.einsum('ijkt,ijkt->ijt', (input_im2col, guidance_kernel.repeat(1, c, 1, 1)))
        # output = torch.sum(input_im2col * guidance_kernel, dim=2)
        return output.view(bs, c, h, w)


class CSPNFusion(nn.Module):
    def __init__(self, n_feats=64, n_reps=6):
        super().__init__()

        self.n_reps = n_reps

        self.kernel_conf_layer = convbn(n_feats, 2)
        self.gating_layer = convbn(n_feats, 1)
        self.gating_view_layer = convbn(n_feats, 1)

        self.iter_guide_layer3 = CSPNGuidanceAccelerate(n_feats, 3)
        self.average_kernel3 = generate_average_kernel(kernel_size=3)
        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, guidance_feats, deep_image, projected, projected_masks, n, v):
        h, w = guidance_feats.shape[-2:]
        modified_3 = projected

        gating, gating_view = self.compute_conf_map(guidance_feats, n, v, h, w)
        gating_view = gating_view.view(-1, 1, h, w) * projected_masks
        projected_mask = (torch.sum(projected_masks.view(n, v, -1, h, w), dim=1) > 0).float()
        gating = gating * projected_mask

        guide3 = self.iter_guide_layer3(guidance_feats)
        for _ in range(self.n_reps):
            modified_3 = self.CSPN3(guide3, modified_3, projected)

        modified = modified_3.view(n, v, -1, h, w)
        gating_view = gating_view.view(n, v, -1, h, w)

        modified = torch.sum((modified * gating_view), dim=1) / (torch.sum(gating_view, dim=1) + 1e-7)

        return (modified * gating + deep_image * (1 - gating)) * 2.0 - 1.0, modified * 2.0 - 1.0

    def compute_conf_map(self, guidance_feats, n, v, h, w):
        gating = torch.sigmoid(self.gating_layer(torch.mean(guidance_feats.view(n, v, -1, h, w), dim=1)))
        gating_view = torch.sigmoid(self.gating_view_layer(guidance_feats).view(n, v, -1, h, w))
        
        return gating, gating_view


class CSPNFusionOld(nn.Module):
    def __init__(self, n_feats=64, n_reps=3):
        super().__init__()

        self.n_reps = n_reps

        self.kernel_conf_layer = convbn(n_feats, 2)
        self.gating_layer = convbn(n_feats, 1)
        self.gating_view_layer = convbn(n_feats, 1)

        self.iter_guide_layer3 = CSPNGuidanceAccelerate(n_feats, 3)
        self.iter_guide_layer5 = CSPNGuidanceAccelerate(n_feats, 5)

        self.average_kernel3 = generate_average_kernel(kernel_size=3)
        self.average_kernel5 = generate_average_kernel(kernel_size=5)

        self.CSPN3 = CSPNAccelerate(kernel_size=3, dilation=1, padding=1, stride=1)
        self.CSPN5 = CSPNAccelerate(kernel_size=5, dilation=1, padding=2, stride=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, guidance_feats, deep_image, projected, projected_masks, n, v):
        h, w = guidance_feats.shape[-2:]
        modified_3 = modified_5 = projected

        gating_view, kernel_conf3, kernel_conf5 = self.compute_conf_map(guidance_feats, n, v, h, w)

        guide3 = self.iter_guide_layer3(guidance_feats)
        guide5 = self.iter_guide_layer5(guidance_feats)

        guide3 = average_by_kernel(guide3, self.average_kernel3)
        guide5 = average_by_kernel(guide5, self.average_kernel5)
        
        image_extend = deep_image.unsqueeze(1).repeat(1, v, 1, 1, 1).view(-1, 3, h, w)
        gating_view = gating_view.view(-1, 1, h, w) * projected_masks

        for _ in range(self.n_reps):
            modified_3 = self.CSPN3(guide3, modified_3, projected)
            modified_3 = (1 - gating_view) * image_extend + gating_view * modified_3
            modified_5 = self.CSPN5(guide5, modified_5, projected)
            modified_5 = (1 - gating_view) * image_extend + gating_view * modified_5

        modified = kernel_conf3 * modified_3 + kernel_conf5 * modified_5

        return torch.sum((modified.view(n, v, -1, h, w) * self.softmax(gating_view.view(n, v, -1, h, w))), dim=1)

    def compute_conf_map(self, guidance_feats, n, v, h, w):
        gating_view = torch.sigmoid(self.gating_view_layer(guidance_feats).view(n, v, -1, h, w))

        kernel_conf = self.softmax(self.kernel_conf_layer(guidance_feats))
        kernel_conf3 = kernel_conf[:, 0:1, :, :]
        kernel_conf5 = kernel_conf[:, 1:2, :, :]
        
        return gating_view, kernel_conf3, kernel_conf5


class DeformFuser(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_feats = 32

        self.color_diff_conv = ColorDiff()

        self.feat_diff_conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0), nn.GELU(),
                                            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0), nn.GELU())

        self.base_conv = nn.Sequential(nn.Conv2d(96, 128, kernel_size=1, stride=1, padding=0), nn.GELU(),
                                       nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0), nn.GELU())
        
        self.fusion = CSPNFusion(32)

    def forward(self, diff_feats, deep_color, projected_colors, projected_masks):
        projected_colors = projected_colors.permute(1, 0, 2, 3, 4)
        projected_masks = projected_masks.permute(1, 0, 2, 3, 4)

        # diff_feats: N x V x C x H x W
        n, v, c, ih, iw = diff_feats.shape
        oh, ow = deep_color.shape[-2:]

        diff_feats, extended_deep_color, projected_colors, projected_masks = self.scale_inputs(
            diff_feats, deep_color, projected_colors, projected_masks, v, c, ih, iw, oh, ow)

        diff_colors = self.color_diff_conv(torch.cat([extended_deep_color, projected_colors], dim=1))
        diff = torch.cat([diff_feats, diff_colors], dim=1)
        guidance = self.base_conv(diff)

        return self.fusion(guidance, deep_color, projected_colors, projected_masks, n, v)

    def scale_inputs(self, diff_feats, deep_color, projected_colors, projected_masks, v, c, ih, iw, oh, ow):
        projected_colors = projected_colors.contiguous().view(-1, 3, ih, iw)
        projected_colors = functional.interpolate(projected_colors, size=(oh, ow), mode='bilinear', align_corners=False)

        projected_masks = projected_masks.contiguous().view(-1, 1, ih, iw)
        projected_masks = functional.interpolate(projected_masks, size=(oh, ow), mode='bilinear', align_corners=False)

        diff_feats = diff_feats.contiguous().view(-1, c, ih, iw)
        diff_feats = functional.interpolate(diff_feats, size=(oh, ow), mode='bilinear', align_corners=False)
        diff_feats = self.feat_diff_conv(diff_feats)

        extended_deep_color = deep_color.unsqueeze(1).repeat(1, v, 1, 1, 1).reshape(-1, 3, oh, ow)
        return diff_feats, extended_deep_color, projected_colors, projected_masks


class DeformFuserLegacy(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_feats = 32
        self.n_neighbors = 4

        self.color_diff_conv = ColorDiff()

        self.feat_diff_conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0), nn.GELU(),
                                            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0), nn.GELU())

        self.base_conv = nn.Sequential(nn.Conv2d(96, 128, kernel_size=1, stride=1, padding=0), nn.GELU(),
                                       nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0), nn.GELU())
        
        self.s_conv = nn.Sequential(nn.Conv2d(32, self.n_neighbors * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                    nn.BatchNorm2d(self.n_neighbors * 2))
        
        self.nc_conv = nn.Sequential(nn.Conv2d(32, self.n_neighbors, kernel_size=3, stride=1, padding=1, bias=False), nn.Softmax(dim=1))
        self.vc_conv = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False), nn.Sigmoid())
        self.tc_conv = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False), nn.Sigmoid())

    def forward(self, diff_feats, deep_color, projected_colors):
        projected_colors = projected_colors.permute(1, 0, 2, 3, 4)

        # diff_feats: N x V x C x H x W
        n, v, c, ih, iw = diff_feats.shape
        oh, ow = deep_color.shape[-2:]

        diff_feats, extended_deep_color, projected_colors = self.scale_inputs(diff_feats, deep_color, projected_colors, v, c, ih, iw, oh, ow)

        diff_colors = self.color_diff_conv(torch.cat([extended_deep_color, projected_colors], dim=1))
        diff = torch.cat([diff_feats, diff_colors], dim=1)
        base = self.base_conv(diff)

        sampling = self.compute_base_sampling(n, v, oh, ow, diff.device)
        sampling = sampling + self.s_conv(base).view(-1, 2, oh, ow).permute(0, 2, 3, 1)

        n_conf = self.nc_conv(base).view(n, v, self.n_neighbors, -1, oh, ow)
        v_conf = self.vc_conv(base).view(n, v, -1, oh, ow)
        t_conf = self.tc_conv(torch.mean(base.view(n, v, -1, oh, ow), dim=1))

        projected_colors = projected_colors.unsqueeze(1).repeat(1, self.n_neighbors, 1, 1, 1).view(-1, 3, oh, ow)
        deform_projected_colors = functional.grid_sample(projected_colors, sampling, mode='bilinear', align_corners=False)
        deform_projected_colors = deform_projected_colors.view(n, v, self.n_neighbors, -1, oh, ow)
        deform_colors = torch.sum(deform_projected_colors * n_conf, dim=2) / torch.sum(n_conf, dim=2)
        deform_color = torch.sum(deform_colors * v_conf, dim=1) / torch.sum(v_conf, dim=1)

        return t_conf * deform_color + deep_color * (1 - t_conf), deform_color

    def scale_inputs(self, diff_feats, deep_color, projected_colors, v, c, ih, iw, oh, ow):
        projected_colors = projected_colors.contiguous().view(-1, 3, ih, iw)
        projected_colors = functional.interpolate(projected_colors, size=(oh, ow), mode='bilinear', align_corners=False)

        diff_feats = diff_feats.contiguous().view(-1, c, ih, iw)
        diff_feats = functional.interpolate(diff_feats, size=(oh, ow), mode='bilinear', align_corners=False)
        diff_feats = self.feat_diff_conv(diff_feats)

        extended_deep_color = deep_color.unsqueeze(1).repeat(1, v, 1, 1, 1).reshape(-1, 3, oh, ow)
        return diff_feats, extended_deep_color, projected_colors

    def compute_base_sampling(self, n, v, height, width, device):
        ys = torch.linspace(-1, 1, height, device=device)
        xs = torch.linspace(-1, 1, width, device=device)
        ym, xm = torch.meshgrid([ys, xs], indexing='ij')
        sampling = torch.cat([xm.unsqueeze(-1).unsqueeze(0), ym.unsqueeze(-1).unsqueeze(0)], dim=-1)
        sampling = sampling.repeat(n * v * self.n_neighbors, 1, 1, 1)
        return sampling