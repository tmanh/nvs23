import torch
import torch.nn as nn
import torch.nn.functional as functional

from mmcv.cnn import ConvModule


class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(UpSample, self).__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, up_x, concat_with):
        if up_x.shape[-2:] != concat_with.shape[-2:]:
            up_x = functional.interpolate(up_x, size=[concat_with.shape[-2], concat_with.shape[-1]], mode='bilinear', align_corners=True)
        return self.convB(self.convA(torch.cat([up_x, concat_with], dim=1)))


class UpSamplePlus(nn.Sequential):
    def __init__(self, skip_input, output_features, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super().__init__()
        self.convA = ConvModule(skip_input, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.convB = ConvModule(output_features, output_features, kernel_size=3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, x, concat_with):
        up_x = functional.interpolate(x, size=[concat_with.shape[-2], concat_with.shape[-1]], mode='bilinear', align_corners=True)
        return self.convB(self.convA(up_x + concat_with))
