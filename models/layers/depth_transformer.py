import torch
import torch.nn as nn
import torch.nn.functional as functional

from tma_model_zoo.universal import *
from tma_model_zoo.enhancement import *


class NeckBlock(nn.Module):
    def forward(self, x):
        return x + self.convs(x) if self.residual else self.convs(x)


class SimpleNeck(nn.Module):
    def forward(self, xs):
        return [conv(x) for conv, x in zip(self.convs, xs)]


class DenseDecoder(nn.Module):
    def forward(self, inputs):
        list_feats = []
        for index, feat in enumerate(inputs):
            if index == 0:
                temp_feat = self.conv_list[index](feat)
            else:
                skip_feat = feat
                temp_feat = self.conv_list[index](list_feats[-1], skip_feat)

            list_feats.append(temp_feat)

        return self.depth_pred(list_feats[-1]), list_feats


class FusionModule(nn.Module):
    def forward(self, color_feats, depth_feats):
        if color_feats.shape[-1] < depth_feats.shape[-1] or color_feats.shape[-2] < depth_feats.shape[-2]:
            color_feats = functional.pad(color_feats, (0, depth_feats.shape[-1] - color_feats.shape[-1], 0, depth_feats.shape[-2] - color_feats.shape[-2]))
        df = torch.cat([color_feats, depth_feats], dim=1)
        att = self.sigmoid(self.dconv1b(self.dconv1a(df)))
        return self.dconv2c(self.dconv2b(self.dconv2a(color_feats.contiguous() * att + depth_feats * (self.one - att))))


class LightDepthTransformer(nn.Module):
    def estimate(self, color):
        color_feats = [self.shallow_bb(color), *self.backbone(color)]
        neck_feats = self.neck(color_feats)
        depth, up_feats = self.decode(neck_feats[::-1])

        output = self.sigmoid(depth)

        return output, up_feats, color_feats[1:]

    def forward(self, color, depth):
        estimated, c_feats, swin_feats = self.estimate(color)
        estimated = functional.interpolate(estimated, size=color.shape[-2:], mode='bilinear', align_corners=False)

        if depth is None:
            return None, estimated, swin_feats

        f0 = self.shallow(depth)

        f1 = self.conv1(f0)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        fn = self.conv6(f5)

        d5 = self.dconv51(torch.cat([fn, f5], dim=1))
        dm5 = self.dconv52(c_feats[0], d5)
        dm5 = functional.interpolate(dm5, size=f4.shape[-2:], mode='bilinear', align_corners=False)
        
        d4 = self.dconv41(torch.cat([dm5, f4], dim=1))
        dm4 = self.dconv42(c_feats[1], d4)
        dm4 = functional.interpolate(dm4, size=f3.shape[-2:], mode='bilinear', align_corners=False)

        d3 = self.dconv31(torch.cat([dm4, f3], dim=1))
        dm3 = self.dconv32(c_feats[2], d3)
        dm3 = functional.interpolate(dm3, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        d2 = self.dconv21(torch.cat([dm3, f2], dim=1))
        dm2 = self.dconv22(c_feats[3], d2)
        dm2 = functional.interpolate(dm2, size=f1.shape[-2:], mode='bilinear', align_corners=False)

        d1 = self.dconv11(torch.cat([dm2, f1], dim=1))
        dm1 = self.dconv12(c_feats[4], d1)
        dm1 = functional.interpolate(dm1, size=f0.shape[-2:], mode='bilinear', align_corners=False)

        completed = self.sigmoid(self.out_conv(dm1))

        if self.opt.dataset != 'scannet':
            mask = (depth > self.eps).float()
            completed = completed * (1 - mask) + depth * mask

        return completed, estimated, swin_feats