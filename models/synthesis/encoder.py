import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from models.layers.swin import SwinTransformerV2


import torch
import torchvision.models as models
import torch.nn as nn


class WideResNetMultiScale(nn.Module):
    def __init__(self, depth=False):
        super(WideResNetMultiScale, self).__init__()
        # Load pretrained WideResNet50_2
        if not depth:
            resnet50 = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        else:
            resnet50 = models.wide_resnet50_2()
            resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Extract layers for different scales
        self.conv1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
        )  # Scale 1
        self.act_maxpool = nn.Sequential(
            resnet50.relu,
            resnet50.maxpool
        )
        self.layer1 = resnet50.layer1  # Scale 2
        self.layer2 = resnet50.layer2  # Scale 3
        self.layer3 = resnet50.layer3  # Scale 4
        self.layer4 = resnet50.layer4  # Scale 5

    def forward(self, x, B, V):
        # Extract multi-scale features
        features = []
        x = self.conv1(x)
        features.append(x)
        x = self.act_maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features


# Modify the model to expose intermediate layers for multi-scale features
class MultiScaleResNet50(nn.Module):
    def __init__(self):
        super(MultiScaleResNet50, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Extract layers for different scales
        self.conv1 = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
        )  # Scale 1
        self.act_maxpool = nn.Sequential(
            resnet50.relu,
            resnet50.maxpool
        )
        self.layer1 = resnet50.layer1  # Scale 2
        self.layer2 = resnet50.layer2  # Scale 3
        self.layer3 = resnet50.layer3  # Scale 4
        self.layer4 = resnet50.layer4  # Scale 5

    def forward(self, x, B, V):
        # Extract multi-scale features
        features = []
        x = self.conv1(x)
        features.append(x.view(B, V, *x.shape[1:]))
        x = self.act_maxpool(x)
        x = self.layer1(x)
        features.append(x.view(B, V, *x.shape[1:]))
        x = self.layer2(x)
        features.append(x.view(B, V, *x.shape[1:]))
        x = self.layer3(x)
        features.append(x.view(B, V, *x.shape[1:]))
        x = self.layer4(x)
        features.append(x.view(B, V, *x.shape[1:]))
        return features
    

class ConvNeXtLargeMultiScale(nn.Module):
    def __init__(self):
        super(ConvNeXtLargeMultiScale, self).__init__()
        # Load the pretrained ConvNeXt_Large model
        base_model = models.convnext_large(models.ConvNeXt_Large_Weights.IMAGENET1K_V1)

        # Extract the stages from the model
        self.stage1 = base_model.features[0]  # First stage
        self.stage2 = base_model.features[1]  # Second stage
        self.stage3 = base_model.features[2]  # Third stage
        self.stage4 = base_model.features[3]  # Fourth stage

    def forward(self, x, B, V):
        features = []

        # Pass through each stage and collect features
        x = self.stage1(x)
        features.append(x.view(B, V, *x.shape[1:]))

        x = self.stage2(x)
        features.append(x.view(B, V, *x.shape[1:]))

        x = self.stage3(x)
        features.append(x.view(B, V, *x.shape[1:]))

        x = self.stage4(x)
        features.append(x.view(B, V, *x.shape[1:]))

        return features


class ColorFeats(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.backbone = SwinTransformerV2(window_size=8)
        # self.backbone.load_pretrained()
        # self.backbone.eval()

        self.backbone = WideResNetMultiScale(depth=False)

        # Define decoder layers with PixelShuffle
        self.u1 = nn.Sequential(
            nn.Conv2d(2048, 1024 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.u2 = nn.Sequential(
            nn.Conv2d(1024 * 2, 512 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.u3 = nn.Sequential(
            nn.Conv2d(512 * 2, 256 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.u4 = nn.Sequential(
            nn.Conv2d(256 * 2, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.u5 = nn.Sequential(
            nn.Conv2d(64 * 2, 64 * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, colors):
        B, V, C, H, W = colors.shape
        with torch.no_grad():
            x = colors.view(B * V, C, H, W)
            feats = self.backbone(x, B, V)
        
        u1 = self.u1(feats[4])
        u2 = self.u2(torch.cat([feats[3], u1[:, :, :feats[3].shape[-2], :feats[3].shape[-1]]], dim=1))
        u3 = self.u3(torch.cat([feats[2], u2[:, :, :feats[2].shape[-2], :feats[2].shape[-1]]], dim=1))
        u4 = self.u4(torch.cat([feats[1], u3[:, :, :feats[1].shape[-2], :feats[1].shape[-1]]], dim=1))
        u5 = self.u5(torch.cat([feats[0], u4[:, :, :feats[0].shape[-2], :feats[0].shape[-1]]], dim=1))

        return u5.view(B, V, -1, H, W)
    
    def freeze(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()


class RadioEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
        model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True)
        model.cuda().eval()

    def forward(self, x):
        nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
        x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
        with torch.no_grad():
            _, spatial_features = self.model(x, feature_fmt='NCHW')
        return spatial_features
    
    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()