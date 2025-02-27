import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from models.layers.swin import SwinTransformerV2
from utils.hubconf import radio_model

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
    

class MultiScaleEfficientNetV2(nn.Module):
    def __init__(self):
        super(MultiScaleEfficientNetV2, self).__init__()
        # Load the pretrained EfficientNetV2_S model
        effnet = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        
        # EfficientNetV2_S has a features attribute (a Sequential of blocks).
        # We can split these blocks into stages to extract multi-scale features.
        # The splits below are one possible grouping. You might need to adjust these based on your needs.
        self.stage0 = nn.Sequential(effnet.features[0])            # e.g., initial stem block
        self.stage1 = nn.Sequential(*effnet.features[1:3])           # next 2 blocks
        self.stage2 = nn.Sequential(*effnet.features[3:5])           # next 2 blocks
        self.stage3 = nn.Sequential(*effnet.features[5:8])           # next 3 blocks
        self.stage4 = nn.Sequential(*effnet.features[8:])            # remaining blocks

    def forward(self, x, B, V):
        features = []
        
        # Stage 0: initial stem
        x0 = self.stage0(x)
        features.append(x0.view(B, V, *x0.shape[1:]))
        
        # Stage 1
        x1 = self.stage1(x0)
        features.append(x1.view(B, V, *x1.shape[1:]))
        
        # Stage 2
        x2 = self.stage2(x1)
        features.append(x2.view(B, V, *x2.shape[1:]))
        
        # Stage 3
        x3 = self.stage3(x2)
        features.append(x3.view(B, V, *x3.shape[1:]))
        
        # Stage 4
        x4 = self.stage4(x3)
        features.append(x4.view(B, V, *x4.shape[1:]))
        
        return features
    

class MultiScaleSwin(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone = SwinTransformerV2(window_size=8)
        self.backbone.load_pretrained()
        self.backbone.eval()

    def forward(self, x, B, V):
        feats = self.backbone(x)
        return feats
        
    def freeze(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()


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
        self.model = radio_model(version='radio_v2.5-l').cuda().eval()

    def forward(self, x):
        with torch.no_grad():
            B, V, C, H, W = x.shape
            x = x.view(B * V, -1, H, W)

            nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
            x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
            _, spatial_features = self.model(x, feature_fmt='NCHW')
            
            H, W = spatial_features.shape[-2:]
            spatial_features = spatial_features.view(B, V, -1, H, W)
            
        return spatial_features
    
    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()