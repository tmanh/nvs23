import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from models.layers.swin import SwinTransformerV2


import torch
import torchvision.models as models
import torch.nn as nn


class WideResNetMultiScale(nn.Module):
    def __init__(self):
        super(WideResNetMultiScale, self).__init__()
        # Load pretrained WideResNet50_2
        resnet50 = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        
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
    

class ConvNeXt2LargeMultiScale(nn.Module):
    def __init__(self):
        super(ConvNeXt2LargeMultiScale, self).__init__()
        # Load the pretrained ConvNeXt_Large model
        self.base_model = convnextv2_huge()
        sd = torch.load('convnextv2_huge_22k_512_ema.pt', weights_only=False)['model']
        self.base_model.load_state_dict(sd)

    def forward(self, x, B, V):
        features = []

        for i in range(4):
            x = self.base_model.downsample_layers[i](x)
            x = self.base_model.stages[i](x)
            features.append(x.view(B, V, *x.shape[1:]))

        return features


class SwinColorFeats(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.backbone = SwinTransformerV2(window_size=8)
        # self.backbone.load_pretrained()
        # self.backbone.eval()

        self.backbone = WideResNetMultiScale()

    def forward(self, colors):
        B, V, C, H, W = colors.shape
        with torch.no_grad():
            x = colors.view(B * V, C, H, W)
            features = self.backbone(x, B, V)
        return features
    
    def freeze(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
