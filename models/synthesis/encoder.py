import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from models.layers.swin import SwinTransformerV2


class SwinColorFeats(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = SwinTransformerV2(window_size=8)
        self.backbone.load_pretrained()
        self.backbone.eval()

    def forward(self, colors):
        B, V, C, H, W = colors.shape
        with torch.no_grad():
            feats = self.backbone(colors.view(B * V, C, H, W))
            reshaped_feats = []
            for f in feats:
                _, c, h, w = f.shape
                reshaped_feats.append(f.view(B, V, c, h, w))
        return reshaped_feats
    
    def freeze(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
