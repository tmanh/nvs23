import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from models.layers.swin import SwinTransformerV2


class SwinColorFeats(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.backbone = SwinTransformerV2(window_size=8)
        # self.backbone.load_pretrained()
        # self.backbone.eval()

        self.backbone = models.vgg19(pretrained=True).features
        self.backbone.eval()

        self.selected_layers = [3, 8, 17, 26, 35]  # Correspond to different conv/pool layers in VGG19

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.pre_conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(768, 64, 3, 1, 1),
                ),
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(384, 32, 3, 1, 1),
                ),
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(192, 64, 3, 1, 1),
                ),
                None
            ]
        )

    def forward(self, colors):
        colors = (colors - self.mean) / self.std

        B, V, C, H, W = colors.shape
        with torch.no_grad():
            features = []
    
        x = colors.view(-1, C, H, W)
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.selected_layers:
                feats.append(x)
        for f in feats:
            print(f.shape)
        exit()
        feats = self.backbone()

        hf, wf = feats[0].shape[-2:]
        merge = []
        for i, f in enumerate(feats[::-1]):
            if self.pre_conv[i] is not None:
                f = self.pre_conv[i](f)
                f = F.interpolate(f, size=(hf, wf), mode='nearest')
            merge.append(f)
        
        return torch.cat(merge, dim=1).view(B, V, -1, hf, wf)
    
    def freeze(self):
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
        self.std.eval()
        self.mean.eval()