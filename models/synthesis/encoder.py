import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.swin import SwinTransformerV2


class SwinColorFeats(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = SwinTransformerV2(window_size=8)
        self.backbone.load_pretrained()
        self.backbone.eval()

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
        B, V, C, H, W = colors.shape
        with torch.no_grad():
            feats = self.backbone(colors.view(-1, C, H, W))

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