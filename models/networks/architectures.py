import torch
import torchvision

import torch.nn as nn
from models.layers.normalization import BatchNorm_StandingStats
    

def get_conv_layer(opt, use_3D=False):
    if "spectral" in opt.norm_G:
        return (lambda in_c, out_c, k, p, s: nn.utils.spectral_norm(nn.Conv3d(in_c, out_c, k, p, s))) if use_3D else (lambda in_c, out_c, k, p, s: nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, k, p, s)))
    elif use_3D:
        return lambda in_c, out_c, k, p, s: nn.Conv3d(in_c, out_c, k, p, s)
    else:
        return lambda in_c, out_c, k, p, s: nn.Conv2d(in_c, out_c, k, p, s)


def get_batchnorm_layer(opt):
    norm_G = opt.norm_G.split(":")[1]
    if norm_G in ["batch", "spectral_batch"]:
        norm_layer = nn.BatchNorm2d
    elif norm_G == "spectral_batchstanding":
        norm_layer = BatchNorm_StandingStats
    elif norm_G == "spectral_instance":
        norm_layer = nn.InstanceNorm2d
    return norm_layer


# VGG architecture, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
        ).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # Normalize the image so that it is in the appropriate range
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]