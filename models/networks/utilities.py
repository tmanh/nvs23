import torch.nn as nn
from models.networks.architectures import (
    ResNetDecoder,
    ResNetEncoder,
    UNetDecoder64,
    UNetEncoder64,
)
EPS = 1e-2


def get_encoder(opt, downsample=True):
    if opt.refine_model_type == "unet":
        print("UNet encoder")
        encoder = UNetEncoder64(opt, channels_in=3, channels_out=64)
    elif "resnet" in opt.refine_model_type:
        print("ResNet encoder")
        encoder = ResNetEncoder(opt, channels_in=3, channels_out=64, downsample=downsample, norm=opt.norm_G)
    return encoder


def get_decoder(opt):
    if opt.refine_model_type == "unet":
        decoder = UNetDecoder64(opt, channels_in=64, channels_out=3)
    elif "resnet" in opt.refine_model_type:
        print("RESNET decoder")
        decoder = ResNetDecoder(opt, channels_in=64, channels_out=3)

    return decoder
