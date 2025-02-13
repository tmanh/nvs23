import torch.nn as nn

from .architectures import ResNetDecoder, UNetDecoder64


class Decoder(nn.Module):
    def __init__(self, opt, norm="batch", channels_in=None, channels_out=3):
        super().__init__()
        self.decode_in_dim = opt.decode_in_dim if channels_in is None else channels_in
        if self.decode_in_dim != 64:
            self.shallow = nn.Sequential(nn.Conv2d(self.decode_in_dim, 64, kernel_size=1), nn.ReLU(inplace=True))

        if opt.refine_model_type == "unet":
            self.model = UNetDecoder64(opt, channels_in=64, channels_out=channels_out)
        elif "resnet" in opt.refine_model_type:
            print("RESNET decoder")
            self.model = ResNetDecoder(opt, channels_in=64, channels_out=channels_out, norm=norm)
    
    def forward(self, gen_fs):
        if self.decode_in_dim != 64:
            gen_fs = self.shallow(gen_fs)

        _, _, H, W = gen_fs.shape
        return self.model(gen_fs)


class AttentionDecoder(nn.Module):
    def __init__(self, opt, expand_ratio=2):
        super().__init__()
        decode_in_dim = opt.decode_in_dim

        self.out_conv = nn.Sequential(
            nn.BatchNorm2d(decode_in_dim),
            nn.Conv2d(decode_in_dim, decode_in_dim * expand_ratio, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(decode_in_dim * expand_ratio, 3, 1, padding=0)
        )
            
    def forward(self, gen_fs):
        return self.out_conv(gen_fs)