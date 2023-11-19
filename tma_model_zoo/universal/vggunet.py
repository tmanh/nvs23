import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as functional
import torchvision.models as tmodels


def get_vgg_net(net):  # sourcery skip: raise-specific-error
    if net == 'vgg16':
        return tmodels.vgg16(weights=tmodels.VGG16_Weights.DEFAULT).features
    elif net == 'vgg19':
       return tmodels.vgg19(weights=tmodels.VGG19_Weights.DEFAULT).features
    else:
        raise Exception('invalid vgg net')


def create_pooling_layer(pool):  # sourcery skip: raise-specific-error
    if pool == 'average':
        enc = [nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)]
    elif pool == 'max':
        enc = [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)]
    else:
        raise Exception("invalid pool")
    
    return enc


class VGGUNet(nn.Module):
    def __init__(self, net='vgg16', pool='average', n_encoder_stages=3, n_decoder_convs=2, freeze_vgg=True):
        super().__init__()

        vgg = get_vgg_net(net)

        # self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        encs = []
        enc = []
        encs_channels = []
        channels = -1
        for mod in vgg:
            if isinstance(mod, nn.Conv2d):
                channels = mod.out_channels

            if isinstance(mod, nn.MaxPool2d):
                encs.append(nn.Sequential(*enc))
                encs_channels.append(channels)
                enc = create_pooling_layer(pool)
                n_encoder_stages -= 1
            else:
                enc.append(mod)
        self.encs = nn.ModuleList(encs)
        
        if freeze_vgg:
            for e in self.encs:
                for param in e.parameters():
                    param.requires_grad = False

        cin = encs_channels[-1] + encs_channels[-2]
        decs = []
        for idx, cout in enumerate(reversed(encs_channels[:-1])):
            decs.append(self._dec(cin, cout, n_convs=n_decoder_convs))
            cin = cout + encs_channels[max(-idx - 3, -len(encs_channels))]
        self.decs = nn.ModuleList(decs)

    def _dec(self, channels_in, channels_out, n_convs=2):
        mods = []
        for _ in range(n_convs):
            mods.extend(
                (
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(inplace=False),
                )
            )

            channels_in = channels_out
        return nn.Sequential(*mods)

    def forward(self, x):
        x = self.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        feats = []
        for enc in self.encs:
            x = enc(x)
            feats.append(x)

        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = functional.interpolate(x0, size=(x1.shape[2], x1.shape[3]), mode='nearest')
            x = torch.cat((x0, x1), dim=1)
            del x0, x1
            x = dec(x)
            feats.append(x)

        x = feats.pop()
        return x

    @staticmethod
    def normalize(tensor, mean, std, inplace=False):
        if not torch.is_tensor(tensor):
            raise TypeError(f'tensor should be a torch tensor. Got {type(tensor)}.')

        if not inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(std, dtype=dtype, device=tensor.device)

        if (std == 0).any():
            raise ValueError(f'std evaluated to zero after conversion to {dtype}, leading to division by zero.')

        if mean.ndim == 1 and tensor.ndim == 3:
            mean = mean[:, None, None]
        if mean.ndim == 1 and tensor.ndim == 4:
            mean = mean[None, :, None, None]
        
        if std.ndim == 1 and tensor.ndim == 3:
            std = std[:, None, None]
        if std.ndim == 1 and tensor.ndim == 4:
            std = std[None, :, None, None]
        
        tensor.sub_(mean).div_(std)
        return tensor


class VGGResidualUNet(nn.Module):
    def __init__(self, net='vgg16', pool='average', n_encoder_stages=3, n_decoder_convs=2, freeze_vgg=True):
        super().__init__()

        vgg = get_vgg_net(net)

        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        encs = []
        enc = []
        encs_channels = []
        channels = -1
        for mod in vgg:
            if isinstance(mod, nn.Conv2d):
                channels = mod.out_channels

            if isinstance(mod, nn.MaxPool2d):
                encs.append(nn.Sequential(*enc))
                encs_channels.append(channels)
                enc = create_pooling_layer(pool)
                n_encoder_stages -= 1
            else:
                enc.append(mod)
        self.encs = nn.ModuleList(encs)

        if freeze_vgg:
            for e in self.encs:
                for param in e.parameters():
                    param.requires_grad = False

        encs_channels = [encs_channels[0]] + encs_channels
        cin = encs_channels[-1]
        decs = []
        for cout in reversed(encs_channels[:-2]):
            decs.append(self._dec(cin, cout, n_convs=n_decoder_convs))
            cin = cout
        self.decs = nn.ModuleList(decs)

    def _dec(self, channels_in, channels_out, n_convs=2):
        mods = []
        for _ in range(n_convs):
            mods.extend(
                (
                    nn.Conv2d(
                        channels_in,
                        channels_out,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.ReLU(inplace=False),
                )
            )

            channels_in = channels_out
        return nn.Sequential(*mods)

    def forward(self, x):
        x = self.normalize(x)
        feats = []
        for enc in self.encs:
            x = enc(x)
            feats.append(x)
 
        for dec in self.decs:
            x0 = feats.pop()
            x1 = feats.pop()
            x0 = functional.interpolate(x0, size=(x1.shape[2], x1.shape[3]), mode='nearest')
            x = x0 + x1
            del x0, x1
            x = dec(x)

            feats.append(x)

        x = feats.pop()
        return x
