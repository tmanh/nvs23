from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

import sys
import torch
from utils.hubconf import radio_model


if __name__ == "__main__":
    #model_version="radio_v2.5-g" # for RADIOv2.5-g model (ViT-H/14)
    # model_version="radio_v2.5-h" # for RADIOv2.5-H model (ViT-H/16)
    model_version="radio_v2.5-l" # for RADIOv2.5-L model (ViT-L/16)
    #model_version="radio_v2.5-b" # for RADIOv2.5-B model (ViT-B/16)
    #model_version="e-radio_v2" # for E-RADIO
    model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True)
    model.cuda().eval()

    x = Image.open('dtu_down_4/DTU/Rectified/scan1/image/000007.png').convert('RGB')
    x = pil_to_tensor(x).to(dtype=torch.float32, device='cuda')
    x.div_(255.0)  # RADIO expects the input values to be between 0 and 1
    x = x.unsqueeze(0) # Add a batch dimension

    nearest_res = model.get_nearest_supported_resolution(*x.shape[-2:])
    x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)

    # RADIO expects the input to have values between [0, 1]. It will automatically normalize them to have mean 0 std 1.
    # summary, spatial_features = model(x)
    model = radio_model(version='radio_v2.5-l').cuda()
    with torch.no_grad():
        summary, spatial_features = model(x, feature_fmt='NCHW')
    # x = torch.rand(1, 3, 224, 224, device='cuda')
    print(spatial_features.shape)