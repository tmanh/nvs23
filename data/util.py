import torch
import re

import torchvision.transforms.functional as F_t
import numpy as np
import cv2

from torchvision import transforms


def detect_black(input_img, threshold = -0.95):
    return torch.bitwise_and(input_img[:, 0:1] <= threshold, torch.bitwise_and(input_img[:, 1:2] <= threshold, input_img[:, 2:3] <= threshold))


class ColorJitter(object):
    def __init__(
        self,
        hue_range=0.15,
        saturation_range=0.15,
        brightness_range=0.15,
        contrast_range=0.15,
    ):
        self.hue_range = [-hue_range, hue_range]
        self.saturation_range = [1 - saturation_range, 1 + saturation_range]
        self.brightness_range = [1 - brightness_range, 1 + brightness_range]
        self.contrast_range = [1 - contrast_range, 1 + contrast_range]

    def apply_color_jitter(self, images):
        # apply the same color jitter over batch of images
        hue_factor = np.random.uniform(*self.hue_range)
        saturation_factor = np.random.uniform(*self.saturation_range)
        brightness_factor = np.random.uniform(*self.brightness_range)
        contrast_factor = np.random.uniform(*self.contrast_range)
        for i in range(len(images)):
            tmp = (images[i] + 1.0) * 0.5
            tmp = F_t.adjust_saturation(tmp, saturation_factor)
            tmp = F_t.adjust_hue(tmp, hue_factor)
            tmp = F_t.adjust_contrast(tmp, contrast_factor)
            tmp = F_t.adjust_brightness(tmp, brightness_factor)
            images[i] = tmp * 2.0 - 1.0
        return images


def load_pfm(file):
    with open(file, 'rb') as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == 'PF':
            color = True
        elif header.decode("ascii") == 'Pf':
            color = False
        else:
            raise RuntimeError('Not a PFM file.')

        dim_match = re.match(
            r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii")
        )
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise RuntimeError('Malformed PFM header.')

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        data = np.fromfile(file, f'{endian}f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        file.close()
        return data, scale

def get_image_to_tensor_balanced(image_size=0):
    ops = []
    if image_size > 0:
        ops.append(transforms.Resize(image_size))
    ops.extend(
        [transforms.ToTensor(), transforms.Normalize([0.0], [1.0]),]
    )
    return transforms.Compose(ops)

def get_mask_to_tensor():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,))]
    )

def read_depth(path):
    depth =cv2.imread(path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    mask = np.isinf(depth)
    depth[mask] = 0.0
    return depth[:,:, 0]