import os
import lpips
import imageio
import argparse 
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from omegaconf import OmegaConf
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from options.options import get_dataset

from models.synthesis.lightformer import LightFormer


def peak_signal_noise_ratio_mask(image_true, image_test, mask, data_range=None):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Parameters
    ----------
    image_true : ndarray
        Ground-truth image, same shape as im_test.
    image_test : ndarray
        Test image.
    data_range : int, optional
        The data range of the input image (distance between minimum and
        maximum possible values).  By default, this is estimated from the image
        data-type.

    Returns
    -------
    psnr : float
        The PSNR metric.

    Notes
    -----
    .. versionchanged:: 0.16
        This function was renamed from ``skimage.measure.compare_psnr`` to
        ``skimage.metrics.peak_signal_noise_ratio``.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    """

    image_true, image_test = image_true.astype(np.float32), image_test.astype(np.float32)

    err = np.sum(((image_true - image_test) * mask) ** 2) / np.sum(mask)
    return 10 * np.log10((data_range ** 2) / err)


def tensor_to_image(image):
    img = torch.clamp(image.permute(1, 2, 0), min=-1, max=1).detach().cpu().numpy() * 0.5 + 0.5
    img = np.ceil(img * 255) / 255.0
    return img


class ProposedModel(nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def forward(self, input_imgs, output_imgs, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, raw_depth):
        return self.model.eval_one_step(input_imgs, output_imgs, K, K_inv, src_RTs, src_RTinvs, dst_RTs, dst_RTinvs, raw_depth)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True

    cfg = OmegaConf.load('configs/train.yaml')
    model = LightFormer(cfg).to(device)
    sd = torch.load('exp/checkpoints/0020000.pt', weights_only=False)
    model.load_state_dict(sd)

    H, W = 512, 384

    adepths = torch.tensor(np.load('wildrgb/apple_002/depths.npy'))
    adepths = adepths.unsqueeze(1).unsqueeze(0).float().cuda()

    acolors = torch.tensor(np.load('wildrgb/apple_002/colors.npy'))
    acolors = acolors.permute(0, 3, 1, 2).unsqueeze(0).float().cuda()

    aK = torch.tensor(np.load('wildrgb/apple_002/intrinsic.npy'))
    aK = aK.unsqueeze(0).float().cuda()

    aRTs = torch.tensor(np.load('wildrgb/apple_002/pose.npy'))
    aRTs = aRTs.unsqueeze(0).float().cuda()

    dst_RTs = aRTs[:, 0, :, :]
    dst_RTs = dst_RTs.view(1, 1, 4, 4)

    aRTs_inv = torch.inverse(aRTs)
    dst_RTinvs = torch.inverse(dst_RTs)

    depths = adepths[:, 1:]
    colors = acolors[:, 1:] * 2.0 - 1.0
    K = aK[:, 0]
    src_RTs = aRTs[:, 1:]
    src_RTinvs = aRTs_inv[:, 1:]

    N, V, _, oH, oW = colors.shape
    colors = F.interpolate(
        colors.view(N * V, 3, oH, oW),
        size=(H, W),
        mode='bilinear',
        align_corners=True,
        antialias=True
    ).view(N, V, 3, H, W)
    depths = F.interpolate(
        depths.view(N * V, 1, oH, oW),
        size=(H, W),
        mode='nearest'
    ).view(N, V, 1, H, W)

    K[:, 0] = W / oW * K[:, 0]
    K[:, 1] = H / oH * K[:, 1]
    
    model.eval()
    with torch.no_grad():
        syn, warped = model(
            depths,
            colors,
            K,
            
            src_RTs,
            src_RTinvs,
            
            dst_RTs, 
            dst_RTinvs,
            visualize=True,
        )

    out = F.interpolate(
        acolors[:, 0].view(1, 3, oH, oW),
        size=(H, W),
        mode='bilinear',
        align_corners=True,
        antialias=True
    ).view(N, 3, H, W)
    
    out = (out * 255.0).clamp(0, 255.0)
    out = out[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('output/out.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    syn = ((syn + 1.0) / 2.0 * 255.0).clamp(0, 255.0)
    syn = syn[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('output/syn.png', cv2.cvtColor(syn, cv2.COLOR_RGB2BGR))

    warped = ((warped + 1.0) / 2.0 * 255.0).clamp(0, 255.0)
    for k in range(warped.shape[0]):
        out = warped[k, 0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'output/out_{k}.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--train_depth_only", action="store_true", default=False)
    parser.add_argument("--name", type=str, default="results")
    parser.add_argument("--src_list", type=str, default='22 25 28')
    parser.add_argument("--input_view", type=int, default=3)
    parser.add_argument("--winsize", type=int, default=256)
    parser.add_argument("--lpips_batch_size", type=int, default=1)
    parser.add_argument("--model_type", type=str, default='mine')
    parser.add_argument("--DW", type=int, default=200)
    parser.add_argument("--DH", type=int, default=150)
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))

    main(args)
