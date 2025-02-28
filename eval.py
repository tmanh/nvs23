import os
import argparse 
import numpy as np
from omegaconf import OmegaConf

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.util import load_pfm
from models.synthesis.fwd import FWD
from models.synthesis.deepblendplus import DeepBlendingPlus
from models.synthesis.local_syn import LocalGRU, LocalSimGRU
from models.synthesis.global_syn import GlobalGRU


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

    path = 'datasets/dtu_down_4/DTU/Rectified/scan21/image'
    dpath = 'datasets/dtu_down_4/Depths_2/scan21'
    im_names = ['000029.png', '000022.png', '000025.png', '000028.png']  # , '000028.png'
    indices = [29, 22, 25, 28]

    camera = np.load('datasets/dtu_down_4/camera.npy', allow_pickle=True)

    acolors = []
    adepths = []
    aK = []
    aRTs = []
    scale = 1000
    for i, idx in zip(im_names, indices):
        img = cv2.imread(os.path.join(path, i))
        img = torch.tensor(img) / 255.0# * 2 - 1
        img = img.permute(2, 0, 1).unsqueeze(0)

        dep, _ = load_pfm(os.path.join(dpath, i.replace('0000', '000000').replace('png', 'pfm')))
        dep = torch.tensor(dep.copy()) / scale
        dep = dep.unsqueeze(0).unsqueeze(0)
        dep = F.interpolate(dep, size=(300, 400), mode='nearest')

        Rt, k = camera[idx]
        Rt[:3, 3] = Rt[:3, 3] / scale
        K = np.eye(4, 4)
        K[:3, :3] = k
        aK.append(
            torch.tensor(K).unsqueeze(0)
        )
        aRTs.append(
            torch.inverse(torch.tensor(Rt).unsqueeze(0))
        )
        acolors.append(img)
        adepths.append(dep)

    acolors = torch.stack(acolors, dim=1).float().cuda()
    adepths = torch.stack(adepths, dim=1).float().cuda()
    aK = torch.stack(aK, dim=1).float().cuda()
    aRTs = torch.stack(aRTs, dim=1).float().cuda()

    H, W = 300, 400

    dst_RTs = aRTs[:, 0, :, :]
    dst_RTs = dst_RTs.view(1, 1, 4, 4)

    aRTs_inv = torch.inverse(aRTs)
    dst_RTinvs = torch.inverse(dst_RTs)

    K = aK[:, 0]
    depths = adepths[:, 1:]
    colors = acolors[:, 1:]
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
    
    ## "/home/antruong/anaconda3/envs/render/lib/python3.10/site-packages/torch/nn/modules/module.py", line 2215
    cfg = OmegaConf.load('configs/train.yaml')
    # model = FWD(cfg).to(device)
    # sd = torch.load('weights/fwd.pt', weights_only=False)
    # model = DeepBlendingPlus(cfg).to(device)
    # sd = torch.load('weights/deepblend.pt', weights_only=False)
    # model = LocalGRU(cfg).to(device)
    # sd = torch.load('weights/local.pt', weights_only=False)
    model = GlobalGRU(cfg).to(device)
    sd = torch.load('weights/global2.pt', weights_only=False)
    # model = LocalSimGRU(cfg).to(device)
    # sd = torch.load('weights/local_sim.pt', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        out, _, warped = model(
            depths,
            colors,
            K,
            
            src_RTs,
            src_RTinvs,
            
            dst_RTs, 
            dst_RTinvs,
            visualize=True,
        )

    gt = F.interpolate(
        acolors[:, 0].view(1, 3, oH, oW),
        size=(H, W),
        mode='bilinear',
        align_corners=True,
        antialias=True
    ).view(N, 3, H, W)
    
    os.makedirs('output', exist_ok=True)
    out = (out * 255.0).clamp(0, 255.0)
    out = out[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('output/out.png', out)

    gt = (gt * 255.0).clamp(0, 255.0)
    gt = gt[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('output/gt.png', gt)

    lw = (warped * 255.0).clamp(0, 255.0)
    for k in range(lw.shape[1]):
        out = lw[0, k].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'output/out_{k}.png', out)
    

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
