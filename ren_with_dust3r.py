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
from utils.dust3r_wrapper import load_model, run_dust3r


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

    src_colors, dst_colors, src_depths, dst_depths, K, src_RTs, dst_RTs = load_data()

    B, V, C, H, W = src_colors.shape

    src_RTinvs = torch.inverse(src_RTs)
    dst_RTinvs = torch.inverse(dst_RTs)

    ## "/home/antruong/anaconda3/envs/render/lib/python3.10/site-packages/torch/nn/modules/module.py", line 2215
    cfg = OmegaConf.load('configs/train.yaml')
    model = LocalSimGRU(cfg).to(device)
    sd = torch.load('weights/local_sim.pt', weights_only=False)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        out, _, warped = model(
            src_depths,
            src_colors,
            K,
            
            src_RTs,
            src_RTinvs,
            
            dst_RTs, 
            dst_RTinvs,
            visualize=True,
        )

    # model.fit_depth(
    #     acolors[:, :1].repeat((1, colors.shape[1], 1, 1, 1)),
    #     depths,
    #     colors,
    #     K,
            
    #     src_RTs,
    #     src_RTinvs,
            
    #     dst_RTs, 
    #     dst_RTinvs,
    #     visualize=True,
    # )

    out = out.view(1, 3, H, W)
    gt = dst_colors.view(1, 3, H, W)
    
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


def scan_images(path):
    return [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))
    ]


def load_data(
        path='datasets/dtu_down_4/DTU/Rectified/scan21/image'
):
    indices = [29, 22, 25, 28]

    images = scan_images(path)
    images = [images[i] for i in indices]

    model = load_model()
    imgs, dpts, ks, rts = run_dust3r(images, model)

    src_colors = torch.tensor(imgs[1:]).permute((0, 3, 1, 2)).unsqueeze(0).float().cuda()
    dst_colors = torch.tensor(imgs[:1]).permute((0, 3, 1, 2)).unsqueeze(0).float().cuda()
    src_depths = torch.tensor(dpts[1:]).unsqueeze(1).unsqueeze(0).float().cuda()
    dst_depths = torch.tensor(dpts[:1]).unsqueeze(1).unsqueeze(0).float().cuda()

    K = torch.tensor(ks[0]).unsqueeze(0).float().cuda()
    src_RTs = torch.tensor(rts[1:]).unsqueeze(0).float().cuda()
    dst_RTs = torch.tensor(rts[:1]).unsqueeze(0).float().cuda()

    return src_colors, dst_colors, src_depths, dst_depths, K, src_RTs, dst_RTs
    

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
