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

    adepths = torch.tensor(np.load('wildrgb/apple_002/depths.npy'))
    adepths = adepths.unsqueeze(1).unsqueeze(0).float().cuda()

    cps = [
        'wildrgb/apple_002/00000/gt_enhanced.png',
        'wildrgb/apple_002/00057/gt_enhanced.png',
        'wildrgb/apple_002/00087/gt_enhanced.png',
        'wildrgb/apple_002/00137/gt_enhanced.png', 
        'wildrgb/apple_002/00170/gt_enhanced.png',
        'wildrgb/apple_002/00207/gt_enhanced.png', 
        'wildrgb/apple_002/00240/gt_enhanced.png',
        'wildrgb/apple_002/00273/gt_enhanced.png'
    ]
    acolors = []
    for c in cps:
        acolors.append(cv2.imread(c))
    acolors = np.array(acolors) / 255.0
    acolors = torch.tensor(acolors)
    acolors = acolors.permute(0, 3, 1, 2).unsqueeze(0).float().cuda()

    aK = torch.tensor(np.load('wildrgb/apple_002/intrinsic.npy'))
    aK = aK.unsqueeze(0).float().cuda()

    aRTs = torch.tensor(np.load('wildrgb/apple_002/pose.npy'))
    aRTs = aRTs.unsqueeze(0).float().cuda()

    dst_RTs = torch.tensor(
        np.array(
            [
                [0.9998969, 0.0142635, 0.0016333, 0.01],
                [-0.0142360, 0.9997751, -0.0157192, 0.01],
                [-0.0018571, 0.0156943, 0.9998751, 0],
                [0, 0, 0, 1.0],
            ]
        )
    ).float().cuda()
    dst_RTs = dst_RTs.view(1, 1, 4, 4)

    aRTs_inv = torch.inverse(aRTs)
    dst_RTinvs = torch.inverse(dst_RTs)

    depths = adepths
    colors = acolors * 2.0 - 1.0
    K = aK[:, 0]
    src_RTs = aRTs
    src_RTinvs = aRTs_inv

    # model.load_state_dict(sd['state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        out, warped = model(
            depths,
            colors,
            K,
            src_RTinvs,
            src_RTs,
            dst_RTinvs,
            dst_RTs, 
            visualize=True
        )

    out = ((out + 1.0) / 2.0 * 255.0).clamp(0, 255.0)
    out = out[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    cv2.imwrite('out.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    warped = ((warped + 1.0) / 2.0 * 255.0).clamp(0, 255.0)
    for k in range(warped.shape[0]):
        out = warped[k, 0].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'out_{k}.png', cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def compute_mean_image(warped):
    mean_mask = (warped > 0).astype(np.float32)
    total_mean_mask = np.sum(mean_mask, axis=0)
    total_mean_mask[total_mean_mask == 0] = 1.0
    mean_image = np.sum(warped.astype(np.float32), axis=0) / total_mean_mask
    mean_image = mean_image.astype(np.uint8)
    return mean_mask, mean_image


def save_results(outpath, evaluation):
    finish_file_name = os.path.join(outpath, "finish.txt")
    with open(finish_file_name, 'w', buffering=1) as finish_file:
        write_evaluation_results(finish_file, evaluation)


def write_evaluation_results(finish_file, evaluation):
    finish_file.write("-----------all dataset evaluation------------\n")
    pred_psnr_collect = []
    pred_ssim_collect = []
    pred_lpips_collect = []
    for scan in evaluation.keys():
        pred_psnr = evaluation[scan]['all_pred_psnr']
        pred_ssim = evaluation[scan]['all_pred_ssim']
        pred_lpips = evaluation[scan]['all_pred_lpips']
        pred_psnr_collect.append(pred_psnr)
        pred_ssim_collect.append(pred_ssim)
        pred_lpips_collect.append(pred_lpips)
        finish_file.write(f"{scan}: output psnr {pred_psnr}, output ssim {pred_ssim}, lpips {pred_lpips}\n")

    finish_file.write(f"Total: output psnr {np.mean(pred_psnr_collect)}, output ssim {np.mean(pred_ssim_collect)}, lpips{np.mean(pred_lpips_collect)}\n")

    finish_file.write("-----------excluded dataset evaluation-----------\n")
    pred_psnr_collect = []
    pred_ssim_collect = []
    pred_lpips_collect = []

    for scan in evaluation.keys():
        pred_psnr = evaluation[scan]['exclu_pred_psnr']
        pred_ssim = evaluation[scan]['exclu_pred_ssim']
        pred_lpips = evaluation[scan]['excludee_pred_lpips']
        pred_psnr_collect.append(pred_psnr)
        pred_ssim_collect.append(pred_ssim)
        pred_lpips_collect.append(pred_lpips)
        finish_file.write(f"{scan}: output psnr {pred_psnr}, output ssim {pred_ssim}, pred lpips {pred_lpips}\n")

    finish_file.write(f"Total: output psnr {np.mean(pred_psnr_collect)}, output ssim {np.mean(pred_ssim_collect)},pred lpips {np.mean(pred_lpips_collect)}\n")


def compute_lpips(args, device, lpips_vgg, gts, preds, gts_exclude, preds_exclude):
    gts = torch.stack(gts)
    preds = torch.stack(preds)
    preds_spl = torch.split(preds, args.lpips_batch_size, dim=0)
    gts_spl = torch.split(gts, args.lpips_batch_size, dim=0)
    
    lpips_all = []  
    with torch.no_grad():
        for predi, gti in zip(preds_spl, gts_spl):
            lpips_i = lpips_vgg(predi.to(device=device), gti.to(device=device))
            lpips_all.append(lpips_i)
        lpips_all = torch.cat(lpips_all)
    lpips_total = lpips_all.mean().item()
    
    lpips_all = []  
    gts_exclude = torch.stack(gts_exclude)
    preds_exclude = torch.stack(preds_exclude)
    preds_spl = torch.split(preds_exclude, args.lpips_batch_size, dim=0)
    gts_spl = torch.split(gts_exclude, args.lpips_batch_size, dim=0)
    with torch.no_grad():
        for predi, gti in zip(preds_spl, gts_spl):
            lpips_i = lpips_vgg(predi.to(device=device), gti.to(device=device))
            lpips_all.append(lpips_i)
        lpips_all = torch.cat(lpips_all)
    lpips_exclude = lpips_all.mean().item()

    return lpips_total, lpips_exclude


def compute_ssim_psnr(base_exclude_views, batch, results, scan_path, src_list):
    target_view_list = np.ones(len(batch['images']))
    target_view_list[src_list] = 0
    target_view_list = np.nonzero(target_view_list)[0]
    target_view_list = target_view_list.tolist()

    for k in range(results[1]['ProjectedDepths'][0].shape[1]):
        x = results[1]['ProjectedDepths'][0][:, k]
        for i in range(3):
            y = x[i]
            y = (y - y.min()) / (y.max() - y.min()) * 255
            y = y.detach().cpu().numpy().astype(np.uint8)
            imageio.imwrite(os.path.join(scan_path, f"pd_{k:06d}_{i:06d}.png"), y)

    for k in range(results[1]['Warped'].shape[0]):
        x = (results[1]['Warped'][k] / 2 + 0.5) * 255

        for i in range(3):
            y = x[i].permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
            imageio.imwrite(os.path.join(scan_path, f"warped_{k:06d}_{i:06d}.png"), y)

    depths = batch['depths']
    for i in range(3):
        x = (depths[i] - depths[i].min()) / (depths[i].max() - depths[i].min()) * 255
        x = x.detach().cpu().numpy().astype(np.uint8)[0, 0]
        imageio.imwrite(os.path.join(scan_path, "raw_depth_{0:06d}.png".format(i)), x)


    ssim_dict = []
    psnr_dict = []
    ssim_exclude_dict = []
    psnr_exclude_dict = []
    gts = []
    preds = []
    gts_exclude = []
    preds_exclude = []
    for i in range(results[1]['OutputImg'].shape[0]):
        target_view = target_view_list[i]
        target = results[1]['OutputImg'][i]
        pred = results[1]['PredImg'][i]
        gts.append(target)
        preds.append(pred)

        target = tensor_to_image(target)
        pred = tensor_to_image(pred)

        imageio.imwrite(os.path.join(scan_path, "output_view_{0:06d}.png".format(target_view)), np.uint8(pred * 255.0))
        imageio.imwrite(os.path.join(scan_path, "target_view_{0:06d}.png".format(target_view)), np.uint8(target * 255.0))

        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        mask = ((1 - ((gray == 0) | (gray == 255))) * 255).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # We calculate the psnr and ssim
        psnr_pred = peak_signal_noise_ratio(target, pred, data_range=1)
        ssim_pred = structural_similarity(target, pred, channel_axis=-1, data_range=1)

        psnr_dict.append(psnr_pred)
        ssim_dict.append(ssim_pred)
        if target_view not in base_exclude_views:
            gts_exclude.append(results[1]['OutputImg'][i])
            preds_exclude.append(results[1]['PredImg'][i])
            psnr_exclude_dict.append(psnr_pred)
            ssim_exclude_dict.append(ssim_pred)
    
    return ssim_dict, psnr_dict, ssim_exclude_dict, psnr_exclude_dict, gts, preds, gts_exclude, preds_exclude


def compute_depth_error(opts, batch, results):
    depth_imgs = []
    if "depths" in batch.keys():
        for i in range(opts.input_view_num):
            if torch.cuda.is_available():
                depth_imgs.append(batch["depths"][i].cuda())
            else:
                depth_imgs.append(batch["depths"][i])

    pred_depth = results[1]["Completed"][:opts.input_view_num]
    H, W = results[1]['Completed'][0].shape[-2], results[1]['Completed'][0].shape[-1]

    if opts.down_sample:
        depth_imgs = torch.cat(depth_imgs, 1)
        gt_depth = F.interpolate(depth_imgs, size=(H, W), mode="nearest").unsqueeze(2)
    else:
        gt_depth = torch.stack(depth_imgs, 1)
    gt_depth = gt_depth.contiguous().view(-1, 1, H, W)

    return nn.MSELoss()(pred_depth[gt_depth > 0.0001], gt_depth[gt_depth > 0.0001])


def write_input_images(output_path, scan_name, batch, results):
    input_view1 = batch['images'][0][0] * 0.5 + 0.5
    input_view2 = batch['images'][1][0] * 0.5 + 0.5
    input_view3 = batch['images'][2][0] * 0.5 + 0.5

    input_depth1 = results[1]['Completed'][0, 0, 0]
    input_depth2 = results[1]['Completed'][0, 1, 0]
    input_depth3 = results[1]['Completed'][0, 2, 0]

    input_view1 = input_view1.permute(1,2,0).detach().cpu().numpy()
    input_view2 = input_view2.permute(1,2,0).detach().cpu().numpy()
    input_view3 = input_view3.permute(1,2,0).detach().cpu().numpy()
    input_depth1 = np.uint8(input_depth1.detach().cpu().numpy() / input_depth1.max().item() * 255.0)
    input_depth2 = np.uint8(input_depth2.detach().cpu().numpy() / input_depth2.max().item() * 255.0)
    input_depth3 = np.uint8(input_depth3.detach().cpu().numpy() / input_depth3.max().item() * 255.0)
    scan_path = os.path.join(output_path, scan_name)

    if os.path.exists(scan_path) is False:
        os.makedirs(scan_path)
        
    imageio.imwrite(os.path.join(scan_path, "input_view1.png"), np.uint8(input_view1 * 255.0))
    imageio.imwrite(os.path.join(scan_path, "input_view2.png"), np.uint8(input_view2 * 255.0))
    imageio.imwrite(os.path.join(scan_path, "input_view3.png"), np.uint8(input_view3 * 255.0))

    imageio.imwrite(os.path.join(scan_path, "input_depth1.png"), input_depth1)
    imageio.imwrite(os.path.join(scan_path, "input_depth2.png"), input_depth2)
    imageio.imwrite(os.path.join(scan_path, "input_depth3.png"), input_depth3)
    return scan_path


def create_evaluation_dicts(args):
    # We follow the paradigm of pixel-nerf for evaluation. 
    # The input/src view is pre-defined and there are some pre-defined excluded views.
    # We report performance with and wo these excluded views.
    src_list = args.src_list
    src_list = list(map(int, src_list.split()))
    base_exclude_views = deepcopy(src_list)
    base_exclude_views.extend([3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39])
    evaluation = OrderedDict()
    return src_list, base_exclude_views, evaluation


def load_dataset(args, opts):
    print("Loaded model")
    dataset = get_dataset(opts)
    test_set = dataset(stage='test', opts=opts)
    dataloader = DataLoader(
        test_set,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
    )
    output_path = os.path.join(args.output_path, args.name)
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)
    return dataloader, output_path


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
