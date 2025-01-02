# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from models.losses.ssim import ssim
from models.losses.architectures import VGG19


class SynthesisLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_loss_from_name(self, name):
        if name == 'l1':
            loss = L1LossWrapper()
        elif name == 'content':
            loss = PerceptualLoss(self.opt)
        elif name == 'PSNR':
            loss = PSNR()
        elif name == 'SSIM':
            loss = SSIM()
        elif name == 'l2':
            loss = L2LossWarpper()

        if torch.cuda.is_available():
            return loss.cuda()

    def forward(self, pred_img, gt_img, coarse=None):
        losses = [loss(pred_img[:, :, :gt_img.shape[-2], :gt_img.shape[-1]], gt_img, coarse) for loss in self.losses]

        loss_dir = {}
        for i, l in enumerate(losses):
            if "Total Loss" in l.keys():
                if "Total Loss" in loss_dir.keys():
                    loss_dir["Total Loss"] = (
                        loss_dir["Total Loss"]
                        + l["Total Loss"] * self.lambdas[i]
                    )
                else:
                    loss_dir["Total Loss"] = l["Total Loss"]

            loss_dir = dict(l, **loss_dir)  # Have loss_dir override l

        return loss_dir


class PSNR(nn.Module):
    def forward(self, pred_img, gt_img, coarse):
        pred_img = pred_img * 0.5 + 0.5
        gt_img = gt_img * 0.5 + 0.5
        bs = pred_img.size(0)
        mse_err = (pred_img - gt_img).pow(2).sum(dim=1).view(bs, -1).mean(dim=1)

        psnr = 10 * (1 / mse_err).log10()
        return {'psnr': psnr.mean()}


class SSIM(nn.Module):
    def forward(self, pred_img, gt_img, coarse):
        if len(pred_img.shape) == 5:
            return {"ssim": ssim(pred_img[1], gt_img[0])}
        else:
            return {"ssim": ssim(pred_img, gt_img)}


# Wrapper of the L1Loss so that the format matches what is expected
class L1LossWrapper(nn.Module):
    def forward(self, pred_img, gt_img, coarse=None):
        err = nn.L1Loss()(pred_img, gt_img)
        if coarse is not None:
            err += nn.L1Loss()(coarse, gt_img)
        return {"L1": err, "Total Loss": err}

class L2LossWarpper(nn.Module):
    def forward(self, pred_img, gt_img, coarse):
        err = nn.MSELoss()(pred_img, gt_img)
        if coarse is not None:
            err += nn.MSELoss()(coarse, gt_img)
        return {"L2":err, "Total Loss":err}


# Implementation of the perceptual loss to enforce that a
# generated image matches the given image.
# Adapted from SPADE's implementation
# (https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py)
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VGG19(
            requires_grad=False,
        )  # Set to false so that this part of the network is frozen
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()  # Reshape for broadcasting
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()    # Reshape for broadcasting

    def forward(self, pred_img, gt_img):
        pred_img = (pred_img - self.mean) / self.std
        gt_img = (gt_img - self.mean) / self.std

        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)

        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        loss = sum(self.weights[i] * self.criterion(pred_fs[i], gt_fs[i]) for i in range(len(gt_fs)))
        return loss * 0.25
