# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses.ssim import ssim
from models.losses.architectures import VGG19


def dilate_mask(mask, kernel_size=3):
    """
    Apply dilation to a mask.

    Args:
        mask (torch.Tensor): Input mask of shape (N, 1, H, W), with 0s and 1s.
        kernel_size (int): Size of the dilation kernel.

    Returns:
        torch.Tensor: Dilated mask of the same shape as input.
    """
    # Create a circular kernel for dilation
    padding = kernel_size // 2
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=mask.device)
    
    # Dilation is a max-pooling operation using the kernel
    dilated_mask = F.conv2d(mask.float(), kernel, padding=padding) > 0
    return dilated_mask.float()


def masked_l1_loss(pred, target, mask=None):
    """
    Compute L1 loss with a mask.

    Args:
        pred (torch.Tensor): Predicted tensor.
        target (torch.Tensor): Ground truth tensor.
        mask (torch.Tensor): Mask of shape (N, 1, H, W), where 1 indicates valid regions.

    Returns:
        torch.Tensor: Masked L1 loss.
    """
    loss = torch.abs(pred - target)
    loss = (loss * mask).sum() / (mask.sum() + 1e-7) if mask is not None else loss.mean()
    return loss


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
    def __init__(self, epsilon=1e-6, h=0.5):
        super().__init__()
        self.model = VGG19(
            requires_grad=False,
        )  # Set to false so that this part of the network is frozen
        self.criterion = nn.L1Loss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()  # Reshape for broadcasting
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()    # Reshape for broadcasting
        self.epsilon = epsilon
        self.h = h

    def forward(self, pred_img, gt_img, mask=None):
        pred_img = (pred_img - self.mean) / self.std
        gt_img = (gt_img - self.mean) / self.std

        gt_fs = self.model(gt_img)
        pred_fs = self.model(pred_img)

        # Collect the losses at multiple layers (need unsqueeze in
        # order to concatenate these together)
        loss = sum(
            self.compute_contextual_loss(
                self.compute_similarity(pred_fs[i], gt_fs[i]),
                mask 
            ) for i in range(len(gt_fs))
        )
        return loss
    
    def compute_similarity(self, pred_features, target_features):
        """
        Compute cosine similarity between features.

        Args:
            pred_features (torch.Tensor): Predicted feature map of shape (N, C, H, W).
            target_features (torch.Tensor): Target feature map of the same shape.

        Returns:
            torch.Tensor: Cosine similarity of shape (N, H, W, H, W).
        """
        # Normalize feature maps along the channel dimension
        pred_norm = pred_features / (torch.norm(pred_features, dim=1, keepdim=True) + self.epsilon)
        target_norm = target_features / (torch.norm(target_features, dim=1, keepdim=True) + self.epsilon)

        # Compute cosine similarity (N, H, W)
        cosine_similarity = torch.sum(pred_norm * target_norm, dim=1)  # Dot product over channels
        return cosine_similarity
    
    def compute_contextual_loss(self, cosine_similarity, mask=None):
        """
        Compute contextual loss using similarity map.

        Args:
            cosine_similarity (torch.Tensor): Similarity map of shape (N, H, W, H, W).
            mask (torch.Tensor): Optional mask of shape (N, 1, H, W).

        Returns:
            torch.Tensor: Contextual loss.
        """
        sim_exp = torch.exp((cosine_similarity - 1) / self.h)  # Exponential transformation
        sim_exp_sum = torch.sum(sim_exp, dim=(1, 2), keepdim=True)  # Normalize over spatial dimensions
        contextual_similarity = sim_exp / (sim_exp_sum + self.epsilon)  # Normalize to sum to 1

        # Contextual loss: maximize similarity
        cx_loss = -torch.log(contextual_similarity + self.epsilon)

        # Apply mask if provided
        if mask is not None:
            _mask = F.interpolate(mask, size=cx_loss.shape[-2:], align_corners=True, mode='bilinear')
            cx_loss = cx_loss * _mask.squeeze(1)  # Apply mask (N, H, W)

        # Average over valid pixels
        if mask is not None:
            return cx_loss.sum() / (_mask.sum() + self.epsilon)
        else:
            return cx_loss.mean()
