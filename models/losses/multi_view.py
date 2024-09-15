import torch
import torch.nn as nn

from models.losses.ssim import SSIM


class ReprojectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.ssim = SSIM()

    def forward(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss