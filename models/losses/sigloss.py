import torch
import torch.nn as nn


class SigLoss(nn.Module):
    def __init__(self, valid_mask=True, loss_weight=1.0, max_depth=None, warm_up=False, warm_iter=200):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001

        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input_tensor, target):
        valid_mask = target > 0.001
        if self.max_depth is not None:
            valid_mask = torch.logical_and(valid_mask, target <= self.max_depth)
        input_tensor = input_tensor[valid_mask]
        target = target[valid_mask]

        if self.warm_up and self.warm_up_counter < self.warm_iter:
            g = torch.log(input_tensor + self.eps) - torch.log(target + self.eps)
            g = 0.15 * torch.pow(torch.mean(g), 2)
            self.warm_up_counter += 1
            return torch.sqrt(g)

        g = torch.log(input_tensor + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt, **kwargs):
        return self.loss_weight * self.sigloss(depth_pred, depth_gt)