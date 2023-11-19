# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..universal.depthformer_basics import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.
    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss. Defaults to False.
        reduction (str, optional): . Defaults to 'mean'. Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss item to be included into
            the backward graph, `loss_` must be the prefix of the name. Defaults to 'loss_ce'.
    """

    def __init__(self,
                 loss_weight=1.0):
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = loss_weight
        

    @torch.no_grad()
    def accuracy(self, output, target, topk=(1, 5, )):
        """Computes the precision@k for the specified values of k"""
        if target.numel() == 0:
            return [torch.zeros([], device=output.device)]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def forward(self, input, target):
        loss_ce = F.cross_entropy(input.squeeze(), target)
        acc = self.accuracy(input.squeeze(), target)
        loss_cls = self.loss_weight * loss_ce
        return loss_cls, acc


@LOSSES.register_module()
class SigLoss(nn.Module):
    """SigLoss.
    Args:
        valid_mask (bool, optional): Whether filter invalid gt
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self, valid_mask=True, loss_weight=1.0, max_depth=None, warm_up=False, warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input_tensor, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
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
        return self.loss_weight * self.sigloss(depth_pred, depth_gt,)
