import torch
import torch.nn as nn


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, x):
        return SwishImplementation.apply(x)


_has_silu = 'silu' in dir(torch.nn.functional)
Swish = nn.SiLU if _has_silu else MemoryEfficientSwish