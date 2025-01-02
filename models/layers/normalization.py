# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from torch.nn.parameter import Parameter


# BatchNorm layers are taken from the BigGAN code base.
# https://github.com/ajbrock/BigGAN-PyTorch/blob/a5557079924c3070b39e67f2eaea3a52c0fb72ab/layers.py
# Distributed under the MIT licence.

# Normal, non-class-conditional BN
class BatchNorm_StandingStats(nn.Module):
    def __init__(self, output_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.output_size = output_size
        # Prepare gain and bias layers
        self.register_parameter("gain", Parameter(torch.ones(output_size)))
        self.register_parameter("bias", Parameter(torch.zeros(output_size)))

        # epsilon to avoid dividing by 0
        self.eps = eps
        
        # Momentum
        self.momentum = momentum

        self.bn = bn(output_size, self.eps, self.momentum)

    def forward(self, x, y=None):
        gain = self.gain.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return self.bn(x, gain=gain, bias=bias)


class bn(nn.Module):
    def __init__(self, num_channels, eps=1e-5, momentum=0.1):
        super().__init__()

        # momentum for updating stats
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("stored_mean", torch.zeros(num_channels))
        self.register_buffer("stored_var", torch.ones(num_channels))
        self.register_buffer("accumulation_counter", torch.zeros(1))
        # Accumulate running means and vars
        self.accumulate_standing = False

    # reset standing stats
    def reset_stats(self):
        self.stored_mean[:] = 0
        self.stored_var[:] = 0
        self.accumulation_counter[:] = 0

    def forward(self, x, gain, bias):
        if self.training:
            out, mean, var = manual_bn(
                x, gain, bias, return_mean_var=True, eps=self.eps
            )
            # If accumulating standing stats, increment them
            with torch.no_grad():
                if self.accumulate_standing:
                    self.stored_mean[:] = self.stored_mean + mean.data
                    self.stored_var[:] = self.stored_var + var.data
                    self.accumulation_counter += 1.0
                # If not accumulating standing stats, take running averages
                else:
                    self.stored_mean[:] = (
                        self.stored_mean * (1 - self.momentum)
                        + mean * self.momentum
                    )
                    self.stored_var[:] = (
                        self.stored_var * (1 - self.momentum) + var * self.momentum
                    )
            return out
        # If not in training mode, use the stored statistics
        else:
            mean = self.stored_mean.view(1, -1, 1, 1)
            var = self.stored_var.view(1, -1, 1, 1)
            # If using standing stats, divide them by the accumulation counter
            if self.accumulate_standing:
                mean = mean / self.accumulation_counter
                var = var / self.accumulation_counter
            return fused_bn(x, mean, var, gain, bias, self.eps)


# Fused batchnorm op
def fused_bn(x, mean, var, gain=None, bias=None, eps=1e-5):
    # Apply scale and shift--if gain and bias are provided, fuse them here
    # Prepare scale
    scale = torch.rsqrt(var + eps)
    # If a gain is provided, use it
    if gain is not None:
        scale = scale * gain
    # Prepare shift
    shift = mean * scale
    # If bias is provided, use it
    if bias is not None:
        shift = shift - bias
    return x * scale - shift


# Manual BN
# Calculate means and variances using mean-of-squares minus mean-squared
def manual_bn(x, gain=None, bias=None, return_mean_var=False, eps=1e-5):
    # Cast x to float32 if necessary
    float_x = x.float()
    # Calculate expected value of x (m) and expected value of x**2 (m2)
    # Mean of x
    m = torch.mean(float_x, [0, 2, 3], keepdim=True)
    # Mean of x squared
    m2 = torch.mean(float_x ** 2, [0, 2, 3], keepdim=True)
    # Calculate variance as mean of squared minus mean squared.
    var = m2 - m ** 2
    # Cast back to float 16 if necessary
    var = var.type(x.type())
    m = m.type(x.type())
    # Return mean and variance for updating stored mean/var if requested
    if return_mean_var:
        return fused_bn(x, m, var, gain, bias, eps), m.squeeze(), var.squeeze()
    else:
        return fused_bn(x, m, var, gain, bias, eps)
