# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations


import torch.nn as nn
import torch
import torch.nn.functional as F 
from timm.layers import trunc_normal_
import math

# Load model directly
from models.layers.vision_lstm.dvision_lstm2 import DViLBlockPair


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)

            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x


class Mlp(nn.Module):
    def __init__(self, in_dim, out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        return x


class ViLViewFuseLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        depth = 1
        self.blocks = nn.ModuleList(
            [
                DViLBlockPair(
                    dim=dim,
                    num_blocks=depth * 2,
                )
                for _ in range(depth)
            ],
        )

        self.m_in = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=7, padding=3),
            nn.ReLU(inplace=True)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, m):
        B, nf, C, H, W = x.shape

        m = m.view(B * nf, 1, H, W)
        m = self.m_in(m)
        m = m.view(B, nf, C, H, W)
        m_seq = m.permute(0, 3, 4, 1, 2).reshape(B * H * W, nf, C)
        x_seq = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, nf, C)
        for bb in self.blocks:
            x_seq = x_seq + bb(x_seq, m_seq)
        return x_seq.contiguous().view(-1, H, W, C).permute(0, 3, 1, 2)
    

class ViLViewMergeLayer(ViLViewFuseLayer):
    def __init__(self, dim):
        super().__init__(dim)

        self.m_out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Softmax(dim=1)
        )

        self.apply(self._init_weights)

    def forward(self, x, m):
        B, nf, C, H, W = x.shape

        m = m.view(B * nf, 1, H, W)
        m = self.m_in(m)
        m = m.view(B, nf, C, H, W)
        m_seq = m.permute(0, 3, 4, 1, 2).reshape(B * H * W, nf, C)
        x_seq = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, nf, C)
        for bb in self.blocks:
            x_seq = x_seq + bb(x_seq, m_seq)
        
        m_seq = self.m_out(torch.cat([x_seq, m_seq], dim=-1))
        
        out = torch.sum(x_seq * m_seq, dim=1)
    
        return out.contiguous().view(-1, H, W, C).permute(0, 3, 1, 2)
