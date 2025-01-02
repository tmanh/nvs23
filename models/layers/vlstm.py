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
from timm.models.layers import trunc_normal_
import math

# Load model directly
from models.layers.vision_lstm.vision_lstm2 import ViLBlockPair


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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class ViLViewLayer(nn.Module):
    def __init__(self, dim, hidden, drop=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim

        self.in_mlp = Mlp(
            in_features=dim,
            hidden_features=dim,
            out_features=hidden,
            act_layer=act_layer,
            drop=drop
        )

        self.mamba = nn.Sequential(
            ViLBlockPair(
                dim=hidden, # Model dimension d_model
            ),
        )

        self.out_mlp = Mlp(
            in_features=hidden,
            hidden_features=hidden,
            out_features=dim,
            act_layer=act_layer,
            drop=drop
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

    def forward(self, x):
        B, nf, C, H, W = x.shape
        self.mamba.nframes = nf

        assert C == self.dim
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, nf, C)

        x_flat_prj = self.in_mlp(x_flat)
        x_seq_prj = self.mamba(x_flat_prj)

        out = x_flat + self.out_mlp(x_seq_prj)
        out = out.view(B, H, W, nf, C).permute(0, 3, 4, 1, 2)

        return out


class ViLViewFuseLayer(nn.Module):
    def __init__(self, dim, hidden, drop=0., act_layer=nn.GELU):
        super().__init__()
        self.dim = dim

        self.in_mlp = Mlp(
            in_features=dim,
            hidden_features=dim,
            out_features=hidden,
            act_layer=act_layer,
            drop=drop
        )

        self.mamba = nn.Sequential(
            ViLBlockPair(
                dim=hidden, # Model dimension d_model
            ),
        )

        self.out_mlp = Mlp(
            in_features=hidden,
            hidden_features=hidden,
            out_features=dim + 1,
            act_layer=act_layer,
            drop=drop
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

    def forward(self, x):
        B, nf, C, H, W = x.shape
        self.mamba.nframes = nf

        assert C == self.dim
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, nf, C)

        x_flat_prj = self.in_mlp(x_flat)
        x_seq_prj = self.mamba(x_flat_prj)
        alpha_beta = self.out_mlp(x_seq_prj)

        alpha = F.softmax(alpha_beta[:, :, -1:], dim=1)

        out = torch.sum(alpha * (x_flat + alpha_beta[:, :, :-1]), dim=1)
        out = out.view(-1, H, W, C).permute(0, 3, 1, 2)

        return out