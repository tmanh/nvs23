# This file is licensed under Apache-2.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_lstm2 import *
# from .vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d


class DViLLayer(ViLLayer):
    def __init__(
            self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=4,
            conv_kind="2d",
            init_weights="original",
            seqlens=None,
            num_blocks=None,
    ):
        super().__init__(
            dim,
            direction,
            expansion,
            qkv_block_size,
            proj_bias,
            norm_bias,
            conv_bias,
            conv_kernel_size,
            conv_kind,
            init_weights,
            seqlens,
            num_blocks
        )

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        
        self.d_proj_up = nn.Linear(
            in_features=dim,
            out_features=inner_dim,
            bias=proj_bias,
        )

        self.dq_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.dk_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        self.reset_m_parameters()

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape

        # alternate direction in successive layers
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
            d = d.flip(dims=[1])
        else:
            raise NotImplementedError

        # up-projection
        x_inner = self.proj_up(x)
        d_mlstm = self.d_proj_up(d)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # mlstm branch
        x_mlstm_conv = self.conv(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)# * torch.sigmoid(self.dq_proj(d_mlstm))
        k = self.k_proj(x_mlstm_conv_act)# * torch.sigmoid(self.dk_proj(d_mlstm))
        v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # output / z branch
        h_state = h_tilde_state_skip * F.silu(z)

        # down-projection
        x = self.proj_down(h_state)

        # reverse alternating flip
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError

        return x

    def reset_m_parameters(self):
        small_init_(self.d_proj_up.weight, dim=self.dim)
        if self.d_proj_up.bias is not None:
            nn.init.zeros_(self.d_proj_up.bias)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            # use the embedding dim instead of the inner embedding dim
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.dq_proj)
        _init_qkv_proj(self.dk_proj)


class DViLBlock(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.init_weights = init_weights

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = DViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.layer(x, d)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()


class DViLBlockPair(nn.Module):
    def __init__(
            self,
            dim,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
    ):
        super().__init__()
        self.rowwise_from_top_left = DViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )
        self.rowwise_from_bot_right = DViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

    def forward(self, x, d):
        x = self.rowwise_from_top_left(x, d)
        x = self.rowwise_from_bot_right(x, d)
        return x
