#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OSA.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:07:42 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

from torch import einsum
from einops import rearrange

from .osa_utils import *
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
from models.layers.vision_lstm.vision_lstm2 import ViLBlock, ViLBlockPair


# attention related classes
class Attention(nn.Module):     # NOTE: Spatial attention (MLP style)
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out
        out = self.to_out(out)

        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)
    

class VisualDepthAttention(nn.Module):     # NOTE: Spatial attention (MLP style)
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qv = nn.Linear(dim + 1, dim, bias = False)
        self.to_kv = nn.Linear(dim + 1, dim, bias = False)
        self.to_vv = nn.Linear(dim + 1, dim, bias = False)

        self.to_qd = nn.Linear(dim + 1, 1, bias = False)
        self.to_kd = nn.Linear(dim + 1, 1, bias = False)
        self.to_vd = nn.Linear(dim + 1, 1, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim + 1, dim + 1, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        qv, kv, vv = self.to_qv(x), self.to_kv(x), self.to_vv(x)  #self.to_qkv(x).chunk(3, dim = -1)
        qd, kd, vd = self.to_qd(x), self.to_kd(x), self.to_vd(x)  #self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        qv, kv, vv = map(
            lambda t: rearrange(
                t, 'b n (h d) -> b h n d', h = h),
                (qv, kv, vv)
        )

        qd, kd, vd = map(
            lambda t: rearrange(
                t, 'b n (h d) -> b h n d', h = 1),
                (qd, kd, vd)
        )

        # scale
        qv = qv * self.scale
        qd = qd * self.scale

        # sim
        sim_v = einsum('b h i d, b h j d -> b h i j', qv, kv)
        sim_d = einsum('b h i d, b h j d -> b h i j', qd, kd)

        # attention
        sim_m = sim_v * F.sigmoid(sim_d)
        attn_v = self.attend(sim_m)
        attn_d = self.attend(torch.sum(sim_m, dim=1, keepdim=True))

        # aggregate
        out_v = einsum('b h i j, b h j d -> b h i d', attn_v, vv)
        out_d = einsum('b h i j, b h j d -> b h i d', attn_d, vd)
        
        # merge heads
        out_v = rearrange(out_v, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)
        out_d = rearrange(out_d, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out
        out = self.to_out(torch.cat([out_v, out_d], dim=-1))

        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)
    

class MultiViewAttentionFusion(nn.Module):     # NOTE: Spatial attention (MLP style)
    def __init__(
        self,
        dim,
        dim_head = 1,
        dropout = 0.,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        dim_head = dim
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.view_attend = nn.Sequential(
            nn.Softmax(dim = 1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout),
        )

    def forward(self, x, d):
        N, V, C, H, W = x.shape
        h = self.heads

        x = x.permute(0, 3, 4, 1, 2).contiguous().view(N * H * W, V, C)

        # project for queries, keys, values
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)  #self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h i d -> b i (h d)')
        out = out.float()
        
        # combine heads out
        att = self.view_attend(self.to_out(out))
        mask = (d > 0).float().permute(0, 3, 4, 1, 2).view(N * H * W, V, -1)
        att = att * mask
        att = att / (att.sum(dim=1, keepdim=True) + 1e-7)
        out = att * x
        out = rearrange(out, '(b h w) v c -> b v c h w', b=N , h=H, w=W)

        out = torch.sum(out, dim=1)

        return out


class SimpleAttention(nn.Module):     # NOTE: Spatial attention (MLP style)
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        # flatten
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d ) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)
        x = rearrange(x, 'b (w1 w2) d -> b w1 w2 d', w1 = window_height, w2 = window_width)

        # combine heads out
        out = x + self.to_out(out)

        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)


class Block_Attention(nn.Module):     # NOTE: Spatial attention (Conv2D style)
    def __init__(
        self,
        dim,
        dim_head = 32,
        bias=False, 
        dropout = 0.,
        window_size = 7,
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.ps = window_size
        self.scale = dim_head ** -0.5

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        # project for queries, keys, values
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1) 

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b (h d) (x w1) (y w2) -> (b x y) h (w1 w2) d', h = self.heads, w1=self.ps, w2=self.ps), (q, k, v))

        # scale
        q = q * self.scale

        # sim
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # attention
        attn = self.attend(sim)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, '(b x y) head (w1 w2) d -> b (head d) (x w1) (y w2)', x=h//self.ps, y=w//self.ps, head=self.heads, w1 = self.ps, w2 = self.ps)

        out = self.to_out(out)
        return out


class Channel_Attention(nn.Module):  # NOTE: Channel Attention (Conv2D style)
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 7
    ):
        super(Channel_Attention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)

        out = self.project_out(out)

        return out
    

class SimpleChannelAttention(nn.Module):  # NOTE: Channel Attention (Conv2D style)
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 7
    ):
        super(SimpleChannelAttention, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (h w) head d (ph pw)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (h w) head d (ph pw) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)

        out = x + self.project_out(out)

        return out


class Channel_Attention_grid(nn.Module):     # NOTE: Channel Attention (Conv2D + Grid style)
    def __init__(
        self, 
        dim, 
        heads, 
        bias=False, 
        dropout = 0.,
        window_size = 7
    ):
        super(Channel_Attention_grid, self).__init__()
        self.heads = heads

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
       
        self.ps = window_size

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        qkv = qkv.chunk(3, dim=1) 

        q,k,v = map(lambda t: rearrange(t, 'b (head d) (h ph) (w pw) -> b (ph pw) head d (h w)', ph=self.ps, pw=self.ps, head=self.heads), qkv)
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        out = rearrange(out, 'b (ph pw) head d (h w) -> b (head d) (h ph) (w pw)', h=h//self.ps, w=w//self.ps, ph=self.ps, pw=self.ps, head=self.heads)

        out = self.project_out(out)

        return out


class OSA_Block(nn.Module):
    def __init__(self, channel_num=64, bias = True, ffn_bias=True, window_size=8, dropout=0.0):
        super(OSA_Block, self).__init__()

        w = window_size
        self.window_size = window_size

        self.layer = nn.Sequential(
            MBConv(
                channel_num,
                channel_num,
                downsample = False,
                expansion_rate = 1,
                shrinkage_rate = 0.25
            ),
                
            Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w),  # block-like attention
            PreNormResidual(channel_num, Attention(dim = channel_num, dim_head = channel_num // 4, dropout = dropout)),
            Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)),

            # channel-like attention
            Conv_PreNormResidual(channel_num, Channel_Attention(dim = channel_num, heads=4, dropout = dropout, window_size = window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)),
                
            Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w),  # grid-like attention
            PreNormResidual(channel_num, Attention(dim = channel_num, dim_head = channel_num//4, dropout = dropout, window_size = window_size)),
            Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)),

            # channel-like attention
            Conv_PreNormResidual(channel_num, Channel_Attention_grid(dim = channel_num, heads=4, dropout = dropout, window_size = window_size)),
            Conv_PreNormResidual(channel_num, Gated_Conv_FeedForward(dim = channel_num, dropout = dropout)),
        )

    def forward(self, x):
        # Step 0: pad feature maps to multiples of window size
        H, W = x.shape[-2:]
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))

        # Step 1: attention
        out = self.layer(x)
        return out[..., :H, :W]
    

class VFBlock(nn.Module):
    def __init__(self, channel_num=64, window_size=8, dropout=0.0, conv_only=True):
        super().__init__()

        self.window_size = window_size

        ############## Spatial
        self.view_layer = MultiViewAttentionFusion(channel_num + 1)

    def forward(self, x, d):
        return self.view_layer(x, d)