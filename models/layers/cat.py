from tma_model_zoo.universal.swin import Mlp, window_partition, window_reverse
from .osa import *


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, query, key, value):
        B, X, Y, W1, W2, C = query.shape

        query = rearrange(query, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        key = rearrange(key, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
        value = rearrange(value, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]
        
        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, '(b x y) (w1 w2) d -> b x y w1 w2 d', x=X, y=Y, w1=W1, w2=W2)
        
        return x


class ICATBlock(nn.Module):
    """ Implementation of CAT Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        patch_size (int): Patch size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value.
        qk_scale (float | None, optional): Default qk scale is head_dim ** -0.5.
        drop (float, optional): Dropout rate.
        attn_drop (float, optional): Attention dropout rate.
        drop_path (float, optional): Stochastic depth rate.
        act_layer (nn.Module, optional): Activation layer.
        norm_layer (nn.Module, optional): Normalization layer.
        rpe (bool): Use relative position encoding or not.
    """

    def __init__(
            self, dim, num_heads, window_size, mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer=nn.LayerNorm
        ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(
            dim=dim,
            num_heads=num_heads,
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim, act_layer=act_layer
        )
        
        # need to be changed in different stage during forward phase
        self.H = None
        self.W = None
        
    def forward(self, x, y):
        H, W = x.shape[-2:]
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_r, 0, pad_b))
        y = F.pad(y, (0, pad_r, 0, pad_b))

        x = rearrange(x, 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.window_size, w2=self.window_size)

        for i in range(y.shape[1]):
            t = rearrange(y[:, i], 'b d (x w1) (y w2) -> b x y w1 w2 d', w1=self.window_size, w2=self.window_size)
            x = x + self.attn(x, t, t)
        
        x = x + self.mlp(self.norm2(x))

        x = rearrange(x, 'b x y w1 w2 d -> b d (x w1) (y w2)')
        
        return x[..., :H, :W]