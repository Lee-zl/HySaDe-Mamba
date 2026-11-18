import time
import math
from functools import partial
from typing import Optional, Callable
from thop import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

# an alternative for mamba_ssm (in which causal_conv1d is needed)
try:
    from selective_scan import selective_scan_fn as selective_scan_fn_v1
    from selective_scan import selective_scan_ref as selective_scan_ref_v1
except:
    pass

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop

    assert not with_complex

    flops = 0  # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


# class MultiHeadAttention(nn.Module):
#     def __init__(self, in_channels, num_heads=4):
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.head_dim = in_channels // num_heads
#         assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"

#         self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

#     def forward(self, x):
#         batch_size, channels, height, width = x.shape
#         q = self.query(x).view(batch_size, self.num_heads, self.head_dim, height * width)
#         k = self.key(x).view(batch_size, self.num_heads, self.head_dim, height * width)
#         v = self.value(x).view(batch_size, self.num_heads, self.head_dim, height * width)

#         attn_weights = torch.einsum("bnqd,bnkd->bnqk", q, k) / (self.head_dim ** 0.5)
#         attn_weights = F.softmax(attn_weights, dim=-1)

#         attn_output = torch.einsum("bnqk,bnvd->bnqd", attn_weights, v)
#         attn_output = attn_output.view(batch_size, channels, height, width)
#         attn_output = self.out_conv(attn_output)

#         return attn_output

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,  #
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  # 就是[96, 192, 384, 768]
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj1 = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs, )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0))  # 参数矩阵 (K=4, dt_rank + d_state * 2, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        '''spectrum'''
        # 初始化部分
        self.x_proj_spectrum = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight_spectrum = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj_spectrum], dim=0))  # (K=2, N, inner)
        del self.x_proj_spectrum

        self.dt_projs_spectrum = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight_spectrum = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs_spectrum], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias_spectrum = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs_spectrum], dim=0))  # (K=2, inner)
        del self.dt_projs_spectrum
        self.conv_layer1 = None
        self.conv_layer2 = None
        self.deconv_layer11 = None
        self.deconv_layer12 = None
        self.deconv_layer21 = None
        self.deconv_layer22 = None
        '''  111'''
        self.A_logs_spectrum = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds_spectrum = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.linear_layer1 = None
        self.activation_function = nn.ReLU()
        self.linear_layer2 = None
        self.linear_layer3 = None
        self.linear_layer4 = None

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        # self.selective_scan = selective_scan_fn
        self.forward_core1 = self.forward_corev1
        self.forward_core2 = self.forward_corev2
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.ISSMG = None
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    '''(B, C ,H ,W) ->(B, 1, C, HW)'''

    def forward_corev1(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def conv2d_flops_counter_hook(self, conv_module, input, output):
        batch_size = input[0].size(0)
        output_dims = output.shape[2:]
        kernel_dims = conv_module.kernel_size
        in_channels = conv_module.in_channels
        out_channels = conv_module.out_channels
        groups = conv_module.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = int(
            kernel_dims[0] * kernel_dims[1] *
            in_channels * filters_per_channel
        )

        active_elements_count = batch_size * int(
            output_dims[0] * output_dims[1]
        )

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if conv_module.bias is not None:
            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops
        conv_module.total_ops = torch.Tensor([int(overall_flops)])

    def linear_flops_counter_hook(self, linear_module, input, output):
        batch_size = input[0].size(0)
        input_dim = input[0].size(1)
        output_dim = linear_module.out_features

        total_flops = batch_size * input_dim * output_dim

        if linear_module.bias is not None:
            total_flops += output_dim * batch_size

        linear_module.total_ops = torch.Tensor([int(total_flops)])

    def depthwise_conv_flops_counter_hook(self, conv_module, input, output):
        batch_size = input[0].size(0)
        output_dims = output.shape[2:]
        kernel_dims = conv_module.kernel_size
        in_channels = conv_module.in_channels
        groups = conv_module.groups

        filters_per_channel = 1
        conv_per_position_flops = int(
            kernel_dims[0] * kernel_dims[1] *
            in_channels * filters_per_channel
        )

        active_elements_count = batch_size * int(
            output_dims[0] * output_dims[1]
        )

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if conv_module.bias is not None:
            bias_flops = in_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops
        conv_module.total_ops = torch.Tensor([int(overall_flops)])

    def pointwise_conv_flops_counter_hook(self, conv_module, input, output):
        batch_size = input[0].size(0)
        output_dims = output.shape[2:]
        in_channels = conv_module.in_channels
        out_channels = conv_module.out_channels

        filters_per_channel = 1
        conv_per_position_flops = int(
            1 * 1 *
            in_channels * out_channels
        )

        active_elements_count = batch_size * int(
            output_dims[0] * output_dims[1]
        )

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if conv_module.bias is not None:
            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops
        conv_module.total_ops = torch.Tensor([int(overall_flops)])

    def forward_corev2(self, x: torch.Tensor):
        device = x.device

        B, C, _, _ = x.shape
        K = 2

        if self.conv_layer1 is None:
            self.conv_layer1 = nn.Conv2d(C, C, kernel_size=4, stride=2, padding=1, groups=C, bias=False).to(device)
            self.conv_layer1.register_forward_hook(self.depthwise_conv_flops_counter_hook)
        if self.conv_layer2 is None:
            self.conv_layer2 = nn.Conv2d(C, C, kernel_size=4, stride=2, padding=1, groups=C, bias=False).to(device)
            self.conv_layer2.register_forward_hook(self.depthwise_conv_flops_counter_hook)
        x = self.conv_layer1(x)  # (B, C, H/2, W/2)
        x = self.conv_layer2(x)  # (B, C, H/4, W/4)
        H, W = x.shape[2], x.shape[3]
        L = H * W

        x1 = x.view(B, C, -1)
        x2 = torch.flip(x1, dims=[1])

        if self.linear_layer1 is None:
            self.linear_layer1 = nn.Linear(L, C, bias=False).to(device)
            self.linear_layer1.register_forward_hook(self.linear_flops_counter_hook)
        x1 = self.activation_function(self.linear_layer1(x1)).permute(0, 2, 1).contiguous()  # (B, C, C)

        if self.linear_layer2 is None:
            self.linear_layer2 = nn.Linear(L, C, bias=False).to(device)
            self.linear_layer2.register_forward_hook(self.linear_flops_counter_hook)
        x2 = self.activation_function(self.linear_layer2(x2)).permute(0, 2, 1).contiguous()  # (B, C, C)

        xs = torch.cat([x1, x2], dim=1)  # (B, 2*C, C)

        x_dbl = torch.einsum("b k l d, k c l -> b k c d", xs.view(B, K, -1, C), self.x_proj_weight_spectrum)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, C), self.dt_projs_weight_spectrum)

        xs = xs.float().view(B, -1, C)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, C)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, C)  # (b, k, d_state, l)  torch.Size([1, 2, 16,  192])
        Cs = Cs.float().view(B, K, -1, C)  # (b, k, d_state, l)
        Ds = self.Ds_spectrum.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs_spectrum.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias_spectrum.float().view(-1)  # (k * d)

        # Ensure A has the correct shape (dim, dstate)
        As = As.view(-1, self.d_state)  # torch.Size([768, 16])
        # import pdb; pdb.set_trace()
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, C)
        assert out_y.dtype == torch.float

        y1 = out_y[:, 0].permute(0, 2, 1).contiguous()
        if self.linear_layer3 is None:
            self.linear_layer3 = nn.Linear(C, L).to(device)
            self.linear_layer3.register_forward_hook(self.linear_flops_counter_hook)

        y1 = self.activation_function(self.linear_layer3(y1))  # (B, C, L)

        y2 = out_y[:, 1].permute(0, 2, 1).contiguous()
        if self.linear_layer4 is None:
            self.linear_layer4 = nn.Linear(C, L).to(device)
            self.linear_layer4.register_forward_hook(self.linear_flops_counter_hook)
        y2 = self.activation_function(self.linear_layer4(y2))
        y2 = torch.flip(y2, dims=[1])

        y1 = y1.view(B, C, H, W)
        y2 = y2.view(B, C, H, W)
        if self.deconv_layer11 is None:
            self.deconv_layer11 = nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1, groups=C, bias=False).to(
                device)
            self.deconv_layer11.register_forward_hook(self.depthwise_conv_flops_counter_hook)

        if self.deconv_layer12 is None:
            self.deconv_layer12 = nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1, groups=C, bias=False).to(
                device)
            self.deconv_layer12.register_forward_hook(self.depthwise_conv_flops_counter_hook)
        if self.deconv_layer21 is None:
            self.deconv_layer21 = nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1, groups=C, bias=False).to(
                device)
            self.deconv_layer21.register_forward_hook(self.depthwise_conv_flops_counter_hook)
        if self.deconv_layer22 is None:
            self.deconv_layer22 = nn.ConvTranspose2d(C, C, kernel_size=4, stride=2, padding=1, groups=C, bias=False).to(
                device)
            self.deconv_layer22.register_forward_hook(self.depthwise_conv_flops_counter_hook)

        y1 = self.deconv_layer11(y1)  # (B, C, H*2, W*2)
        y1 = self.deconv_layer12(y1)  # (B, C, H*4, W*4)
        y2 = self.deconv_layer21(y2)  # (B, C, H*2, W*2)
        y2 = self.deconv_layer22(y2)  # (B, C, H*4, W*4)

        return y1, y2

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj1(x)
        x_spatial, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        ##### Spatial Scan
        x_spatial = x_spatial.permute(0, 3, 1, 2).contiguous()  # (B, 2C, H, W)
        x_spatial = self.act(self.conv2d1(x_spatial))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core1(x_spatial)

        ##### Spectral Scan
        x_spectrum = self.in_proj2(x)
        x_spectrum = x_spectrum.permute(0, 3, 1, 2).contiguous()
        x_spectrum = self.act(self.conv2d2(x_spectrum))
        y9, y10 = self.forward_core2(x_spectrum)

        assert y1.dtype == torch.float32

        y1_ = y1 + y2 + y3 + y4
        y2_ = y9 + y10  # (B, C, L)

        y1_ = torch.transpose(y1_, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y1_ = self.out_norm(y1_)
        y1_ = y1_ * F.silu(z)

        y2_ = torch.transpose(y2_, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y2_ = self.out_norm(y2_)
        y2_ = y2_ * F.silu(z)

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # if self.ISSMG is None:
        #     self.ISSMG = ImprovedSpatialSpectralMixtureGate(y1_.size(1) + y2_.size(1)).to(device)
        #     self.ISSMG.register_forward_hook(self.depthwise_conv_flops_counter_hook)
        # feats = self.ISSMG(y1_, y2_)
        # out = self.out_proj(feats)
        out = self.out_proj(y1_ + y2_)
        # out = self.out_proj(y1_)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


class VSSLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            downsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x


class VSSLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
            self,
            dim,
            depth,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            d_state=16,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:  # is this really applied? Yes, but been overriden later in VSSM!
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x


# dims=[96, 192, 384, 768], dims_decoder=[768, 384, 192, 96]
class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[160, 320, 640, 1280], dims_decoder=[1280, 640, 320, 160], d_state=16, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        # WASTED absolute position embedding ======================
        self.ape = False
        # self.ape = False
        # drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer_up(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,  # 20240109
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i_layer]):sum(depths_decoder[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer)

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1], dim_scale=4, norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        skip_list = []
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)
        return x, skip_list

    def forward_features_up(self, x, skip_list):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-inx])

        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        x, skip_list = self.forward_features(x)
        x = self.forward_features_up(x, skip_list)
        x = self.forward_final(x)

        return x







