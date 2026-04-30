import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except:
    print('mamba_ssm not found')
from einops import rearrange, repeat

import math
import copy
try:
    from mamba_util import PatchMerging,SimplePatchMerging, Stem, SimpleStem, Mlp, RoPE,RMTPatchEmbed,RMTPatchMerging
except:
    from .mamba_util import PatchMerging, SimplePatchMerging, Stem, SimpleStem, Mlp, RoPE, RMTPatchEmbed,RMTPatchMerging

from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from typing import Tuple, Union
# try:
#     from fla.ops.linear_attn import chunk_linear_attn
# except:
#     print('fla not found')
#check  https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/linear_attn.py
class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])

def rope(x, shape, base=10000):
    channel_dims, feature_dim = shape[:-1], shape[-1]
    k_max = feature_dim // (2 * len(channel_dims))

    assert feature_dim % k_max == 0

    # angles
    theta_ks = 1 / (base ** (torch.arange(k_max, device=x.device) / k_max))
    angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d, device=x.device) for d in channel_dims], indexing='ij')], dim=-1)

    # rotation
    rotations_re = torch.cos(angles).unsqueeze(dim=-1)
    rotations_im = torch.sin(angles).unsqueeze(dim=-1)

    x = x.reshape(*x.shape[:-1], -1, 2)
    x_re = x[..., :1]
    x_im = x[..., 1:]
    pe_x = torch.cat([x_re * rotations_re - x_im * rotations_im, x_im * rotations_re + x_re * rotations_im], dim=-1)
    return pe_x.flatten(-2)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (B, num_windows, window_size*window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().flatten(1,2) # (B, num_windows, window_size, window_size, C)
    return windows.flatten(2, 3)# (B, num_windows, window_size*window_size, C)

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def decompose_matrix(A, d, niter=2):
    # Perform SVD
    U, S, V = torch.svd_lowrank(A, q=d, niter=niter)

    # Truncate U, S, and V to get dimensions N x d and d x N
    U_d = U[:, :d]
    S_d = S[:d]
    V_d = V[:, :d]

    # Construct the two matrices
    # Matrix 1 (N x d)
    M1 = U_d * torch.sqrt(S_d)
    # Matrix 2 (d x N)
    M2 = (V_d * torch.sqrt(S_d))
    return M1, M2

def get_sine_svd(pos_embed, svd_dim=128, niter=2):
    pos_embed = pos_embed.flatten(0,2)
    pos_sim = pos_embed@pos_embed.T
    pos_softmax = F.softmax(pos_sim, dim=-1)
    #pos_softmax = pos_softmax * (pos_softmax > 0.1*pos_softmax.mean())
    trancated_q, trancated_k = decompose_matrix(pos_softmax, svd_dim, niter)
    trancated_q, trancated_k = trancated_q.unsqueeze(0), trancated_k.unsqueeze(0) # (1, N, svd_dim), (1, N, svd_dim)
    return [trancated_q, trancated_k]

class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
class RelPos2d(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
    def generate_2d_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # (H*W 2)
        mask = grid[:, None, :] - grid[None, :, :]  # (H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]  # (n H*W H*W)
        return mask

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen: Tuple[int], ):
        '''
        slen: (h, w)
        h * w == l
        '''

        index = torch.arange(slen[0] * slen[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])  # (l d1)
        sin = sin.reshape(slen[0], slen[1], -1)  # (h w d1)
        cos = torch.cos(index[:, None] * self.angle[None, :])  # (l d1)
        cos = cos.reshape(slen[0], slen[1], -1)  # (h w d1)
        mask = self.generate_2d_decay(slen[0], slen[1])  # (n l l)
        rel_pos = ((sin, cos), mask)
        return rel_pos


class PositionEmbeddingSine(nn.Module):
    """
    #    https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=32, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, h, w):
        not_mask = torch.ones([x.shape[0], h, w],dtype=torch.bool,device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32) # [batch_size, h, w]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) # [64]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats) # [64]

        pos_x = x_embed[:, :, :, None] / dim_t # [batch_size, h, w, 64]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3)# [batch_size, h, w, 128]
        return pos


class ConvFFN(nn.Module):

    def __init__(self, channels, expansion=2, drop=0.0):
        super().__init__()

        self.dim1 = channels
        self.dim2 = channels * expansion
        self.linear1 = nn.Conv2d(self.dim1, self.dim2, 1, 1, 0)
        self.drop1 = nn.Dropout(drop, inplace=True)
        self.act = nn.GELU()
        self.linear2 = nn.Conv2d(self.dim2, self.dim1, 1, 1, 0)
        self.drop2 = nn.Dropout(drop, inplace=True)
        self.dwc = nn.Conv2d(self.dim2, self.dim2, 3, 1, 1, groups=self.dim2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = x + self.dwc(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x

class StandardAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = inner_dim

    def forward(self, x, H, W, relpos):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Mamba2(nn.Module):
    def __init__(
        self,
        d_model,
        d_conv=3, #default to 3 for 2D
        conv_init=None,
        expand=2,
        headdim=64, #default to 64
        ngroups=1,
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        activation="silu", #default to silu
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=False, #default to False, for custom implementation
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
        linear_attn_duality=False,
        d_state = 64,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.headdim = headdim
        self.d_state = d_state
        if ngroups == -1:
            ngroups = self.d_inner // self.headdim #equivalent to multi-head attention
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.activation = activation
        #convert chunk_size to triton.language.int32
        self.chunk_size = chunk_size#torch.tensor(chunk_size,dtype=torch.int32)
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.partial_win_size = kwargs.get('partial_win_size', -1) #default to -1
        self.win_only = kwargs.get('win_only', False) #default to False
        self.ssd_aexp = kwargs.get('ssd_aexp', False) #default to 2
        self.ssd_positve_dA = kwargs.get('ssd_positve_dA', False) #default to False, ablation for linear attn duality
        self.ssd_norm_da = kwargs.get('ssd_norm_da', False)
        self.ssd_linear_norm = kwargs.get('ssd_linear_norm', False)
        self.win_norm = kwargs.get('win_norm', False)
        self.zact = kwargs.get('zact', False)
        self.multi_branch = kwargs.get('multi_branch', True)
        if self.ssd_linear_norm:
            self.elu = nn.ELU()
        if self.multi_branch:
            # Order: [z, x, B, C, dt]
            d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        else:
            # Order: [x, B, C, dt]
            d_in_proj = 1 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, int(d_in_proj), bias=bias, **factory_kwargs) #
        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state


        self.conv2d = nn.Conv2d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            groups=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)
        # self.conv1d.weight._no_weight_decay = True

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        # self.register_buffer("A_log", torch.zeros(self.nheads, dtype=torch.float32, device=device), persistent=True)
        self.A_log._no_weight_decay = True

        # ===== MC-NCD: token-dependent rank-r diagonal A =====
        # A_t = -softplus(W_A * SiLU(W_g * x_t) + a_bias) * Delta_t,  shape (B, L, H, S)
        # Reduces to VSSD baseline at step 0 due to zero-init of W_A.
        self.use_mc_ncd = kwargs.get('use_mc_ncd', True)
        self.mc_rank = kwargs.get('mc_rank', 8)
        if self.use_mc_ncd:
            # W_g: d_model -> r, shared across heads
            self.W_g = nn.Linear(self.d_model, self.mc_rank, bias=False, **factory_kwargs)
            # W_A: r -> H * S, per-head independent, near-zero init for safe start.
            # We use a tiny random init instead of strict zero so that W_g receives
            # non-zero gradients from step 1 (otherwise DDP with default settings
            # complains about unused parameters).  The output deviation from VSSD
            # baseline at step 0 is ~1e-6, well within float32 noise.
            self.W_A = nn.Linear(self.mc_rank, self.nheads * self.d_state, bias=False, **factory_kwargs)
            nn.init.normal_(self.W_A.weight, mean=0.0, std=1e-4)
            # a_bias: (H * S,), inverse-softplus of VSSD's initial A so that
            # at step 0:  -softplus(0 + a_bias) == -A_init  ==  VSSD's effective A.
            with torch.no_grad():
                a_init = A.unsqueeze(-1).expand(self.nheads, self.d_state).clone()  # (H, S)
                a_init = a_init.clamp(min=1e-4)
                # inverse_softplus(y) = y + log(1 - exp(-y))  (numerically stable form)
                inv_a = a_init + torch.log(-torch.expm1(-a_init))
            self.a_bias = nn.Parameter(inv_a.flatten().to(dtype=dtype))  # (H*S,)
            self.a_bias._no_weight_decay = True
        # =====================================================

        # D "skip" parameter
        if kwargs.get('dscale', False):
            self.D = Scale(dim=self.d_inner, init_value=1.0, trainable=True)
        else:
            self.D = nn.Parameter(torch.ones(self.nheads, device=device))
            self.D._no_weight_decay = True
        # modified from RMSNormGated to layer norm
        #assert RMSNormGated is not None
        #self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        self.norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        #linear attention duality
        self.linear_attn_duality = linear_attn_duality

        #lepe positional encoding
        if kwargs.get('lepe', False):
            self.lepe = nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1, groups=self.d_inner)
        else:
            self.lepe = None
        if kwargs.get('rope', False):
            HW = kwargs.get('input_resolution')#FIXME: fix the resolution in dynamic input
            self.ropes = RoPE(shape=(HW[0], HW[1], self.d_state), base=10000)
        else:
            self.ropes = None
        self.ab_bias = kwargs.get('ab_bias', False)
        self.decouple_hw = kwargs.get('decouple_hw', False)
        self.kwargs = kwargs


    def non_casual_linear_attn(self, x, dt, A, B, C, D, H=None, W=None, relpos=None, last_kv=None, A_tilde=None):
        '''
        non-casual attention duality of mamba v2
        x: (B, L, H, D), equivalent to V in attention
        dt: (B, L, nheads)
        A: (nheads) or (d_inner, d_state)
        B: (B, L, d_state), equivalent to K in attention
        C: (B, L, d_state), equivalent to Q in attention
        D: (nheads), equivalent to the skip connection
        A_tilde: (B, L, H, S) optional MC-NCD token-dependent diagonal A; if given, overrides scalar A.
        '''
        skip = x
        batch, seqlen, head, dim = x.shape
        dstate = B.shape[2]
        V = x.permute(0, 2, 1, 3) # (B, H, L, D)
        dt = dt.permute(0, 2, 1) # (B, H, L)
        if A_tilde is not None:
            # MC-NCD path: A_tilde is (B, L, H, S) -> permute to (B, H, L, S)
            A_tilde_p = A_tilde.permute(0, 2, 1, 3)                       # (B, H, L, S)
            dA = dt.unsqueeze(-1) * A_tilde_p                             # (B, H, L, S)
        else:
            dA = dt.unsqueeze(-1) * A.view(1, -1, 1, 1)                   # (B, H, L, 1)
        if self.ssd_aexp: dA = 1/dA.exp()
        if self.ssd_positve_dA: dA = -dA
        if self.ssd_norm_da: dA = dA / torch.sum(dA, dim=-2, keepdim=True)

        K = B.view(batch, 1, seqlen, dstate)# (B, 1, L, D)


        Q = C.view(batch, 1, seqlen, dstate)  # (B, 1, L, dstate)
        if self.ssd_linear_norm:
            Q, K = self.elu(Q) + 1.0, self.elu(K) + 1.0
            if self.kwargs.get('exp_da', False):
                dA = dA.softmax(dim=-2) * seqlen
            else:
                dA = dA/dA.max(dim=-2, keepdim=True)[0]

            K, K_plain = K * dA, K
            Q_pre, K_pre = Q, K
            if self.ropes is not None:
                if H!=self.ropes.rotations.shape[0] or W!=self.ropes.rotations.shape[1]:
                    Q = rope(Q.view(batch, H, W, dstate),(H,W,dstate)).view(batch, -1, dstate).unsqueeze(1)
                    K = rope(K.flatten(0,1).view(batch*head, H, W, dstate),(H,W,dstate)).view(batch, head, H, W, dstate).flatten(2,3)
                else:
                    Q = self.ropes(Q.view(batch, H, W, dstate)).view(batch, -1, dstate).unsqueeze(1) # (B, 1, L, dstate)
                    K = self.ropes(K.flatten(0,1).view(batch*head, H, W, dstate)).view(batch, head, H, W, dstate).flatten(2,3) # (B, H, L, dstate)

            if getattr(self, "__DEBUG__", False):
                A_mat = dA.cpu().detach().numpy()
                A_mat = A_mat.reshape(batch, -1, H, W)
                setattr(self, "__data__", dict(
                    dA=A_mat, H=H, W=W, V=V,Q=Q, K=K))
            Q = Q / (head * dim)#/ (head * dim)  ##  # avoid overlarge scale #(B, 1, L, dstate)
            Q_pre = Q_pre / (head * dim)#/ (head * dim)  # (Q_nomi * dstate)
            nomi = (Q_pre @ K_pre.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6) # (B, 1, L, dstate)
            KV = (K.transpose(-2, -1) * (seqlen ** -0.5)) @ (V * (seqlen ** -0.5)) # (B, H, dstate, D)
            x = (Q @ KV)
            x = x / nomi

        else:
            if self.ropes is not None:
                Q = self.ropes(Q.view(batch, H, W, dstate)).view(batch, -1, dstate).unsqueeze(1)
                K = self.ropes(K.view(batch, H, W, dstate)).view(batch, -1, dstate).unsqueeze(1)

            if self.kwargs.get('exp_da', False):
                dA = dA.softmax(dim=-2)
                Kscaled = K * dA
                KV = Kscaled.transpose(-2, -1) @ V
                x = Q @ KV
            else:
                if A_tilde is not None:
                    # MC-NCD: dA has shape (B, H, L, S), apply to K along dstate dim.
                    # K shape: (B, 1, L, S) -> broadcast to (B, H, L, S)
                    K_scaled = K * dA                                # (B, H, L, S)
                    if Q.dtype != K_scaled.dtype:
                        Q = Q.to(K_scaled.dtype)
                    KV = K_scaled.transpose(-2, -1) @ V              # (B, H, S, D)
                    x = Q @ KV                                        # (B, H, L, D)
                else:
                    V_scaled = V * dA

                    if Q.dtype != V_scaled.dtype or Q.dtype != V_scaled.dtype:
                        Q, K = Q.to(V_scaled.dtype), K.to(V_scaled.dtype)
                    KV = K.transpose(-2, -1) @ V_scaled # (B, H, dstate, D)
                    x = Q @ KV # (B, H, L, D)
        if self.kwargs.get('dscale', False):
            x = x.permute(0, 2, 1, 3).contiguous() + self.D(skip.flatten(2,3)).view(batch, seqlen, head, dim)
        else:
            x = x + V * D.view(1, -1, 1, 1).repeat(batch, 1, seqlen, 1)
            x = x.permute(0, 2, 1, 3).contiguous()  # (B, L, H, D)
        return x, KV


    def forward(self, u, H, W, relpos, seq_idx=None, last_kv = None):
        """
        u: (B,C,H,W)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        # ===== MC-NCD: compute token-dependent diagonal A_tilde =====
        # Only used in non_casual_linear_attn path; chunk-scan path keeps scalar A.
        A_tilde = None
        if self.use_mc_ncd and self.linear_attn_duality:
            g = F.silu(self.W_g(u))                                    # (B, L, r)
            a_raw = self.W_A(g) + self.a_bias                          # (B, L, H*S)
            # negative for SSD stability;  shape (B, L, H, S)
            A_tilde = -F.softplus(a_raw).view(batch, seqlen, self.nheads, self.d_state)
        # ============================================================
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        if self.multi_branch:
            zxbcdt = self.in_proj(u)  # (B, L, d_in_proj)
            z, xBC, dt = torch.split(
                zxbcdt, [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
        else:
            xbcdt = self.in_proj(u)  # (B, L, d_in_proj)
            xBC, dt = torch.split(
                xbcdt, [self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
            )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)
        assert self.activation in ["silu", "swish"]


        # #2D Convolution
        xBC = xBC.view(batch, H, W, -1).permute(0, 3, 1, 2).contiguous()
        xBC = self.act(self.conv2d(xBC))
        xBC = xBC.permute(0, 2, 3, 1).view(batch, H*W, -1).contiguous()
        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        x, dt, A, B, C = to_ttensor(x, dt, A, B, C)
        if self.linear_attn_duality:
            y, KV = self.non_casual_linear_attn(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt, A, B, C, self.D, H, W, relpos, last_kv,
                A_tilde=A_tilde
            )
        else:
            if self.kwargs.get('bidirection', False):
                #assert self.ngroups == 2 #only support bidirectional with 2 groups
                x = to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)).chunk(2, dim=-2)
                B = to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                C = to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)).chunk(2, dim=-2)
                dt = dt.chunk(2, dim=-1) # (B, L, nheads) -> (B, L, nheads//2)*2
                A, D = A.chunk(2, dim=-1), self.D.chunk(2,dim=-1) # (nheads) -> (nheads//2)*2
                y_forward = mamba_chunk_scan_combined(
                    x[0], dt[0], A[0], B[0], C[0], chunk_size=self.chunk_size, D=D[0], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y_backward = mamba_chunk_scan_combined(
                    x[1].flip(1), dt[1].flip(1), A[1], B[1].flip(1), C[1].flip(1), chunk_size=self.chunk_size, D=D[1], z=None, seq_idx=seq_idx,
                    initial_states=initial_states, **dt_limit_kwargs
                )
                y = torch.cat([y_forward, y_backward.flip(1)], dim=-2)
            else:
                y = mamba_chunk_scan_combined(
                    to_ttensor(rearrange(x, "b l (h p) -> b l h p", p=self.headdim)),
                    to_ttensor(dt),
                    to_ttensor(A),
                    to_ttensor(rearrange(B, "b l (g n) -> b l g n", g=self.ngroups)),
                    to_ttensor(rearrange(C, "b l (g n) -> b l g n", g=self.ngroups)),
                    chunk_size=self.chunk_size,
                    D=to_ttensor(self.D),
                    z=None,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                    **dt_limit_kwargs,
                )
        y = rearrange(y, "b l h p -> b l (h p)")

        # # Multiply "gate" branch and apply extra normalization layer
        # y = self.norm(y, z)
        y = self.norm(y)
        if self.multi_branch:
            if self.zact: z = self.act(z)
            y = y*z
        out = self.out_proj(y) # (B, L, D)
        #if self.kwargs.get('kv_scale', False): return out, KV
        return out


class VMAMBA2Block(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality=False, d_state = 64, first_block = False, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        kwargs['input_resolution'] = input_resolution
        self.attn_type = kwargs.get('attn_type', 'mamba2')
        if self.attn_type == 'standard':
            self.attn = StandardAttention(dim=dim, heads=num_heads, dim_head=dim // num_heads, dropout=drop, **kwargs)
        elif self.attn_type == 'mamba2':
            self.attn = Mamba2(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size,
                                linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_cpe = kwargs.get('use_cpe', True)
        if self.use_cpe:
            self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
            if kwargs.get('cpe_norm', None) is not None:
                self.cpe_norm_type = kwargs.get('cpe_norm', None)
                if self.cpe_norm_type == 'bn':
                    self.cpe_norm1 = nn.BatchNorm2d(dim)
                    self.cpe_norm2 = nn.BatchNorm2d(dim)
            else:
                self.cpe_norm_type = None
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.use_res_scale = kwargs.get('res_scale', False)
        #self.kv_scale = kwargs.get('kv_scale', False)
        if self.use_res_scale:
            self.layerscale1 = Scale(dim=dim, init_value=1e-6)
            self.res_scale1 = Scale(dim=dim, init_value=1)
            self.layerscale2 = Scale(dim=dim, init_value=1e-6)
            self.res_scale2 = Scale(dim=dim, init_value=1)

    def forward(self, x, H=None, W=None, relpos=None, last_kv=None):
        B, L, C = x.shape
        x_init = x
        kv = None
        if H & W is None:
            H, W = self.input_resolution
            assert L == H * W, "input feature has wrong size"
        # SSD or Standard Attention
        if self.use_cpe:
            if self.cpe_norm_type is not None:
                x = x + self.cpe_norm1(self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2))).flatten(2).permute(0, 2, 1)
            else:
                x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        shortcut = x
        if self.use_res_scale:
            x = self.res_scale1(shortcut) + self.layerscale1(self.drop_path(self.attn(self.norm1(x), H, W, relpos)))
        else:
            x = shortcut + self.drop_path(self.attn(self.norm1(x), H, W, relpos))
        # FFN
        if self.use_cpe:
            if self.cpe_norm_type is not None:
                x = x + self.cpe_norm2(self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2))).flatten(2).permute(0, 2, 1)
            else:
                x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        if self.use_res_scale:
            x = self.res_scale2(x) + self.layerscale2(self.drop_path(self.mlp(self.norm2(x))))
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, kv


class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256, linear_attn_duality=False, d_state=64, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            VMAMBA2Block(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                      ssd_expansion=ssd_expansion, ssd_ngroups=ssd_ngroups, ssd_chunk_size=ssd_chunk_size,
                      linear_attn_duality=linear_attn_duality, d_state=d_state, first_block=(i==0), **kwargs)
            for i in range(depth)])

        # patch merging layer
        self.rmt_downsample =  kwargs.get('rmt_downsample', False)
        if downsample is not None:
            if self.rmt_downsample:
                self.downsample = downsample(dim, dim*2)
            else:
                self.downsample = downsample(input_resolution, dim)
        else:
            self.downsample = None

        self.trans_svd = kwargs.get('trans_svd', False)
        if self.trans_svd:
            self.RelPos = kwargs.get('preset_pos')
    def forward(self, x, H=None, W=None,):
        last_kv = None
        if self.trans_svd:
            rel_pos = self.RelPos
            if rel_pos is not None:
                rel_pos[0] = rel_pos[0].type_as(x)
                rel_pos[1] = rel_pos[1].type_as(x)
        else:
            rel_pos = None
        for blk in self.blocks:
            if self.use_checkpoint:
                x, last_kv = checkpoint.checkpoint(blk, x, H, W, rel_pos, last_kv=last_kv)
            else:
                x, last_kv  = blk(x, H, W, rel_pos, last_kv=last_kv)

        if self.downsample is not None:
            if self.rmt_downsample:
                x = x.view(x.shape[0], H, W, -1)
            x = self.downsample(x, H, W)
            if self.rmt_downsample:
                x = x.flatten(1, 2)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class VMAMBA2(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=64, depths=[2, 4, 12, 4], num_heads=[2, 4, 8, 16],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality= False, d_state=64, debug=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.simple_downsample = kwargs.get('simple_downsample', False)
        self.simple_patch_embed = kwargs.get('simple_patch_embed', False)
        self.rmt_downsample = kwargs.get('rmt_downsample', False)
        self.rmt_patch_embed = kwargs.get('rmt_patch_embed', False)
        self.attn_types = kwargs.get('attn_types', ['mamba2']*len(depths))
        self.win_only = kwargs.get('win_only', [False]*len(depths))
        self.use_res_scale = kwargs.get('res_scale', [False]*len(depths))
        #param for SPT decay
        self.temp_ranges = kwargs.get('temp_ranges', [1,10,10,1000])
        self.pos_scale = kwargs.get('pos_scale', [2*math.pi, 2*math.pi, math.pi, math.pi])
        self.trans_svd = kwargs.get('trans_svd', False)
        self.spt_dims = kwargs.get('spt_dims', [64]*len(depths))
        #self.kv_scale = kwargs.get('kv_scale', [False]*len(depths))
        self.dscale = kwargs.get('dscale', [False]*len(depths))
        if d_state is not list: d_state = [d_state]*len(depths)
        if self.mlp_ratio is not list : self.mlp_ratio = [self.mlp_ratio]*len(depths)
        if kwargs.get('async_state', [None])[0] != None: #FIXME: pass state kwargs via dstate directly
            d_state = kwargs.get('async_state', [None])
        if kwargs.get('async_mlp_ratio', [None])[0] != None: #FIXME: pass state kwargs via mlp_ratio directly
            self.mlp_ratio = kwargs.get('async_mlp_ratio', [None])
        if self.simple_patch_embed:
            self.patch_embed = SimpleStem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if self.rmt_patch_embed:
                self.patch_embed = RMTPatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
            else:
                self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        if self.simple_downsample:
            PatchMergingBlock = SimplePatchMerging
        else:
            if self.rmt_downsample:
                PatchMergingBlock = RMTPatchMerging
            else:
                PatchMergingBlock = PatchMerging
        try:
            patches_resolution = self.patch_embed.patches_resolution
        except:
            patches_resolution = [to_2tuple(img_size)[0] // to_2tuple(patch_size)[0], to_2tuple(img_size)[1] // to_2tuple(patch_size)[1]]
        self.patches_resolution =  patches_resolution
        self.stage_resolutions = [56, 28, 14, 7] #FIXME: fix the resolution in dynamic input

        # absolute position embedding
        if self.ape:
            num_patches = self.patch_embed.num_patches
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        #svd position embedding
        if self.trans_svd:
            svd_emds = []
            svd_dims = kwargs.get('svd_dims', [256, 128, 64, 32])
            for i in range(self.num_layers):
                if svd_dims[i] == -1:
                    svd_emds.append(None)
                    continue
                resolution = self.stage_resolutions[i]
                PosSine = PositionEmbeddingSine(temperature=self.temp_ranges[i], scale=self.pos_scale[i])
                pos_sine = PosSine(torch.zeros(1, resolution, resolution), resolution, resolution)
                tmp_emds = get_sine_svd(pos_sine, svd_dims[i])
                svd_emds.append(tmp_emds)
            self.svd_emds = svd_emds
        else:
            self.svd_emds = [None] * self.num_layers
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs['attn_type'] = self.attn_types[i_layer]
            kwargs['win_only'] = self.win_only[i_layer]
            kwargs['res_scale'] = self.use_res_scale[i_layer]
            kwargs['dscale'] = self.dscale[i_layer]
            kwargs['preset_pos'] = self.svd_emds[i_layer]
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio[i_layer],
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMergingBlock if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               ssd_expansion=ssd_expansion,
                               ssd_ngroups=ssd_ngroups,
                               ssd_chunk_size=ssd_chunk_size,
                               linear_attn_duality = linear_attn_duality,
                               d_state = d_state[i_layer],
                               **kwargs)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        self.debug = debug
    def _init_weights(self, m):
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

    @torch.no_grad()
    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        try:
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        except Exception as e:
            #print full error message
            print('get exception', e)
            print('Error in flop_count, set to default value 1e6')
            return 1e6
        del model, input
        #manually check flops for non-casual linear attention
        linear_attn_flops = 0
        for i_layer, layer in enumerate(self.layers):
            if self.attn_types[i_layer] == 'mamba2':
                seq_len = layer.input_resolution[0] * layer.input_resolution[1]
                tmp_flops = len(layer.blocks) * 2 * layer.blocks[0].attn.d_inner * layer.blocks[0].attn.d_state * seq_len
                linear_attn_flops +=  tmp_flops
            elif self.attn_types[i_layer] == 'standard':
                seq_len = layer.input_resolution[0] * layer.input_resolution[1]
                tmp_flops = len(layer.blocks) * 2 * layer.blocks[0].attn.inner_dim * seq_len * seq_len
                linear_attn_flops +=  tmp_flops
        print(f"non_casual_linear_attn_flop: {linear_attn_flops/1e9} G")
        return sum(Gflops.values()) * 1e9

    def forward_features(self, x):
        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        if self.rmt_patch_embed:
            x = x.flatten(1,2)
        H, W = H//4, W//4 # downsampled by patch_embed

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, H, W)
            if layer != self.layers[-1]:
                H, W = H//2, W//2 # downsampled by layer
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        if self.debug:
            x = x.view(x.size(0), x.size(1), H, W)
            return x
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Backbone_VMAMBA2(VMAMBA2):
    def __init__(self, out_indices=(0, 1, 2, 3), pretrained=None, **kwargs):
        super().__init__(**kwargs)
        norm_layer = nn.LayerNorm

        self.out_indices = out_indices
        for i in out_indices:
            layer = norm_layer(self.layers[i].dim)
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)

        del self.head
        del self.norm
        del self.avgpool
        self.load_pretrained(pretrained,key=kwargs.get('key','model_ema')) #FIXME load model by default
        self.rmt_downsample = kwargs.get('rmt_downsample', False)
        self.rmt_patch_embed = kwargs.get('rmt_patch_embed', False)

    def load_pretrained(self, ckpt=None, key="model"):
        if ckpt is None:
            return

        try:
            _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"))
            print(f"Successfully load ckpt {ckpt} from {key}")
            incompatibleKeys = self.load_state_dict(_ckpt[key], strict=False)
            print(incompatibleKeys)
        except Exception as e:
            print(f"Failed loading checkpoint form {ckpt}: {e}")


    def forward(self, x):

        def layer_forward(l, x, H=None, W=None):
            for blk in l.blocks:
                x,_ = blk(x, H, W)
            if l.downsample is not None:
                #y = l.downsample(x, H, W)
                if self.rmt_downsample:
                    y = x.view(x.shape[0], H, W, -1)
                else:
                    y = x
                y = l.downsample(y, H, W)
                if self.rmt_downsample:
                    y = y.flatten(1, 2)
            else:
                y = x
            return x, y

        H, W = x.shape[-2:]
        x = self.patch_embed(x)
        if self.rmt_patch_embed:
            x = x.flatten(1,2)

        if self.simple_patch_embed:
            H, W = H//4, W//4
        else:
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
            H, W = int((H - 1) / 2) + 1, int((W - 1) / 2) + 1
        outs = []
        for i, layer in enumerate(self.layers):
            o, x = layer_forward(layer, x, H, W)  # (B, H, W, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'outnorm{i}')
                out = norm_layer(o)
                B, L, C = out.shape
                out = out.view(B, H, W, C).permute(0, 3, 1, 2) # B, C, H, W
                outs.append(out.contiguous())
            #calculate H, W for next layer, with conv stride 3, stride 2 and padding 1
            H, W = int((H-1)/2)+1, int((W-1)/2)+1

        if len(self.out_indices) == 0:
            return x

        return outs