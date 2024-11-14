
import torch
from torch import nn
from timm.models.layers import trunc_normal_
from einops import rearrange
import typing as t
import torch.nn.functional as F
import pywt
from functools import partial


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x

def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x

class FGA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num:int=2,
            window_size: int = 7,
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg: t.Dict = dict(type='ReLU'),
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
            wave='db1',
    ):
        super(FGA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim // 4, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4


        self.dec_filters, self.rec_filters = create_wavelet_filter(wave, dim, dim, torch.float)
        self.dec_filters = nn.Parameter(self.dec_filters, requires_grad=False)
        self.rec_filters = nn.Parameter(self.rec_filters, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.dec_filters)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.rec_filters)

        self.conv1x1_ll = nn.Conv1d(dim, group_chans, kernel_size=1, bias=False)
        self.conv1x1_lh = nn.Conv1d(dim, group_chans, kernel_size=1, bias=False)
        self.conv1x1_hl = nn.Conv1d(dim, group_chans, kernel_size=1, bias=False)
        self.conv1x1_hh = nn.Conv1d(dim, group_chans, kernel_size=1, bias=False)
        self.convx = nn.Conv2d(dim*4, dim,kernel_size=1, bias=False)
        self.convx1 = nn.Conv2d(dim, dim*4,kernel_size=1, bias=False)
        self.sa_gate = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h = nn.GroupNorm(4, dim)
        self.norm_w = nn.GroupNorm(4, dim)

        self.conv_d = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()

        if window_size == -1:
            self.down_func = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func = self.space_to_chans
                # dimensionality reduction
                self.conv_d = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1, bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
            elif down_sample_mode == 'max_pool':
                self.down_func = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:


        orig_h, orig_w = x.size(2), x.size(3)

        
        """
        The dim of x is (B, C, H, W)
        """
        # Spatial attention priority calculation
        b, c, h_, w_ = x.size()
        x_wave = self.wt_function(x) 

        LL = x_wave[:, :, 0, :, :]
        HL = x_wave[:, :, 1, :, :]
        LH = x_wave[:, :, 2, :, :]
        HH = x_wave[:, :, 3, :, :]


        ll_h = LL.mean(dim=3)
        lh_h = LH.mean(dim=3)
        hl_h = HL.mean(dim=3)
        hh_h = HH.mean(dim=3)


        ll_w = LL.mean(dim=2)
        lh_w = LH.mean(dim=2)
        hl_w = HL.mean(dim=2)
        hh_w = HH.mean(dim=2)

        ll_h = self.conv1x1_ll(ll_h)
        lh_h = self.conv1x1_lh(lh_h)
        hl_h = self.conv1x1_hl(hl_h)
        hh_h = self.conv1x1_hh(hh_h)


        ll_w = self.conv1x1_ll(ll_w)
        lh_w = self.conv1x1_lh(lh_w)
        hl_w = self.conv1x1_hl(hl_w)
        hh_w = self.conv1x1_hh(hh_w)
        

        x_h_attn = self.sa_gate(self.norm_h(torch.cat((ll_h, lh_h, hl_h, hh_h), dim=1)))

        
        x_h_attn = x_h_attn.view(x.size(0), x.size(1), x.size(2) // 2, 1)

        x_w_attn = self.sa_gate(self.norm_w(torch.cat((ll_w, lh_w, hl_w, hh_w), dim=1)))
        x_w_attn = x_w_attn.view(x.size(0), x.size(1), 1, x.size(3) //2 )
        
        x_w_attn = x_w_attn.view(b, c, 1, w_ // 2)

        x = torch.cat((LL,HL,LH,HH),dim=1)
        x = self.convx(x)
        x = x * x_h_attn * x_w_attn
        # Channel attention based on self attention
        # reduce calculations
        y = self.down_func(x)
        y = self.conv_d(y)
        _, _, h_, w_ = y.size()

        # normalization first, then reshape -> (B, H, W, C) -> (B, C, H * W) and generate q, k and v
        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)
        # (B, C, H, W) -> (B, head_num, head_dim, N)
        q = rearrange(q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        k = rearrange(k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))
        v = rearrange(v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)', head_num=int(self.head_num),
                      head_dim=int(self.head_dim))

        # (B, head_num, head_dim, head_dim)
        attn = q @ k.transpose(-2, -1) * self.scaler
        attn = self.attn_drop(attn.softmax(dim=-1))
        # (B, head_num, head_dim, N)
        attn = attn @ v
        # (B, C, H_, W_)
        attn = rearrange(attn, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_), w=int(w_))
        # (B, C, 1, 1)
        attn = attn.mean((2, 3), keepdim=True)
        attn = self.ca_gate(attn)

        attn = attn * x

        attn = self.convx1(attn)
        res_wave = attn.view(x_wave.shape[0], self.dim, 4, attn.shape[2], attn.shape[3])
        x_recon = self.iwt_function(res_wave)  # [N, C, H, W]
        if orig_h % 2 != 0 or orig_w % 2 != 0:
            x_recon = F.pad(x_recon, (0, orig_w % 2, 0, orig_h % 2))
        return x_recon
    


if __name__ == '__main__':
    input1 = torch.randn(2, 128, 64, 64)
    input2 = torch.randn(2, 128, 64, 64)
