import torch
import torch.nn as nn

from ARCHI.TRANSFORM.functions import MultiheadAttentionv4

class StyleTransferBlock_vggv1(nn.Module):
    def __init__(self, channels, attn_residual=True, use_conv=True, use_selfattn=False, n_heads=8, norm_style_key=True):
        super().__init__()
        self.use_conv = use_conv
        self.use_selfattn = use_selfattn
        if use_conv:
            self.conv_layer = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.PReLU(),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),

                nn.BatchNorm2d(channels),
                nn.PReLU(),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            )
        if self.use_selfattn:
            self.self_attn_block = MultiheadAttentionv4(in_channels=channels, n_heads=n_heads, residual=True, attn_scale=1, norm_style_key=norm_style_key)
        self.cross_attn_block = MultiheadAttentionv4(in_channels=channels, n_heads=n_heads, residual=attn_residual, attn_scale=1, norm_style_key=norm_style_key)

    def forward(self, content, style, tau=1.0):
        x = content
        if self.use_conv:
            residual = x
            x = self.conv_layer(x)
            x = x + residual
        if self.use_selfattn:
            x = self.self_attn_block(x, x, 1.0)
        x = self.cross_attn_block(x, style, tau)
        return x

class StyleTransferBlock_vggv2(nn.Module):
    def __init__(self, channels, attn_residual=True, use_conv=True, use_selfattn=True, n_heads=8, norm_style_key=True):
        super().__init__()
        self.use_conv = use_conv
        self.use_selfattn = use_selfattn
        if use_conv:
            self.conv_layer = nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.PReLU(),
                nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            )
        if self.use_selfattn:
            self.self_attn_block = MultiheadAttentionv4(in_channels=channels, n_heads=n_heads, residual=True, attn_scale=1, norm_style_key=norm_style_key)
        self.cross_attn_block = MultiheadAttentionv4(in_channels=channels, n_heads=n_heads, residual=attn_residual, attn_scale=1, norm_style_key=norm_style_key)

    def forward(self, content, style, tau=1.0):
        x = content
        if self.use_conv:
            x = self.conv_layer(x)
        if self.use_selfattn:
            x = self.self_attn_block(x, x, 1.0)
        x = self.cross_attn_block(x, style, tau)
        return x


