import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def pad_to_window(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = x.shape
    pad_h = (math.ceil(h / window_size) * window_size) - h
    pad_w = (math.ceil(w / window_size) * window_size) - w
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, pad_h, pad_w


def crop_padding(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h:
        x = x[:, :, :-pad_h, :]
    if pad_w:
        x = x[:, :, :, :-pad_w]
    return x


def base_upsample(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    return F.interpolate(
        x,
        scale_factor=scale_factor,
        mode="bilinear",
        align_corners=False,
    )


def zero_init_last_conv(module: nn.Module) -> None:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            nn.init.zeros_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
            return


class WindowAttention(nn.Module):
    """Local window attention for image features."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size
        if h % ws != 0 or w % ws != 0:
            raise ValueError("feature size must be padded to the window size")

        windows = x.view(b, c, h // ws, ws, w // ws, ws)
        windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, c)

        qkv = self.qkv(windows)
        qkv = qkv.reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, c)
        out = self.proj(out)

        out = out.view(b, h // ws, w // ws, ws, ws, c)
        return out.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)


class LightweightTransformerBlock(nn.Module):
    """Small transformer block with local attention and convolutional MLP."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm)
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x + self.mlp(x_norm)


class DepthwiseConvBlock(nn.Module):
    """Cheap local feature block used by faster SR variants."""

    def __init__(self, dim: int, expansion: float = 2.0):
        super().__init__()
        hidden_dim = int(dim * expansion)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SharedAttentionGroup(nn.Module):
    """One attention block followed by cheap conv refinement blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        conv_blocks: int = 1,
    ):
        super().__init__()
        self.attn = LightweightTransformerBlock(dim, num_heads, window_size)
        self.conv = nn.Sequential(*[DepthwiseConvBlock(dim) for _ in range(conv_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.attn(x))


class DownsampledDetailAttentionBlock(nn.Module):
    """CAMixer-inspired static block: conv for all pixels, attention on pooled features."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        attention_stride: int = 2,
    ):
        super().__init__()
        self.attention_stride = max(1, attention_stride)
        self.conv_path = DepthwiseConvBlock(dim)
        self.detail_gate = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid(),
        )
        self.attn = LightweightTransformerBlock(dim, num_heads, window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.conv_path(x)
        if self.attention_stride > 1:
            pooled = F.avg_pool2d(x, self.attention_stride, self.attention_stride)
            pooled, pad_h, pad_w = pad_to_window(pooled, self.attn.attn.window_size)
            attended = crop_padding(self.attn(pooled), pad_h, pad_w)
            attended = F.interpolate(
                attended,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            padded, pad_h, pad_w = pad_to_window(x, self.attn.attn.window_size)
            attended = crop_padding(self.attn(padded), pad_h, pad_w)
        gate = self.detail_gate(x)
        return conv + attended * gate


class PixelShuffleUpsampler(nn.Module):
    """PixelShuffle upsampler for x1, x2, x3, and x4 output scales."""

    def __init__(self, dim: int, scale_factor: int):
        super().__init__()
        if scale_factor == 1:
            self.net = nn.Conv2d(dim, 3, 3, padding=1)
        elif scale_factor in (2, 3):
            self.net = nn.Sequential(
                nn.Conv2d(dim, dim * scale_factor * scale_factor, 3, padding=1),
                nn.PixelShuffle(scale_factor),
                nn.GELU(),
                nn.Conv2d(dim, 3, 3, padding=1),
            )
        elif scale_factor == 4:
            self.net = nn.Sequential(
                nn.Conv2d(dim, dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(dim, dim * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(dim, 3, 3, padding=1),
            )
        else:
            raise ValueError("scale_factor must be one of 1, 2, 3, or 4")
        zero_init_last_conv(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SRTransformer(nn.Module):
    """Lightweight image super-resolution transformer.

    The model returns an image with shape (B, 3, H * scale, W * scale). It pads
    features internally so arbitrary input sizes can pass through windowed
    attention, then crops the feature map before upsampling.
    """

    def __init__(
        self,
        dim: int = 96,
        depth: int = 8,
        num_heads: int = 6,
        scale_factor: int = 2,
        window_size: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        if scale_factor not in (1, 2, 3, 4):
            raise ValueError("scale_factor must be one of 1, 2, 3, or 4")
        self.scale_factor = scale_factor
        self.window_size = window_size
        self.residual_scale = residual_scale

        self.input_proj = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.blocks = nn.ModuleList(
            [
                LightweightTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                )
                for _ in range(depth)
            ]
        )
        self.reconstruct = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
        )
        self.upscale = PixelShuffleUpsampler(dim, scale_factor)

    def _pad_to_window(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        return pad_to_window(x, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = base_upsample(x, self.scale_factor)

        feat = self.input_proj(x)
        feat, pad_h, pad_w = self._pad_to_window(feat)
        for block in self.blocks:
            feat = block(feat)
        feat = crop_padding(feat, pad_h, pad_w)

        residual = self.upscale(self.reconstruct(feat))
        return (base + residual * self.residual_scale).clamp(0.0, 1.0)


class HybridSRTransformer(nn.Module):
    """ESRT-style candidate: mostly convolution with a few transformer blocks."""

    def __init__(
        self,
        dim: int = 32,
        depth: int = 4,
        num_heads: int = 4,
        scale_factor: int = 3,
        window_size: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.residual_scale = residual_scale
        attn_count = max(1, depth // 3)
        conv_count = max(1, depth - attn_count)
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GELU(),
        )
        blocks = [DepthwiseConvBlock(dim) for _ in range(conv_count)]
        blocks += [
            LightweightTransformerBlock(dim, num_heads, window_size)
            for _ in range(attn_count)
        ]
        self.blocks = nn.Sequential(*blocks)
        self.reconstruct = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())
        self.upscale = PixelShuffleUpsampler(dim, scale_factor)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = base_upsample(x, self.scale_factor)
        feat = self.input_proj(x)
        feat, pad_h, pad_w = pad_to_window(feat, self.window_size)
        feat = crop_padding(self.blocks(feat), pad_h, pad_w)
        residual = self.upscale(self.reconstruct(feat))
        return (base + residual * self.residual_scale).clamp(0.0, 1.0)


class AttentionSharingSRTransformer(nn.Module):
    """ASID-inspired candidate: fewer attention calls, conv refinement in groups."""

    def __init__(
        self,
        dim: int = 32,
        depth: int = 4,
        num_heads: int = 4,
        scale_factor: int = 3,
        window_size: int = 8,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.window_size = window_size
        self.residual_scale = residual_scale
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GELU(),
        )
        groups = max(1, (depth + 1) // 2)
        self.groups = nn.ModuleList(
            [
                SharedAttentionGroup(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    conv_blocks=1,
                )
                for _ in range(groups)
            ]
        )
        self.reconstruct = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())
        self.upscale = PixelShuffleUpsampler(dim, scale_factor)

    def _pad_to_window(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        return pad_to_window(x, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = base_upsample(x, self.scale_factor)
        feat = self.input_proj(x)
        feat, pad_h, pad_w = self._pad_to_window(feat)
        for group in self.groups:
            feat = group(feat)
        feat = crop_padding(feat, pad_h, pad_w)
        residual = self.upscale(self.reconstruct(feat))
        return (base + residual * self.residual_scale).clamp(0.0, 1.0)


class DetailAwareSRTransformer(nn.Module):
    """CAMixer-like candidate with full-res conv and low-res gated attention."""

    def __init__(
        self,
        dim: int = 32,
        depth: int = 4,
        num_heads: int = 4,
        scale_factor: int = 3,
        window_size: int = 8,
        residual_scale: float = 0.1,
        attention_stride: int = 2,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.residual_scale = residual_scale
        self.input_proj = nn.Sequential(
            nn.Conv2d(3, dim, 3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[
                DownsampledDetailAttentionBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    attention_stride=attention_stride,
                )
                for _ in range(max(1, depth))
            ]
        )
        self.reconstruct = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())
        self.upscale = PixelShuffleUpsampler(dim, scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = base_upsample(x, self.scale_factor)
        feat = self.blocks(self.input_proj(x))
        residual = self.upscale(self.reconstruct(feat))
        return (base + residual * self.residual_scale).clamp(0.0, 1.0)


def build_sr_model(
    variant: str = "baseline",
    dim: int = 48,
    depth: int = 4,
    num_heads: int = 4,
    scale_factor: int = 3,
    window_size: int = 8,
    residual_scale: float = 0.1,
) -> nn.Module:
    variant = variant.lower()
    if variant in {"baseline", "sr_transformer"}:
        return SRTransformer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            scale_factor=scale_factor,
            window_size=window_size,
            residual_scale=residual_scale,
        )
    if variant in {"hybrid", "esrt"}:
        return HybridSRTransformer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            scale_factor=scale_factor,
            window_size=window_size,
            residual_scale=residual_scale,
        )
    if variant in {"shared_attention", "asid"}:
        return AttentionSharingSRTransformer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            scale_factor=scale_factor,
            window_size=window_size,
            residual_scale=residual_scale,
        )
    if variant in {"detail_aware", "camixer"}:
        return DetailAwareSRTransformer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            scale_factor=scale_factor,
            window_size=window_size,
            residual_scale=residual_scale,
        )
    raise ValueError(f"Unknown SR model variant: {variant}")
