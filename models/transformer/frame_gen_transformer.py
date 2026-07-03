import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class EfficientAttention(nn.Module):
    """线性复杂度的局部注意力机制"""
    
    def __init__(self, dim, num_heads=4, window_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        # 深度可分离的QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # 窗口划分
        x = x.view(B, C, H // self.window_size, self.window_size, 
                   W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, self.window_size**2, C)
        
        # 计算注意力（窗口内）
        qkv = self.qkv(x).reshape(-1, self.window_size**2, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).reshape(-1, self.window_size**2, C)
        x = self.proj(x)
        
        # 恢复形状
        x = x.view(B, H // self.window_size, W // self.window_size, 
                   self.window_size, self.window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        
        return x

class LightweightTransformerBlock(nn.Module):
    """轻量Transformer块"""
    
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP用深度可分离卷积替代
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * mlp_ratio), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ratio), dim, 1)
        )
    
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        
        # 注意力
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x_norm)
        
        # MLP
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.mlp(x_norm)
        
        return x

class FrameGenTransformer(nn.Module):
    """帧生成Transformer：从两帧生成中间帧"""
    
    def __init__(self, dim=64, depth=6, num_heads=4):
        super().__init__()
        
        # 输入编码：两帧 → 特征
        self.input_proj = nn.Sequential(
            nn.Conv2d(6, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        
        # Transformer块堆叠
        self.blocks = nn.ModuleList([
            LightweightTransformerBlock(dim, num_heads)
            for _ in range(depth)
        ])
        
        # 输出解码
        self.output_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 3, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, frame1, frame2, t=0.5):
        """
        Args:
            frame1: (B, 3, H, W)
            frame2: (B, 3, H, W)
            t: 时间参数 (0.5表示正中间)
        Returns:
            mid_frame: (B, 3, H, W)
        """
        # 拼接两帧
        x = torch.cat([frame1, frame2], dim=1)  # (B, 6, H, W)
        
        # 编码
        x = self.input_proj(x)
        
        # Transformer处理
        for block in self.blocks:
            x = block(x)
        
        # 解码
        mid_frame = self.output_proj(x)
        
        return mid_frame