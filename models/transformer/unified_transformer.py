import torch
import torch.nn as nn
import torch.nn.functional as F
from .frame_gen_transformer import FrameGenTransformer
from .sr_transformer import SRTransformer

class UnifiedFrameSRModel(nn.Module):
    """统一模型：帧生成 + 超分"""
    
    def __init__(self, frame_gen_dim=64, frame_gen_depth=6, sr_dim=96, sr_depth=8):
        super().__init__()
        self.frame_gen = FrameGenTransformer(dim=frame_gen_dim, depth=frame_gen_depth)
        self.super_res = SRTransformer(dim=sr_dim, depth=sr_depth)
    
    def forward(self, frame1_lr, frame2_lr):
        """
        Args:
            frame1_lr: (B, 3, H, W) 低分辨率帧1
            frame2_lr: (B, 3, H, W) 低分辨率帧2
        Returns:
            mid_frame_lr: (B, 3, H, W) 生成的中间帧（低分辨率）
            mid_frame_hr: (B, 3, H*2, W*2) 生成的中间帧（高分辨率）
        """
        # 阶段1：生成中间帧（低分辨率）
        mid_frame_lr = self.frame_gen(frame1_lr, frame2_lr)
        
        # 阶段2：超分辨率（低分辨率 → 高分辨率）
        mid_frame_hr = self.super_res(mid_frame_lr)
        
        return mid_frame_lr, mid_frame_hr