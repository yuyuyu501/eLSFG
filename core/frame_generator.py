import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class FrameGenerator:
    """帧生成器：从两帧生成中间帧"""
    
    def __init__(
        self,
        model_path: str = None,
        device: str = 'cuda',
        half_precision: bool = True
    ):
        self.device = device
        self.half = half_precision
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 帧缓存
        self.prev_frame = None
        self.frame_count = 0
    
    def _load_model(self, model_path: str):
        """加载帧生成模型"""
        # 简化的模型定义（实际从文件加载）
        from models.transformer.frame_gen_transformer import FrameGenTransformer
        
        model = FrameGenTransformer(
            dim=64,
            depth=6,
            num_heads=4
        ).to(self.device)
        
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model'])
            print(f"✓ 模型已加载: {model_path}")
        
        if self.half:
            model = model.half()
        
        model.eval()
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """预处理帧：numpy → tensor"""
        # 归一化到 [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # HWC → CHW
        frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        
        # 转移到GPU
        frame = frame.to(self.device)
        
        if self.half:
            frame = frame.half()
        
        return frame
    
    def postprocess_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """后处理：tensor → numpy"""
        # CHW → HWC
        frame = tensor.squeeze(0).permute(1, 2, 0)
        
        # [0, 1] → [0, 255]
        frame = (frame.float() * 255).clamp(0, 255).byte()
        
        # 转移到CPU
        frame = frame.cpu().numpy()
        
        return frame
    
    def generate_frame(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        t: float = 0.5
    ) -> np.ndarray:
        """
        生成中间帧
        
        Args:
            frame1: 前一帧
            frame2: 后一帧
            t: 时间参数 (0.5表示正中间)
        
        Returns:
            生成的中间帧
        """
        # 预处理
        tensor1 = self.preprocess_frame(frame1)
        tensor2 = self.preprocess_frame(frame2)
        
        # 推理
        with torch.no_grad():
            mid_frame = self.model(tensor1, tensor2, t)
        
        # 后处理
        result = self.postprocess_frame(mid_frame)
        
        return result
    
    def process_stream(
        self,
        input_frame: np.ndarray,
        interpolate: bool = True
    ) -> list:
        """
        流式处理：输入一帧，输出当前帧+插值帧
        
        Args:
            input_frame: 当前输入帧
            interpolate: 是否生成插值帧
        
        Returns:
            帧列表：[当前帧] 或 [当前帧, 插值帧]
        """
        output_frames = [input_frame]
        
        if interpolate and self.prev_frame is not None:
            # 生成插值帧
            mid_frame = self.generate_frame(self.prev_frame, input_frame)
            output_frames.insert(0, mid_frame)  # 插值帧在前
        
        self.prev_frame = input_frame.copy()
        self.frame_count += 1
        
        return output_frames