import torch
import numpy as np
import threading
import queue
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class ProcessingStats:
    """处理统计信息"""
    capture_fps: float = 0.0
    output_fps: float = 0.0
    frame_gen_time: float = 0.0
    sr_time: float = 0.0
    total_time: float = 0.0
    gpu_memory_used: float = 0.0

class FrameInterpolator:
    """
    帧插值主控制器
    整合屏幕捕获、帧生成、超分辨率、输出渲染
    """
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        enable_frame_gen: bool = True,
        enable_super_res: bool = True,
        target_fps: int = 120,
        target_resolution: Tuple[int, int] = (2560, 1440)
    ):
        self.config = config or {}
        self.enable_frame_gen = enable_frame_gen
        self.enable_super_res = enable_super_res
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        
        # 初始化各模块
        self._init_modules()
        
        # 处理队列
        self.input_queue = queue.Queue(maxsize=5)
        self.output_queue = queue.Queue(maxsize=10)
        
        # 状态
        self.is_running = False
        self.stats = ProcessingStats()
        
        # 性能监控
        self.enable_profiling = True
    
    def _init_modules(self):
        """初始化所有模块"""
        print("=" * 60)
        print("初始化模块...")
        print("=" * 60)
        
        # 1. 屏幕捕获
        from core.screen_capture import ScreenCapture
        self.capture = ScreenCapture(
            capture_fps=self.target_fps // 2 if self.enable_frame_gen else self.target_fps,
            use_dxgi=True
        )
        
        # 2. 帧生成
        if self.enable_frame_gen:
            from core.frame_generator import FrameGenerator
            self.frame_gen = FrameGenerator(
                model_path=self.config.get('frame_gen_model'),
                device='cuda',
                half_precision=True
            )
        
        # 3. 超分辨率
        if self.enable_super_res:
            from core.super_resolution import SuperResolution
            self.super_res = SuperResolution(
                model_path=self.config.get('sr_model'),
                scale_factor=2,
                device='cuda',
                half_precision=True
            )
        
        # 4. 显示输出
        from core.display_output import DisplayOutput
        self.display = DisplayOutput(
            target_resolution=self.target_resolution,
            fps=self.target_fps
        )
        
        print("=" * 60)
        print("✓ 所有模块初始化完成")
        print("=" * 60)
    
    def _processing_loop(self):
        """处理线程主循环"""
        while self.is_running:
            start_time = time.time()
            
            # 获取输入帧
            try:
                frame = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # 处理帧
            output_frames = self._process_frame(frame)
            
            # 输出帧
            for out_frame in output_frames:
                try:
                    self.output_queue.put_nowait(out_frame)
                except queue.Full:
                    pass
            
            # 统计
            if self.enable_profiling:
                self.stats.total_time = time.time() - start_time
                self.stats.gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
    
    def _process_frame(self, frame: np.ndarray) -> list:
        """处理单帧"""
        output_frames = []
        timer = {}
        
        # 1. 帧生成
        if self.enable_frame_gen:
            t0 = time.time()
            interpolated_frames = self.frame_gen.process_stream(frame, interpolate=True)
            timer['frame_gen'] = time.time() - t0
        else:
            interpolated_frames = [frame]
            timer['frame_gen'] = 0.0
        
        # 2. 超分辨率
        if self.enable_super_res:
            t0 = time.time()
            for f in interpolated_frames:
                sr_frame = self.super_res.upscale_frame(f, self.target_resolution)
                output_frames.append(sr_frame)
            timer['sr'] = time.time() - t0
        else:
            output_frames = interpolated_frames
            timer['sr'] = 0.0
        
        # 更新统计
        self.stats.frame_gen_time = timer.get('frame_gen', 0.0)
        self.stats.sr_time = timer.get('sr', 0.0)
        
        return output_frames
    
    def start(self):
        """启动处理"""
        self.is_running = True
        
        # 启动捕获
        self.capture.start()
        
        # 启动处理线程
        self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.process_thread.start()
        
        # 启动显示
        self.display.start()
        
        print(f"\n✓ 帧插值系统已启动")
        print(f"  目标FPS: {self.target_fps}")
        print(f"  目标分辨率: {self.target_resolution}")
        print(f"  帧生成: {'启用' if self.enable_frame_gen else '禁用'}")
        print(f"  超分辨率: {'启用' if self.enable_super_res else '禁用'}")
    
    def stop(self):
        """停止处理"""
        self.is_running = False
        
        self.capture.stop()
        self.display.stop()
        
        print("\n✓ 帧插值系统已停止")
    
    def get_stats(self) -> ProcessingStats:
        """获取统计信息"""
        self.stats.capture_fps = self.capture.get_fps()
        self.stats.output_fps = self.display.get_fps()
        return self.stats
    
    def run_with_gui(self):
        """带GUI的运行模式"""
        from gui.control_panel import ControlPanel
        
        # 启动处理
        self.start()
        
        # 启动GUI
        gui = ControlPanel(self)
        gui.run()
        
        # 停止
        self.stop()