import torch
import numpy as np
import threading
import queue
import time
import cv2
from typing import Tuple, Optional

class DisplayOutput:
    """屏幕输出渲染器"""
    
    def __init__(
        self,
        target_resolution: Tuple[int, int] = (2560, 1440),
        fps: int = 120,
        window_name: str = "Frame Interpolator Output",
        fullscreen: bool = False
    ):
        self.target_resolution = target_resolution
        self.target_fps = fps
        self.window_name = window_name
        self.fullscreen = fullscreen
        
        # 帧队列
        self.frame_queue = queue.Queue(maxsize=10)
        
        # 状态
        self.is_running = False
        self.display_thread = None
        
        # FPS统计
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
    
    def _display_loop(self):
        """显示线程主循环"""
        # 创建窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        if self.fullscreen:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.resizeWindow(
                self.window_name,
                self.target_resolution[0] // 2,
                self.target_resolution[1] // 2
            )
        
        frame_time = 1.0 / self.target_fps
        
        while self.is_running:
            start_time = time.time()
            
            # 获取帧
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # 显示
            cv2.imshow(self.window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)  # 必需，否则窗口不更新
            
            # FPS统计
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.current_fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, frame_time - elapsed)
            time.sleep(sleep_time)
        
        cv2.destroyWindow(self.window_name)
    
    def start(self):
        """启动显示"""
        self.is_running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        print(f"✓ 显示输出已启动 (目标: {self.target_fps} FPS)")
    
    def stop(self):
        """停止显示"""
        self.is_running = False
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        print("✓ 显示输出已停止")
    
    def display_frame(self, frame: np.ndarray):
        """显示单帧"""
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # 丢弃旧帧
    
    def get_fps(self) -> float:
        """获取显示FPS"""
        return self.current_fps