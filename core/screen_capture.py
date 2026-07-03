from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


CaptureRegion = Tuple[int, int, int, int]


@dataclass
class CaptureStats:
    backend: str = "unknown"
    fps: float = 0.0
    width: int = 0
    height: int = 0


class ScreenCapture:
    """Screen capture runtime with dxcam first and GDI fallback on Windows."""

    def __init__(
        self,
        capture_fps: int = 60,
        capture_region: Optional[CaptureRegion] = None,
        use_dxgi: bool = True,
    ):
        self.capture_fps = capture_fps
        self.capture_region = capture_region
        self.use_dxgi = use_dxgi
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
        self.is_running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        self.backend = "unknown"
        self.camera = None
        self.stats = CaptureStats()

        if use_dxgi and self._init_dxcam():
            return
        self._init_gdi()

    def _init_dxcam(self) -> bool:
        try:
            import dxcam

            self.camera = dxcam.create(output_color="RGB")
            self.backend = "dxcam"
            self._update_dimensions_from_region()
            return True
        except Exception:
            self.camera = None
            return False

    def _init_gdi(self) -> None:
        import win32api
        import win32gui

        self.hwnd = win32gui.GetDesktopWindow()
        if self.capture_region:
            _, _, width, height = self.capture_region
        else:
            width = win32api.GetSystemMetrics(0)
            height = win32api.GetSystemMetrics(1)
        self.backend = "gdi"
        self.stats = CaptureStats(backend=self.backend, width=width, height=height)

    def _update_dimensions_from_region(self) -> None:
        if self.capture_region:
            _, _, width, height = self.capture_region
        else:
            try:
                frame = self.camera.grab()
                height, width = frame.shape[:2] if frame is not None else (0, 0)
            except Exception:
                width, height = 0, 0
        self.stats = CaptureStats(backend=self.backend, width=width, height=height)

    def capture_frame_dxgi(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("dxcam backend is not initialized")
        region = None
        if self.capture_region:
            x, y, width, height = self.capture_region
            region = (x, y, x + width, y + height)
        frame = self.camera.grab(region=region)
        if frame is None:
            raise RuntimeError("dxcam did not return a frame")
        return np.ascontiguousarray(frame[:, :, :3])

    def capture_frame_gdi(self) -> np.ndarray:
        import win32con
        import win32gui
        import win32ui

        if self.capture_region:
            left, top, width, height = self.capture_region
        else:
            left, top = 0, 0
            width, height = self.stats.width, self.stats.height

        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        src_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        mem_dc = src_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(src_dc, width, height)
        mem_dc.SelectObject(bitmap)
        mem_dc.BitBlt((0, 0), (width, height), src_dc, (left, top), win32con.SRCCOPY)

        bmp = bitmap.GetBitmapBits(True)
        frame = np.frombuffer(bmp, dtype=np.uint8).reshape((height, width, 4))
        frame = frame[:, :, :3][:, :, ::-1]

        win32gui.DeleteObject(bitmap.GetHandle())
        mem_dc.DeleteDC()
        src_dc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwnd_dc)
        return np.ascontiguousarray(frame)

    def _capture_once(self) -> np.ndarray:
        if self.backend == "dxcam":
            return self.capture_frame_dxgi()
        return self.capture_frame_gdi()

    def _capture_loop(self) -> None:
        frame_time = 1.0 / max(1, self.capture_fps)
        while self.is_running:
            start_time = time.time()
            try:
                frame = self._capture_once()
                self._push_latest(frame)
            except Exception:
                time.sleep(frame_time)
                continue

            self.fps_counter += 1
            now = time.time()
            if now - self.last_fps_time >= 1.0:
                self.current_fps = self.fps_counter / (now - self.last_fps_time)
                self.fps_counter = 0
                self.last_fps_time = now
                self.stats.fps = self.current_fps

            elapsed = time.time() - start_time
            time.sleep(max(0.0, frame_time - elapsed))

    def _push_latest(self, frame: np.ndarray) -> None:
        while self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        self.frame_queue.put_nowait(frame)

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def stop(self) -> None:
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None

    def get_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_fps(self) -> float:
        return self.current_fps

    def get_stats(self) -> CaptureStats:
        self.stats.fps = self.current_fps
        return self.stats
