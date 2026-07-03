from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def check_torch() -> None:
    import torch

    print(f"torch: {torch.__version__}")
    print(f"cuda runtime: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if not torch.__version__.startswith("2.6.0"):
        raise RuntimeError("Expected torch 2.6.0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    x = torch.randn(256, 256, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"gpu smoke mean: {float(y.mean().item()):.6f}")


def check_imports() -> None:
    import cv2
    import mss
    import numpy
    import PySide6

    print(f"numpy: {numpy.__version__}")
    print(f"cv2: {cv2.__version__}")
    print(f"PySide6: {PySide6.__version__}")
    print(f"mss: {mss.__name__}")
    if platform.system() == "Windows":
        import dxcam
        import win32gui

        print(f"dxcam: {dxcam.__name__}")
        print(f"win32gui: {win32gui.__name__}")
    else:
        print("dxcam: skipped on non-Windows")
        print("win32gui: skipped on non-Windows")


def check_qt() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from app.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    print(f"qt window: {window.windowTitle()}")
    print(f"profiles: {window.profile_list.count()}")
    window.close()
    app.quit()


def main() -> int:
    check_torch()
    check_imports()
    check_qt()
    print("environment check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
