from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Install dependencies with: pip install -r requirements.txt")
        return 1

    from app.main_window import MainWindow

    qt_app = QApplication(sys.argv)
    qt_app.setApplicationName("eLSFG")
    qt_app.setOrganizationName("eLSFG")
    window = MainWindow()
    window.show()
    return qt_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
