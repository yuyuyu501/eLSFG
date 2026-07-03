from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.config import AppProfile, ProfileStore
from core.super_resolution import SuperResolutionConfig, SuperResolutionEngine


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.store = ProfileStore()
        self.profiles = self.store.load_all()
        self.engine: SuperResolutionEngine | None = None
        self.running = False

        self.setWindowTitle("eLSFG")
        self.resize(1120, 720)
        self.setMinimumSize(980, 640)
        self._build_ui()
        self._apply_styles()
        self._load_profile_list()
        self.profile_list.setCurrentRow(0)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        root_layout.addWidget(self._build_profile_panel())
        root_layout.addWidget(self._build_control_panel(), stretch=1)
        self.setCentralWidget(root)

    def _build_profile_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("Profiles")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.profile_list = QListWidget()
        self.profile_list.currentRowChanged.connect(self._on_profile_selected)
        layout.addWidget(self.profile_list, stretch=1)

        self.save_button = QPushButton("Save")
        self.save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.save_button.clicked.connect(self._save_current_profile)
        layout.addWidget(self.save_button)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        header = self._build_header()
        layout.addWidget(header)

        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._build_target_group(), 0, 0)
        grid.addWidget(self._build_scaling_group(), 0, 1)
        grid.addWidget(self._build_super_resolution_group(), 1, 0)
        grid.addWidget(self._build_run_group(), 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        layout.addWidget(self._build_preview_group(), stretch=1)
        return panel

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Header")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)

        title = QLabel("eLSFG Control Console")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Super Resolution MVP")
        subtitle.setObjectName("AppSubtitle")

        title_box = QVBoxLayout()
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box, stretch=1)

        self.gpu_label = QLabel(self._device_label())
        self.gpu_label.setObjectName("Badge")
        layout.addWidget(self.gpu_label)
        return frame

    def _build_target_group(self) -> QGroupBox:
        group = QGroupBox("Target")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignLeft)
        self.target_label = QLabel("Debug preview frame")
        self.capture_api_combo = QComboBox()
        self.capture_api_combo.addItems(["debug-preview", "screen-gdi", "dxcam", "wgc-next"])
        self.hotkey_label = QLabel("Alt+S")
        self.always_on_top_check = QCheckBox("Keep output on top")
        form.addRow("Source", self.target_label)
        form.addRow("Capture API", self.capture_api_combo)
        form.addRow("Hotkey", self.hotkey_label)
        form.addRow("", self.always_on_top_check)
        return group

    def _build_scaling_group(self) -> QGroupBox:
        group = QGroupBox("Scaling")
        form = QFormLayout(group)
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(1, 4)
        self.scale_spin.setValue(2)
        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 7680)
        self.width_spin.setSingleStep(160)
        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 4320)
        self.height_spin.setSingleStep(90)
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 100)
        self.sharpness_slider.setValue(20)
        form.addRow("Scale", self.scale_spin)
        form.addRow("Target width", self.width_spin)
        form.addRow("Target height", self.height_spin)
        form.addRow("Sharpness", self.sharpness_slider)
        return group

    def _build_super_resolution_group(self) -> QGroupBox:
        group = QGroupBox("Super Resolution")
        form = QFormLayout(group)
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["bicubic", "bilinear", "nearest", "sr_transformer", "auto"])
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["fast", "balanced", "quality"])
        self.tile_combo = QComboBox()
        self.tile_combo.addItems(["0", "256", "384", "512"])
        self.tile_overlap_spin = QSpinBox()
        self.tile_overlap_spin.setRange(0, 128)
        self.tile_overlap_spin.setValue(16)
        self.half_precision_check = QCheckBox("Use FP16 on CUDA")
        self.half_precision_check.setChecked(True)

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(6)
        self.model_path_label = QLabel("No model selected")
        self.model_path_label.setObjectName("PathLabel")
        browse_button = QPushButton("Browse")
        browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse_button.clicked.connect(self._browse_model)
        model_layout.addWidget(self.model_path_label, stretch=1)
        model_layout.addWidget(browse_button)

        form.addRow("Backend", self.backend_combo)
        form.addRow("Quality", self.quality_combo)
        form.addRow("Model", model_row)
        form.addRow("Tile size", self.tile_combo)
        form.addRow("Tile overlap", self.tile_overlap_spin)
        form.addRow("", self.half_precision_check)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Run")
        layout = QVBoxLayout(group)
        self.start_button = QPushButton("Start")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.setMinimumHeight(38)
        self.start_button.clicked.connect(self._toggle_running)
        layout.addWidget(self.start_button)

        self.preview_button = QPushButton("Run Preview / Benchmark")
        self.preview_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.preview_button.setMinimumHeight(34)
        self.preview_button.clicked.connect(self._run_preview)
        layout.addWidget(self.preview_button)

        self.stats_label = QLabel("Latency: -\nOutput: -\nMemory: -")
        self.stats_label.setObjectName("StatsLabel")
        layout.addWidget(self.stats_label)
        layout.addStretch(1)
        return group

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("Preview")
        layout = QHBoxLayout(group)
        layout.setSpacing(12)
        self.before_label = self._preview_label("Before")
        self.after_label = self._preview_label("After")
        layout.addWidget(self.before_label)
        layout.addWidget(self.after_label)
        return group

    def _preview_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("PreviewPane")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(320, 200)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return label

    def _load_profile_list(self) -> None:
        self.profile_list.clear()
        for profile in self.profiles:
            self.profile_list.addItem(profile.name)

    def _on_profile_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.profiles):
            return
        self._apply_profile(self.profiles[row])

    def _apply_profile(self, profile: AppProfile) -> None:
        self.backend_combo.setCurrentText(profile.backend)
        self.scale_spin.setValue(profile.scale_factor)
        self.width_spin.setValue(profile.target_width)
        self.height_spin.setValue(profile.target_height)
        self.quality_combo.setCurrentText(profile.quality)
        self.model_path_label.setText(profile.model_path or "No model selected")
        self.tile_combo.setCurrentText(str(profile.tile_size))
        self.tile_overlap_spin.setValue(profile.tile_overlap)
        self.half_precision_check.setChecked(profile.half_precision)
        self.capture_api_combo.setCurrentText(profile.capture_api)
        self.hotkey_label.setText(profile.hotkey)
        self.always_on_top_check.setChecked(profile.always_on_top)
        self.status_label.setText(f"Loaded profile: {profile.name}")

    def _profile_from_fields(self) -> AppProfile:
        row = self.profile_list.currentRow()
        name = self.profiles[row].name if 0 <= row < len(self.profiles) else "Default"
        model_path = self.model_path_label.text()
        if model_path == "No model selected":
            model_path = ""
        return AppProfile(
            name=name,
            backend=self.backend_combo.currentText(),
            scale_factor=self.scale_spin.value(),
            target_width=self.width_spin.value(),
            target_height=self.height_spin.value(),
            quality=self.quality_combo.currentText(),
            model_path=model_path,
            tile_size=int(self.tile_combo.currentText()),
            tile_overlap=self.tile_overlap_spin.value(),
            half_precision=self.half_precision_check.isChecked(),
            capture_api=self.capture_api_combo.currentText(),
            hotkey=self.hotkey_label.text(),
            always_on_top=self.always_on_top_check.isChecked(),
        )

    def _save_current_profile(self) -> None:
        row = self.profile_list.currentRow()
        if row < 0:
            return
        profile = self._profile_from_fields()
        self.profiles[row] = profile
        path = self.store.save(profile)
        self.status_label.setText(f"Saved: {path.name}")

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select super-resolution model",
            str(Path.cwd()),
            "Model files (*.pt *.pth *.ckpt);;All files (*.*)",
        )
        if path:
            self.model_path_label.setText(path)

    def _toggle_running(self) -> None:
        self.running = not self.running
        if self.running:
            self.start_button.setText("Stop")
            self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
            self.status_label.setText("Preview runtime ready. Real-time capture is the next milestone.")
        else:
            self.start_button.setText("Start")
            self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.status_label.setText("Stopped")

    def _run_preview(self) -> None:
        profile = self._profile_from_fields()
        frame = self._make_preview_frame(320, 180)
        self._set_preview(self.before_label, frame)

        self.preview_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            config = SuperResolutionConfig(
                backend=profile.backend,
                model_path=profile.model_path or None,
                scale_factor=profile.scale_factor,
                half_precision=profile.half_precision,
                tile_size=profile.tile_size,
                tile_overlap=profile.tile_overlap,
                quality=profile.quality,
            )
            started = time.perf_counter()
            self.engine = SuperResolutionEngine(config)
            output, stats = self.engine.upscale_with_stats(frame)
            total_ms = (time.perf_counter() - started) * 1000.0
            self._set_preview(self.after_label, output)
            self.stats_label.setText(
                f"Latency: {stats.latency_ms:.1f} ms\n"
                f"Output: {stats.output_resolution[0]} x {stats.output_resolution[1]}\n"
                f"Memory: {stats.gpu_memory_mb:.1f} MB\n"
                f"Total: {total_ms:.1f} ms"
            )
            self.status_label.setText(f"Preview finished with {stats.backend} on {stats.device}")
        except Exception as exc:  # UI boundary: show recoverable errors.
            QMessageBox.critical(self, "Preview failed", str(exc))
            self.status_label.setText("Preview failed")
        finally:
            QApplication.restoreOverrideCursor()
            self.preview_button.setEnabled(True)

    def _set_preview(self, label: QLabel, frame: np.ndarray) -> None:
        image = np.ascontiguousarray(frame[:, :, :3])
        height, width, channels = image.shape
        qimage = QImage(image.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    @staticmethod
    def _make_preview_frame(width: int, height: int) -> np.ndarray:
        x = np.linspace(0, 1, width, dtype=np.float32)
        y = np.linspace(0, 1, height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)
        frame = np.stack(
            [
                40 + 160 * xx,
                50 + 120 * yy,
                80 + 100 * (1 - xx * yy),
            ],
            axis=-1,
        ).astype(np.uint8)
        frame[30:34, 28 : width - 28] = [235, 200, 80]
        frame[height - 42 : height - 38, 28 : width - 28] = [60, 220, 190]
        frame[44 : height - 54, 44:48] = [220, 80, 110]
        frame[44 : height - 54, width - 52 : width - 48] = [220, 80, 110]
        return frame

    @staticmethod
    def _device_label() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return f"CUDA: {torch.cuda.get_device_name(0)}"
        except Exception:
            pass
        return "CPU fallback"

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #101317;
                color: #E8EDF2;
                font-family: "Segoe UI", "Microsoft YaHei";
                font-size: 13px;
            }
            #SidePanel, #Header, QGroupBox {
                background: #171B21;
                border: 1px solid #2B323B;
                border-radius: 8px;
            }
            #Header {
                border-left: 4px solid #42C2A6;
            }
            #PanelTitle, #AppTitle {
                font-size: 19px;
                font-weight: 700;
            }
            #AppSubtitle, #StatusLabel, #PathLabel {
                color: #AAB6C3;
            }
            #Badge {
                background: #26332F;
                color: #78E0C3;
                border: 1px solid #365A50;
                border-radius: 5px;
                padding: 6px 10px;
            }
            QGroupBox {
                margin-top: 10px;
                padding: 12px;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #DCE6EF;
            }
            QListWidget, QComboBox, QSpinBox {
                background: #0E1116;
                border: 1px solid #303842;
                border-radius: 5px;
                min-height: 28px;
                padding: 3px 6px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background: #26453F;
                color: #FFFFFF;
            }
            QPushButton {
                background: #24313B;
                border: 1px solid #394755;
                border-radius: 6px;
                padding: 7px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #2D3F4B;
            }
            QPushButton:pressed {
                background: #1B2831;
            }
            #PreviewPane {
                background: #0B0D11;
                border: 1px solid #2E3742;
                border-radius: 6px;
                color: #788694;
                font-size: 15px;
            }
            #StatsLabel {
                color: #D7E0EA;
                background: #11161B;
                border: 1px solid #2A333D;
                border-radius: 6px;
                padding: 8px;
            }
            QSlider::groove:horizontal {
                background: #303842;
                height: 5px;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #F2B84B;
                width: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            """
        )
