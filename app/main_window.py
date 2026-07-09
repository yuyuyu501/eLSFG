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
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.config import AppProfile, ProfileStore
from core.super_resolution import SuperResolutionConfig, SuperResolutionEngine


MODEL_NONE_TEXT = "未选择模型"

PROFILE_LABELS = {
    "AI Quality": "AI 画质",
    "Default": "默认",
    "Game Fast": "游戏快速",
}


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
        scroll = QScrollArea()
        scroll.setObjectName("MainScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._build_control_panel())
        root_layout.addWidget(scroll, stretch=1)
        self.setCentralWidget(root)

    def _build_profile_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setFixedWidth(220)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("配置")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        self.profile_list = QListWidget()
        self.profile_list.currentRowChanged.connect(self._on_profile_selected)
        layout.addWidget(self.profile_list, stretch=1)

        self.save_button = QPushButton("保存")
        self.save_button.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton))
        self.save_button.clicked.connect(self._save_current_profile)
        layout.addWidget(self.save_button)

        self.status_label = QLabel("就绪")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        return panel

    def _build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(700)
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
        grid.setRowMinimumHeight(0, 160)
        grid.setRowMinimumHeight(1, 205)
        layout.addLayout(grid)

        layout.addWidget(self._build_preview_group(), stretch=1)
        return panel

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("Header")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 12, 14, 12)

        title = QLabel("eLSFG 控制台")
        title.setObjectName("AppTitle")
        subtitle = QLabel("超分辨率 MVP")
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
        group = QGroupBox("目标")
        group.setMinimumHeight(155)
        form = self._form_layout(group)
        form.setLabelAlignment(Qt.AlignLeft)
        self.target_label = QLabel("调试预览帧")
        self.capture_api_combo = QComboBox()
        self._add_combo_items(
            self.capture_api_combo,
            [
                ("调试预览", "debug-preview"),
                ("屏幕 GDI", "screen-gdi"),
                ("DXCam", "dxcam"),
                ("WGC 后续", "wgc-next"),
            ],
        )
        self.hotkey_label = QLabel("Alt+S")
        self.always_on_top_check = QCheckBox("输出窗口置顶")
        form.addRow("来源", self.target_label)
        form.addRow("采集方式", self.capture_api_combo)
        form.addRow("快捷键", self.hotkey_label)
        form.addRow("", self.always_on_top_check)
        return group

    def _build_scaling_group(self) -> QGroupBox:
        group = QGroupBox("缩放")
        group.setMinimumHeight(155)
        form = self._form_layout(group)
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
        form.addRow("倍率", self.scale_spin)
        form.addRow("目标宽度", self.width_spin)
        form.addRow("目标高度", self.height_spin)
        form.addRow("锐化", self.sharpness_slider)
        return group

    def _build_super_resolution_group(self) -> QGroupBox:
        group = QGroupBox("超分辨率")
        group.setMinimumHeight(205)
        form = self._form_layout(group)
        self.backend_combo = QComboBox()
        self._add_combo_items(
            self.backend_combo,
            [
                ("Bicubic", "bicubic"),
                ("Bilinear", "bilinear"),
                ("Nearest", "nearest"),
                ("AI 超分模型", "sr_transformer"),
                ("自动", "auto"),
            ],
        )
        self.quality_combo = QComboBox()
        self._add_combo_items(
            self.quality_combo,
            [("快速", "fast"), ("平衡", "balanced"), ("画质", "quality")],
        )
        self.tile_combo = QComboBox()
        self.tile_combo.addItems(["0", "256", "384", "512"])
        self.tile_overlap_spin = QSpinBox()
        self.tile_overlap_spin.setRange(0, 128)
        self.tile_overlap_spin.setValue(16)
        self.half_precision_check = QCheckBox("CUDA 使用 FP16")
        self.half_precision_check.setChecked(True)

        model_row = QWidget()
        model_layout = QHBoxLayout(model_row)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(6)
        self.model_path_label = QLineEdit()
        self.model_path_label.setObjectName("PathLabel")
        self.model_path_label.setReadOnly(True)
        self.model_path_label.setPlaceholderText(MODEL_NONE_TEXT)
        self.model_path_label.setMinimumWidth(0)
        self.model_path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        browse_button = QPushButton("浏览")
        browse_button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        browse_button.clicked.connect(self._browse_model)
        model_layout.addWidget(self.model_path_label, stretch=1)
        model_layout.addWidget(browse_button)

        form.addRow("后端", self.backend_combo)
        form.addRow("质量", self.quality_combo)
        form.addRow("模型", model_row)
        form.addRow("分块大小", self.tile_combo)
        form.addRow("分块重叠", self.tile_overlap_spin)
        form.addRow("", self.half_precision_check)
        return group

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("运行")
        group.setMinimumHeight(155)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setSpacing(8)
        self.start_button = QPushButton("开始")
        self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.start_button.setMinimumHeight(38)
        self.start_button.clicked.connect(self._toggle_running)
        layout.addWidget(self.start_button)

        self.preview_button = QPushButton("运行预览 / 基准测试")
        self.preview_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.preview_button.setMinimumHeight(34)
        self.preview_button.clicked.connect(self._run_preview)
        layout.addWidget(self.preview_button)

        self.stats_label = QLabel("延迟: -\n输出: -\n显存: -")
        self.stats_label.setObjectName("StatsLabel")
        layout.addWidget(self.stats_label)
        layout.addStretch(1)
        return group

    def _build_preview_group(self) -> QGroupBox:
        group = QGroupBox("预览")
        group.setMinimumHeight(255)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(14, 20, 14, 14)
        layout.setSpacing(12)
        self.before_label = self._preview_label("处理前")
        self.after_label = self._preview_label("处理后")
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
            self.profile_list.addItem(PROFILE_LABELS.get(profile.name, profile.name))

    def _on_profile_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.profiles):
            return
        self._apply_profile(self.profiles[row])

    def _apply_profile(self, profile: AppProfile) -> None:
        self._set_combo_data(self.backend_combo, profile.backend)
        self.scale_spin.setValue(profile.scale_factor)
        self.width_spin.setValue(profile.target_width)
        self.height_spin.setValue(profile.target_height)
        self._set_combo_data(self.quality_combo, profile.quality)
        self.model_path_label.setText(profile.model_path)
        self.tile_combo.setCurrentText(str(profile.tile_size))
        self.tile_overlap_spin.setValue(profile.tile_overlap)
        self.half_precision_check.setChecked(profile.half_precision)
        self._set_combo_data(self.capture_api_combo, profile.capture_api)
        self.hotkey_label.setText(profile.hotkey)
        self.always_on_top_check.setChecked(profile.always_on_top)
        self.status_label.setText(f"已加载配置: {PROFILE_LABELS.get(profile.name, profile.name)}")

    def _profile_from_fields(self) -> AppProfile:
        row = self.profile_list.currentRow()
        name = self.profiles[row].name if 0 <= row < len(self.profiles) else "Default"
        current = self.profiles[row] if 0 <= row < len(self.profiles) else AppProfile()
        model_path = self.model_path_label.text()
        if model_path == MODEL_NONE_TEXT:
            model_path = ""
        return AppProfile(
            name=name,
            backend=self._combo_data(self.backend_combo),
            scale_factor=self.scale_spin.value(),
            target_width=self.width_spin.value(),
            target_height=self.height_spin.value(),
            quality=self._combo_data(self.quality_combo),
            model_path=model_path,
            tile_size=int(self.tile_combo.currentText()),
            tile_overlap=self.tile_overlap_spin.value(),
            half_precision=self.half_precision_check.isChecked(),
            capture_api=self._combo_data(self.capture_api_combo),
            hotkey=self.hotkey_label.text(),
            always_on_top=self.always_on_top_check.isChecked(),
            model_variant=current.model_variant,
            model_dim=current.model_dim,
            model_depth=current.model_depth,
            model_heads=current.model_heads,
            model_window_size=current.model_window_size,
        )

    def _save_current_profile(self) -> None:
        row = self.profile_list.currentRow()
        if row < 0:
            return
        profile = self._profile_from_fields()
        self.profiles[row] = profile
        path = self.store.save(profile)
        self.status_label.setText(f"已保存: {path.name}")

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择超分辨率模型",
            str(Path.cwd()),
            "模型文件 (*.pt *.pth *.ckpt);;所有文件 (*.*)",
        )
        if path:
            self.model_path_label.setText(path)

    def _toggle_running(self) -> None:
        self.running = not self.running
        if self.running:
            self.start_button.setText("停止")
            self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
            self.status_label.setText("预览运行时已就绪，实时采集是下一阶段。")
        else:
            self.start_button.setText("开始")
            self.start_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.status_label.setText("已停止")

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
                model_variant=profile.model_variant,
                model_dim=profile.model_dim,
                model_depth=profile.model_depth,
                model_heads=profile.model_heads,
                model_window_size=profile.model_window_size,
            )
            started = time.perf_counter()
            self.engine = SuperResolutionEngine(config)
            output, stats = self.engine.upscale_with_stats(frame)
            total_ms = (time.perf_counter() - started) * 1000.0
            self._set_preview(self.after_label, output)
            self.stats_label.setText(
                f"延迟: {stats.latency_ms:.1f} ms\n"
                f"输出: {stats.output_resolution[0]} x {stats.output_resolution[1]}\n"
                f"显存: {stats.gpu_memory_mb:.1f} MB\n"
                f"总耗时: {total_ms:.1f} ms"
            )
            self.status_label.setText(f"预览完成: {stats.backend} / {stats.device}")
        except Exception as exc:  # UI boundary: show recoverable errors.
            QMessageBox.critical(self, "预览失败", str(exc))
            self.status_label.setText("预览失败")
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
                return f"CUDA：{torch.cuda.get_device_name(0)}"
        except Exception:
            pass
        return "CPU 回退"

    @staticmethod
    def _add_combo_items(combo: QComboBox, items: list[tuple[str, str]]) -> None:
        for label, value in items:
            combo.addItem(label, value)

    @staticmethod
    def _combo_data(combo: QComboBox) -> str:
        data = combo.currentData()
        return str(data) if data is not None else combo.currentText()

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setCurrentText(value)

    @staticmethod
    def _form_layout(parent: QWidget) -> QFormLayout:
        form = QFormLayout(parent)
        form.setContentsMargins(14, 20, 14, 14)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.DontWrapRows)
        return form

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #101317;
                color: #E8EDF2;
                font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI";
                font-size: 13px;
            }
            #MainScroll {
                background: transparent;
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
                padding: 0;
                font-weight: 700;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #DCE6EF;
            }
            QListWidget, QComboBox, QSpinBox, QLineEdit {
                background: #0E1116;
                border: 1px solid #303842;
                border-radius: 5px;
                min-height: 28px;
                padding: 3px 6px;
            }
            QLineEdit:read-only {
                color: #AAB6C3;
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
