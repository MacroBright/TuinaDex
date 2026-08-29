"""Ubuntu-local PyQtGraph window for live stereo video, depth, and point cloud."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .cameras import OpenCVCameraPair
from .config import AppConfig
from .realtime import RealtimeProcessor, RealtimeSnapshot, RealtimeWorker, load_calibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ubuntu realtime stereo point-cloud viewer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--min-depth-mm", type=float, default=200.0)
    parser.add_argument("--max-depth-mm", type=float, default=5000.0)
    parser.add_argument("--stride", type=int, default=4)
    return parser


def _require_local_display() -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError(
            "No Ubuntu desktop display was found. Run this command in an Ubuntu "
            "desktop terminal, not a plain SSH shell."
        )


def _load_qt() -> tuple[Any, Any, Any, Any]:
    # Keep Qt and PyOpenGL on the same XWayland/GLX backend. On this Ubuntu
    # Wayland session Qt otherwise selects EGL while PyOpenGL queries GLX.
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("QT_XCB_GL_INTEGRATION", "xcb_glx")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        import pyqtgraph.opengl as gl
    except ImportError as exc:
        raise RuntimeError(
            "PyQtGraph viewer dependencies are missing. Activate tuinadex_hw and run: "
            "python -m pip install PyQt5==5.15.11 pyqtgraph==0.13.7 PyOpenGL==3.1.10"
        ) from exc
    # opencv-python sets this to its private Qt plugin directory during import.
    # That directory is incompatible with the separately installed PyQt5 runtime.
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(
        QtCore.QLibraryInfo.PluginsPath
    )
    return QtCore, QtGui, QtWidgets, gl


def _build_window_class(
    QtCore: Any,
    QtGui: Any,
    QtWidgets: Any,
    gl: Any,
) -> type:
    class RealtimeWindow(QtWidgets.QMainWindow):
        def __init__(self, worker: RealtimeWorker) -> None:
            super().__init__()
            self.worker = worker
            self.paused = False
            self.last_sequence = -1
            self.camera_initialized = False
            self.closed = False

            self.setWindowTitle("TuinaDex 实时双目点云")
            self.resize(1600, 900)
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)

            toolbar = QtWidgets.QHBoxLayout()
            self.status = QtWidgets.QLabel("正在启动相机…")
            self.status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.pause_button = QtWidgets.QPushButton("暂停")
            self.reset_button = QtWidgets.QPushButton("重置视角")
            self.exit_button = QtWidgets.QPushButton("退出")
            toolbar.addWidget(self.status, 1)
            toolbar.addWidget(self.pause_button)
            toolbar.addWidget(self.reset_button)
            toolbar.addWidget(self.exit_button)
            root.addLayout(toolbar)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            images = QtWidgets.QWidget()
            images_layout = QtWidgets.QVBoxLayout(images)
            images_layout.setContentsMargins(0, 0, 0, 0)
            self.left_image = self._image_panel(images_layout, "左相机（已校正）")
            self.depth_image = self._image_panel(images_layout, "深度伪彩色")
            splitter.addWidget(images)

            self.scene = gl.GLViewWidget()
            self.scene.setBackgroundColor((5, 8, 12, 255))
            self.points = gl.GLScatterPlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=np.empty((0, 4), dtype=np.float32),
                size=2.0,
                pxMode=True,
            )
            self.scene.addItem(self.points)
            splitter.addWidget(self.scene)
            splitter.setSizes([560, 1040])
            root.addWidget(splitter, 1)

            self.pause_button.clicked.connect(self._toggle_pause)
            self.reset_button.clicked.connect(self._request_reset)
            self.exit_button.clicked.connect(self.close)
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(40)
            self.timer.timeout.connect(self._poll)

        def _image_panel(self, layout: Any, title: str) -> Any:
            label = QtWidgets.QLabel(title)
            layout.addWidget(label)
            image = QtWidgets.QLabel("等待图像…")
            image.setAlignment(QtCore.Qt.AlignCenter)
            image.setMinimumSize(320, 220)
            image.setStyleSheet("background: #05080c; color: #aeb8c4;")
            image.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Ignored,
            )
            layout.addWidget(image, 1)
            return image

        def start(self) -> None:
            self.worker.start()
            self.timer.start()

        def _poll(self) -> None:
            state = self.worker.state()
            if state.sequence == self.last_sequence:
                return
            self.last_sequence = state.sequence
            if state.error:
                self.status.setText(f"错误：{state.error}")
                return
            snapshot = state.snapshot
            if snapshot is None:
                self.status.setText("正在等待第一帧…")
                return
            depth_text = (
                f"{snapshot.median_depth_mm:.0f} mm"
                if np.isfinite(snapshot.median_depth_mm)
                else "无有效深度"
            )
            self.status.setText(
                f"{state.fps:4.1f} FPS  |  有效像素 {snapshot.valid_points:,}  |  "
                f"点云 {len(snapshot.points_xyz):,} 点  |  中位深度 {depth_text}"
            )
            if self.paused:
                return
            self._set_image(self.left_image, snapshot.left_bgr)
            self._set_image(self.depth_image, snapshot.depth_bgr)
            self._set_cloud(snapshot)

        def _set_image(self, label: Any, bgr: np.ndarray) -> None:
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            height, width, _ = rgb.shape
            image = QtGui.QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QtGui.QImage.Format_RGB888,
            ).copy()
            pixmap = QtGui.QPixmap.fromImage(image).scaled(
                label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            label.setPixmap(pixmap)

        def _set_cloud(self, snapshot: RealtimeSnapshot) -> None:
            xyz = snapshot.points_xyz.astype(np.float32, copy=True) / 1000.0
            if len(xyz):
                xyz[:, 1:] *= -1.0
                rgba = np.empty((len(xyz), 4), dtype=np.float32)
                rgba[:, :3] = snapshot.colors_rgb.astype(np.float32) / 255.0
                rgba[:, 3] = 1.0
            else:
                rgba = np.empty((0, 4), dtype=np.float32)
            self.points.setData(pos=xyz, color=rgba, size=2.0, pxMode=True)
            if len(xyz) and not self.camera_initialized:
                center = np.median(xyz, axis=0)
                span = np.percentile(xyz, 95, axis=0) - np.percentile(xyz, 5, axis=0)
                distance = max(0.4, float(np.max(span)) * 1.8)
                self.scene.opts["center"] = QtGui.QVector3D(*map(float, center))
                self.scene.setCameraPosition(distance=distance, elevation=5, azimuth=-90)
                self.camera_initialized = True

        def _toggle_pause(self) -> None:
            self.paused = not self.paused
            self.pause_button.setText("继续" if self.paused else "暂停")

        def _request_reset(self) -> None:
            self.camera_initialized = False

        def closeEvent(self, event: Any) -> None:
            self._shutdown()
            event.accept()

        def _shutdown(self) -> None:
            if self.closed:
                return
            self.closed = True
            self.timer.stop()
            try:
                self.worker.stop()
            except RuntimeError as exc:
                print(f"退出警告：{exc}")

    return RealtimeWindow


def run(args: argparse.Namespace) -> int:
    _require_local_display()
    config = AppConfig.from_json(args.config)
    calibration = load_calibration(args.calibration, config.logical_image_size)
    processor = RealtimeProcessor(
        calibration,
        min_depth_mm=args.min_depth_mm,
        max_depth_mm=args.max_depth_mm,
        stride=args.stride,
    )
    worker = RealtimeWorker(lambda: OpenCVCameraPair(config), processor)
    QtCore, QtGui, QtWidgets, gl = _load_qt()
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window_class = _build_window_class(QtCore, QtGui, QtWidgets, gl)
    window = window_class(worker)
    window.show()
    window.start()
    result = int(application.exec_())
    window._shutdown()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
