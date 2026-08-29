"""Ubuntu-local Open3D window for live stereo video, depth, and point cloud."""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path
from time import sleep
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


def _load_open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError(
            "Open3D is not installed. Activate tuinadex_hw and run: "
            "python -m pip install open3d==0.19.0"
        ) from exc
    return o3d


class _Viewer:
    def __init__(self, o3d: Any, worker: RealtimeWorker) -> None:
        self.o3d = o3d
        self.gui = o3d.visualization.gui
        self.rendering = o3d.visualization.rendering
        self.worker = worker
        self.app = self.gui.Application.instance
        self.window = self.app.create_window("TuinaDex 实时双目点云", 1600, 900)
        self.status = self.gui.Label("正在启动相机…")
        self.pause_button = self.gui.Button("暂停")
        self.reset_button = self.gui.Button("重置视角")
        self.exit_button = self.gui.Button("退出")
        self.left_image = self.gui.ImageWidget()
        self.depth_image = self.gui.ImageWidget()
        self.scene_widget = self.gui.SceneWidget()
        self.scene_widget.scene = self.rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background(np.array([0.02, 0.03, 0.04, 1.0]))
        self.material = self.rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = 3.0
        self.paused = False
        self.camera_initialized = False
        self.last_sequence = -1
        self.closing = threading.Event()
        self.poll_thread: threading.Thread | None = None

        for widget in (
            self.status,
            self.pause_button,
            self.reset_button,
            self.exit_button,
            self.left_image,
            self.depth_image,
            self.scene_widget,
        ):
            self.window.add_child(widget)
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.pause_button.set_on_clicked(self._toggle_pause)
        self.reset_button.set_on_clicked(self._reset_view)
        self.exit_button.set_on_clicked(self._exit)

    def start(self) -> None:
        self.worker.start()
        self.poll_thread = threading.Thread(
            target=self._poll,
            name="realtime-viewer-poll",
            daemon=True,
        )
        self.poll_thread.start()

    def _on_layout(self, _context: Any) -> None:
        rect = self.window.content_rect
        gap = 8
        status_height = 30
        toolbar_height = 38
        left_width = max(360, int(rect.width * 0.36))
        image_height = max(180, (rect.height - status_height - toolbar_height - 3 * gap) // 2)
        self.status.frame = self.gui.Rect(rect.x + gap, rect.y, rect.width - 2 * gap, status_height)
        toolbar_y = rect.y + status_height + gap
        button_width = 105
        self.pause_button.frame = self.gui.Rect(rect.x + gap, toolbar_y, button_width, toolbar_height)
        self.reset_button.frame = self.gui.Rect(
            rect.x + gap + button_width + gap, toolbar_y, button_width, toolbar_height
        )
        self.exit_button.frame = self.gui.Rect(
            rect.x + gap + 2 * (button_width + gap), toolbar_y, button_width, toolbar_height
        )
        content_y = toolbar_y + toolbar_height + gap
        self.left_image.frame = self.gui.Rect(rect.x + gap, content_y, left_width, image_height)
        self.depth_image.frame = self.gui.Rect(
            rect.x + gap,
            content_y + image_height + gap,
            left_width,
            image_height,
        )
        scene_x = rect.x + left_width + 2 * gap
        self.scene_widget.frame = self.gui.Rect(
            scene_x,
            content_y,
            max(200, rect.width - left_width - 3 * gap),
            max(200, rect.height - content_y + rect.y - gap),
        )

    def _poll(self) -> None:
        while not self.closing.is_set():
            state = self.worker.state()
            if state.sequence != self.last_sequence:
                self.last_sequence = state.sequence
                self.app.post_to_main_thread(
                    self.window,
                    lambda current=state: self._apply_state(current),
                )
            sleep(0.04)

    def _apply_state(self, state: Any) -> None:
        if self.closing.is_set():
            return
        if state.error:
            self.status.text = f"错误：{state.error}"
            return
        snapshot = state.snapshot
        if snapshot is None:
            self.status.text = "正在等待第一帧…"
            return
        self.status.text = (
            f"{state.fps:4.1f} FPS  |  有效像素 {snapshot.valid_points:,}  |  "
            f"点云 {len(snapshot.points_xyz):,} 点  |  "
            f"中位深度 {snapshot.median_depth_mm:.0f} mm"
        )
        if self.paused:
            return
        self._update_images(snapshot)
        self._update_point_cloud(snapshot)
        self.window.post_redraw()

    def _update_images(self, snapshot: RealtimeSnapshot) -> None:
        left_rgb = cv2.cvtColor(snapshot.left_bgr, cv2.COLOR_BGR2RGB)
        depth_rgb = cv2.cvtColor(snapshot.depth_bgr, cv2.COLOR_BGR2RGB)
        self.left_image.update_image(self.o3d.geometry.Image(np.ascontiguousarray(left_rgb)))
        self.depth_image.update_image(self.o3d.geometry.Image(np.ascontiguousarray(depth_rgb)))

    def _update_point_cloud(self, snapshot: RealtimeSnapshot) -> None:
        if len(snapshot.points_xyz) == 0:
            return
        cloud = self.o3d.geometry.PointCloud()
        cloud.points = self.o3d.utility.Vector3dVector(
            snapshot.points_xyz.astype(np.float64) / 1000.0
        )
        cloud.colors = self.o3d.utility.Vector3dVector(
            snapshot.colors_rgb.astype(np.float64) / 255.0
        )
        if self.scene_widget.scene.has_geometry("live_points"):
            self.scene_widget.scene.remove_geometry("live_points")
        self.scene_widget.scene.add_geometry("live_points", cloud, self.material)
        if not self.camera_initialized:
            bounds = cloud.get_axis_aligned_bounding_box()
            self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
            self.camera_initialized = True

    def _toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.text = "继续" if self.paused else "暂停"

    def _reset_view(self) -> None:
        self.camera_initialized = False

    def _exit(self) -> None:
        self._shutdown()
        self.app.quit()

    def _on_close(self) -> bool:
        self._shutdown()
        return True

    def _shutdown(self) -> None:
        if self.closing.is_set():
            return
        self.closing.set()
        try:
            self.worker.stop()
        except RuntimeError as exc:
            print(f"退出警告：{exc}")
        if self.poll_thread is not None and self.poll_thread is not threading.current_thread():
            self.poll_thread.join(0.5)


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
    o3d = _load_open3d()
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    viewer = _Viewer(o3d, worker)
    viewer.start()
    app.run()
    viewer._shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"错误：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
