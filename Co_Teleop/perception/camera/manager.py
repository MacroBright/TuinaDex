"""CameraManager 多相机集中管理器."""

from __future__ import annotations

from typing import Dict, Optional

from Co_Teleop.recording.teleop_frame import CameraFrame
from .base import BaseCamera


class CameraManager:
    """多相机生命周期与多路采集调度管理器."""

    def __init__(self):
        self._cameras: Dict[str, BaseCamera] = {}

    def register_camera(self, camera: BaseCamera) -> None:
        """注册相机实例."""
        self._cameras[camera.config.camera_id] = camera

    def connect_all(self) -> Dict[str, bool]:
        """连接所有已注册相机."""
        status = {}
        for cam_id, cam in self._cameras.items():
            status[cam_id] = cam.connect()
        return status

    def start_all_streams(self) -> None:
        """启动所有相机的后台连续取帧线程."""
        for cam in self._cameras.values():
            cam.start_stream()

    def stop_all_streams(self) -> None:
        """停止所有相机的取帧线程."""
        for cam in self._cameras.values():
            cam.stop_stream()

    def disconnect_all(self) -> None:
        """安全释放所有相机."""
        for cam in self._cameras.values():
            cam.disconnect()

    def get_latest_frame(self, camera_id: str = "overhead_cam") -> Optional[CameraFrame]:
        """获取指定相机的最新帧."""
        cam = self._cameras.get(camera_id)
        if cam is None:
            return None
        return cam.get_latest_frame()

    def get_all_latest_frames(self) -> Dict[str, CameraFrame]:
        """获取所有可用相机的最新帧快照字典."""
        frames = {}
        for cam_id, cam in self._cameras.items():
            f = cam.get_latest_frame()
            if f is not None:
                frames[cam_id] = f
        return frames

    def __getitem__(self, camera_id: str) -> BaseCamera:
        return self._cameras[camera_id]

    def __contains__(self, camera_id: str) -> bool:
        return camera_id in self._cameras
