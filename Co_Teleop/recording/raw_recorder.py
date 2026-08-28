"""RawRecorder 全量工程遥测录制器 (JSONL + NPZ 离线复盘与诊断)."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Optional

from .teleop_frame import TeleopFrame


class RawRecorder:
    """全量工程与控制诊断数据记录器."""

    def __init__(self, root: Optional[str | Path] = None):
        self.root = Path(root) if root else Path("./datasets/raw_teleop")
        self.root.mkdir(parents=True, exist_ok=True)
        self._current_fpath: Optional[Path] = None
        self._current_file = None
        self._is_recording = False
        self._frame_count = 0

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start_episode(self, episode_id: Optional[str] = None) -> Path:
        """开始录制新 Episode 并打开 JSONL 文件句柄."""
        if episode_id is None:
            episode_id = f"ep_{time.strftime('%Y%m%d_%H%M%S')}"

        ep_dir = self.root / episode_id
        ep_dir.mkdir(parents=True, exist_ok=True)
        self._current_fpath = ep_dir / "raw_telemetry.jsonl"
        self._current_file = open(self._current_fpath, "w", encoding="utf-8")
        self._is_recording = True
        self._frame_count = 0
        return ep_dir

    def record_frame(self, frame: TeleopFrame) -> None:
        """写入一帧原始全量 TeleopFrame 遥测数据."""
        if not self._is_recording or self._current_file is None:
            return

        raw_dict = frame.to_raw_dict()
        line = json.dumps(raw_dict, ensure_ascii=False)
        self._current_file.write(line + "\n")
        self._current_file.flush()
        self._frame_count += 1

    def save_episode(self) -> int:
        """完成并关闭当前 Episode 文件."""
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None
        self._is_recording = False
        count = self._frame_count
        self._frame_count = 0
        return count

    def discard_episode(self) -> None:
        """放弃并删除当前未完成的 Episode."""
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None
        if self._current_fpath and self._current_fpath.parent.exists():
            import shutil
            shutil.rmtree(self._current_fpath.parent, ignore_errors=True)
        self._is_recording = False
        self._frame_count = 0
