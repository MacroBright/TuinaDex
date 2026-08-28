"""LeRobotDatasetWriter 标准数据集录制器 (基于 LeRobot 0.4.4 原生 LeRobotDataset).

职责：
1. 声明 22-DOF 观测/动作特征空间、工作区工业相机 RGB 视频流与 (37,3) 穴位结构化特征；
2. 接收来自 ObservationAggregator 生成的 TeleopFrame 快照；
3. 执行有效性自动清洗 (PAUSED 帧不写入训练集)；
4. 支持 save_episode() 异步落盘与 clear_episode_buffer() 废弃异常 Episode。
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any, Dict, Optional
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    LeRobotDataset = None

from .teleop_frame import FrameValidity, TeleopFrame


class LeRobotDatasetWriter:
    """LeRobotDataset v3 标准录制写入器."""

    def __init__(
        self,
        repo_id: str = "tuina_massage_22dof",
        fps: int = 30,
        root: Optional[str | Path] = None,
        image_height: int = 480,
        image_width: int = 640,
        use_videos: bool = True,
    ):
        self.repo_id = repo_id
        self.fps = fps
        self.root = Path(root) if root else Path("./datasets/lerobot_tuina")
        self.image_height = image_height
        self.image_width = image_width
        self.use_videos = use_videos

        self._dataset: Optional[Any] = None
        self._is_recording = False
        self._current_episode_frames = 0
        self._lock = threading.Lock()

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def episode_frames_count(self) -> int:
        return self._current_episode_frames

    def initialize(self) -> bool:
        """初始化或加载数据集."""
        if not HAS_LEROBOT:
            return False

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (22,),
                "names": [
                    "arm_j1", "arm_j2", "arm_j3", "arm_j4", "arm_j5", "arm_j6",
                    "hand_idx_yaw", "hand_idx_mcp", "hand_idx_pip", "hand_idx_dip",
                    "hand_mid_yaw", "hand_mid_mcp", "hand_mid_pip", "hand_mid_dip",
                    "hand_rng_yaw", "hand_rng_mcp", "hand_rng_pip", "hand_rng_dip",
                    "hand_thb_rot", "hand_thb_cmc", "hand_thb_mcp", "hand_thb_ip",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (22,),
                "names": [
                    "cmd_arm_j1", "cmd_arm_j2", "cmd_arm_j3", "cmd_arm_j4", "cmd_arm_j5", "cmd_arm_j6",
                    "cmd_hand_idx_yaw", "cmd_hand_idx_mcp", "cmd_hand_idx_pip", "cmd_hand_idx_dip",
                    "cmd_hand_mid_yaw", "cmd_hand_mid_mcp", "cmd_hand_mid_pip", "cmd_hand_mid_dip",
                    "cmd_hand_rng_yaw", "cmd_hand_rng_mcp", "cmd_hand_rng_pip", "cmd_hand_rng_dip",
                    "cmd_hand_thb_rot", "cmd_hand_thb_cmc", "cmd_hand_thb_mcp", "cmd_hand_thb_ip",
                ],
            },
            "observation.images.overhead_cam": {
                "dtype": "video",
                "shape": (self.image_height, self.image_width, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.environment.acupoints": {
                "dtype": "float32",
                "shape": (37, 3),
            },
        }

        try:
            self._dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                features=features,
                root=self.root,
                use_videos=self.use_videos,
            )
            return True
        except Exception as e:
            # 若目录已存在则尝试直接加载
            try:
                self._dataset = LeRobotDataset(repo_id=self.repo_id, root=self.root)
                return True
            except Exception:
                return False

    def start_episode(self) -> None:
        """开始一个新的 Episode 录制."""
        with self._lock:
            if self._dataset is None:
                if not self.initialize():
                    raise RuntimeError("Failed to initialize LeRobotDataset.")
            self._is_recording = True
            self._current_episode_frames = 0

    def add_frame(self, frame: TeleopFrame) -> bool:
        """接收一帧 TeleopFrame 并按有效性规则清洗与写入.

        Returns:
            bool: 该帧是否被成功计入 Dataset (PAUSED 帧自动丢弃返回 False).
        """
        if not self._is_recording or self._dataset is None:
            return False

        # 1. 过滤离合暂停帧 (不污染示范数据集)
        if frame.validity == FrameValidity.PAUSED:
            return False

        # 2. 构造符合 LeRobotDataset 规范的输入字典
        lr_frame = frame.to_lerobot_dict()

        # 缺失图像保护 (补零)
        if "observation.images.overhead_cam" not in lr_frame:
            lr_frame["observation.images.overhead_cam"] = np.zeros(
                (self.image_height, self.image_width, 3), dtype=np.uint8
            )

        # 缺失穴位特征保护 (补零)
        if "observation.environment.acupoints" not in lr_frame:
            lr_frame["observation.environment.acupoints"] = np.zeros((37, 3), dtype=np.float32)

        with self._lock:
            self._dataset.add_frame(lr_frame)
            self._current_episode_frames += 1

        return True

    def save_episode(self) -> int:
        """结束并保存当前 Episode 至 Parquet 表与 MP4 视频.

        Returns:
            int: 已录制的 Episode 总数
        """
        with self._lock:
            if not self._is_recording or self._dataset is None:
                return 0

            self._dataset.save_episode()
            self._is_recording = False
            frames_saved = self._current_episode_frames
            self._current_episode_frames = 0
            return self._dataset.num_episodes

    def clear_episode_buffer(self) -> None:
        """丢弃并清空当前未保存的 Episode 缓存 (例如操作失误时按 X 放弃)."""
        with self._lock:
            if self._dataset is not None and hasattr(self._dataset, "clear_episode_buffer"):
                self._dataset.clear_episode_buffer()
            self._is_recording = False
            self._current_episode_frames = 0

    def finalize(self) -> None:
        """最终固化数据集元数据与统计指标."""
        with self._lock:
            if self._dataset is not None:
                self._dataset.finalize()
