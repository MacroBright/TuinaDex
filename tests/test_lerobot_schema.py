"""Schema compatibility test for LeRobot 0.4.4 Dataset integration."""

import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def test_lerobot_schema_compatibility():
    """Verify that LeRobot 0.4.4 Dataset properly records 22D State, 22D Action, Overhead Video and 37-Acupoint features."""
    temp_dir = Path(tempfile.mkdtemp())
    ds_root = temp_dir / "test_lerobot_tuina"

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
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.environment.acupoints": {
            "dtype": "float32",
            "shape": (37, 3),
        },
    }

    try:
        ds = LeRobotDataset.create(
            repo_id="test_tuina_dataset",
            fps=30,
            features=features,
            root=ds_root,
            use_videos=True,
        )

        num_frames = 10
        for i in range(num_frames):
            frame = {
                "observation.state": np.full(22, fill_value=0.1 * i, dtype=np.float32),
                "action": np.full(22, fill_value=0.1 * (i + 1), dtype=np.float32),
                "observation.images.overhead_cam": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "observation.environment.acupoints": np.random.rand(37, 3).astype(np.float32),
                "task": "按揉大椎穴",
            }
            ds.add_frame(frame)

        ds.save_episode()
        ds.finalize()

        assert ds.num_episodes == 1
        assert ds.num_frames == num_frames

        # Verify reading dataset back
        loaded_ds = LeRobotDataset(
            repo_id="test_tuina_dataset",
            root=ds_root,
        )
        assert len(loaded_ds) == num_frames
        sample = loaded_ds[0]
        assert "observation.state" in sample
        assert "action" in sample
        assert sample["observation.state"].shape == (22,)
        assert sample["action"].shape == (22,)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
