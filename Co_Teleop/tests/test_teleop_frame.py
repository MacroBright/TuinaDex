"""Unit tests for TeleopFrame data contract and serialization."""

import time
import numpy as np
import pytest

from Co_Teleop.recording.teleop_frame import (
    AcupointObservation,
    ArmObservation,
    CameraFrame,
    FrameValidity,
    HandObservation,
    TaskMetadata,
    TeleopFrame,
)


def test_teleop_frame_creation():
    arm = ArmObservation(
        q=np.zeros(6, dtype=np.float32),
        dq=np.zeros(6, dtype=np.float32),
        current=np.zeros(6, dtype=np.float32),
        flags=1,
        timestamp=100.0,
    )
    hand = HandObservation(
        q=np.zeros(16, dtype=np.float32),
        currents=np.zeros(16, dtype=np.float32),
        timestamp=100.02,
    )
    cam = CameraFrame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=100.01,
        frame_id=1,
        camera_id="overhead_cam",
    )
    acupoints = AcupointObservation(
        keypoints_2d=np.zeros((37, 3), dtype=np.float32),
        bbox=np.array([100, 100, 400, 400], dtype=np.float32),
        confidence=0.95,
        timestamp=100.03,
    )
    task = TaskMetadata(
        task_name="按揉大椎穴",
        technique="kneading",
        phase="contact",
        target_acupoint="dazhui",
        episode_id="ep_001",
        step_idx=0,
    )

    frame = TeleopFrame(
        frame_id=1,
        timestamp=100.05,
        arm=arm,
        hand=hand,
        overhead_cam=cam,
        observation_state_22d=np.zeros(22, dtype=np.float32),
        action_22d=np.zeros(22, dtype=np.float32),
        cartesian_cmd=np.zeros(6, dtype=np.float32),
        acupoints=acupoints,
        validity=FrameValidity.VALID,
        task=task,
    )

    assert frame.frame_id == 1
    assert frame.validity == FrameValidity.VALID
    assert frame.observation_state_22d.shape == (22,)
    assert frame.action_22d.shape == (22,)
    assert frame.cartesian_cmd.shape == (6,)

    # Test to_lerobot_dict
    lr_dict = frame.to_lerobot_dict()
    assert "observation.state" in lr_dict
    assert "action" in lr_dict
    assert "task" in lr_dict
    assert "observation.images.overhead_cam" in lr_dict
    assert "observation.environment.acupoints" in lr_dict
    assert lr_dict["task"] == "按揉大椎穴"
    assert lr_dict["observation.state"].shape == (22,)
    assert lr_dict["action"].shape == (22,)
    assert lr_dict["observation.environment.acupoints"].shape == (37, 3)

    # Test to_raw_dict
    raw_dict = frame.to_raw_dict()
    assert raw_dict["frame_id"] == 1
    assert raw_dict["validity"] == "VALID"
    assert len(raw_dict["arm"]["q"]) == 6
    assert len(raw_dict["hand"]["q"]) == 16
    assert raw_dict["task"]["target_acupoint"] == "dazhui"


def test_teleop_frame_shape_validation():
    arm = ArmObservation(q=np.zeros(6), dq=np.zeros(6), current=np.zeros(6))
    hand = HandObservation(q=np.zeros(16), currents=np.zeros(16))

    # Invalid state shape (e.g. 21 instead of 22)
    with pytest.raises(AssertionError):
        TeleopFrame(
            frame_id=1,
            timestamp=1.0,
            arm=arm,
            hand=hand,
            observation_state_22d=np.zeros(21, dtype=np.float32),
            action_22d=np.zeros(22, dtype=np.float32),
            cartesian_cmd=np.zeros(6, dtype=np.float32),
        )

    # Invalid action shape (e.g. 6 instead of 22)
    with pytest.raises(AssertionError):
        TeleopFrame(
            frame_id=1,
            timestamp=1.0,
            arm=arm,
            hand=hand,
            observation_state_22d=np.zeros(22, dtype=np.float32),
            action_22d=np.zeros(6, dtype=np.float32),
            cartesian_cmd=np.zeros(6, dtype=np.float32),
        )
