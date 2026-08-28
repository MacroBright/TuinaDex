"""Unit tests for TuinaRobot LeRobot 0.4.4 BYOH integration."""

import numpy as np
import pytest

from packages.lerobot_robot_tuinadex.config_tuinadex import TuinaRobotConfig
from packages.lerobot_robot_tuinadex.tuinadex import TuinaRobot


def test_tuina_robot_lifecycle_and_safety_gate():
    config = TuinaRobotConfig(
        id="test_tuina_001",
        mock_mode=True,
    )
    robot = TuinaRobot(config)

    # 1. Before connection
    assert robot.is_connected is False
    assert robot.is_armed is False

    with pytest.raises(RuntimeError):
        robot.get_observation()

    with pytest.raises(RuntimeError):
        robot.send_action(np.zeros(22, dtype=np.float32))

    # 2. Connect -> SAFE_IDLE (not armed)
    robot.connect()
    assert robot.is_connected is True
    assert robot.is_armed is False

    obs = robot.get_observation()
    assert "observation.state" in obs
    assert "observation.images.overhead_cam" in obs
    assert obs["observation.state"].shape == (22,)

    # 3. send_action before arming -> returns safe hold
    test_action = np.full(22, 0.5, dtype=np.float32)
    ret_action = robot.send_action(test_action)
    assert np.allclose(ret_action, test_action)  # Returns without physical dispatch

    # 4. Cannot arm without gravity confirmation
    with pytest.raises(RuntimeError):
        robot.arm(gravity_confirmed=False)

    # 5. Explicit arm
    robot.arm(gravity_confirmed=True)
    assert robot.is_armed is True

    # 6. send_action when armed
    executed = robot.send_action(test_action)
    assert executed is not None
    if isinstance(executed, dict):
        assert executed["action"].shape == (22,)
    else:
        assert executed.shape == (22,)

    # 7. Disconnect
    robot.disconnect()
    assert robot.is_connected is False
    assert robot.is_armed is False
