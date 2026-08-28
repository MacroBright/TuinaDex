"""Unit tests for TuinaRobot LeRobot 0.4.4 BYOH integration and safety gates."""

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

    # 2. Connect -> SAFE_IDLE (starts poller, not armed)
    robot.connect()
    assert robot.is_connected is True
    assert robot.is_armed is False

    # 3. get_observation from cache
    obs = robot.get_observation()
    assert "observation.state" in obs
    assert "observation.images.overhead_cam" in obs
    assert obs["observation.state"].shape == (22,)

    # 4. send_action before arming -> returns safe hold
    test_action = np.full(22, 0.5, dtype=np.float32)
    ret_action = robot.send_action(test_action)
    assert np.allclose(ret_action, test_action)

    # 5. Cannot arm without gravity confirmation
    with pytest.raises(RuntimeError):
        robot.arm(gravity_confirmed=False)

    # 6. Explicit arm
    robot.arm(gravity_confirmed=True)
    assert robot.is_armed is True

    # 7. send_action when armed (measured dt and supervisor clamp)
    executed = robot.send_action(test_action)
    assert executed is not None
    if isinstance(executed, dict):
        assert executed["action"].shape == (22,)
    else:
        assert executed.shape == (22,)

    # 8. Disconnect (stops pollers)
    robot.disconnect()
    assert robot.is_connected is False
    assert robot.is_armed is False


def test_tuina_robot_real_mode_no_silent_fallback():
    # In real mode without hardware, connect MUST fail with HARDWARE_FAULT
    config = TuinaRobotConfig(
        id="test_tuina_real_no_hardware",
        can_interface="can_non_existent_999",
        hand_serial_port="/dev/ttyUSB_NON_EXISTENT",
        mock_mode=False,
    )
    robot = TuinaRobot(config)

    with pytest.raises(RuntimeError) as exc_info:
        robot.connect()

    assert "HARDWARE_FAULT" in str(exc_info.value)
    assert robot.is_connected is False
    assert robot.is_armed is False
