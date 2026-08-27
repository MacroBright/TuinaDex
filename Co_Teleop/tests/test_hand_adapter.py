"""Co_Teleop.tests.test_hand_adapter — 灵巧手适配层单元测试."""
import numpy as np
import pytest

from Co_Teleop.adapters.hand_adapter import (
    DEFAULT_OPEN_POSE,
    JOINT_DIR,
    LeapHandAdapter,
    NoDriveHandAdapter,
)


def test_nodrive_hand_adapter_basic():
    adapter = NoDriveHandAdapter()
    assert not adapter.is_connected()
    assert adapter.state() == "UNPOWERED"

    adapter.connect()
    assert adapter.is_connected()
    assert adapter.state() == "POWERED_OPEN"

    # 设置角度
    test_angles = np.ones(16) * 0.5
    pos = adapter.set_angles(test_angles)
    expected_pos = DEFAULT_OPEN_POSE + JOINT_DIR * test_angles
    np.testing.assert_allclose(pos, expected_pos)
    np.testing.assert_allclose(adapter.get_current_angles(), test_angles)
    assert adapter.state() == "TELEOP"

    # 丢失保护 relax_step 测试
    p_relax0 = adapter.relax_step(now=10.0, hold_time=0.3, ramp_time=0.6)
    np.testing.assert_allclose(p_relax0, expected_pos)
    p_relax_done = adapter.relax_step(now=12.0, hold_time=0.3, ramp_time=0.6)
    np.testing.assert_allclose(p_relax_done, DEFAULT_OPEN_POSE)
    adapter.reset_loss_state()

    # 回全开位
    open_pos = adapter.set_open()
    np.testing.assert_allclose(open_pos, DEFAULT_OPEN_POSE)
    np.testing.assert_allclose(adapter.get_current_angles(), np.zeros(16))
    assert adapter.state() == "OPEN"

    adapter.disconnect()
    assert not adapter.is_connected()
    assert adapter.state() == "DISCONNECTED"


def test_leap_hand_adapter_offline_math():
    adapter = LeapHandAdapter(port="COM_DUMMY", limits_path="/non_existent_limits.json")
    assert not adapter.is_connected()
    assert adapter.state() == "UNPOWERED"

    # 设置角度测试
    test_angles = np.zeros(16)
    test_angles[0] = 0.8
    test_angles[1] = 0.4
    pos = adapter.set_angles(test_angles)
    expected_pos = adapter.open_pose + JOINT_DIR * test_angles
    np.testing.assert_allclose(pos, expected_pos)

    # 限位裁剪测试
    adapter.motor_limits = (np.zeros(16), np.ones(16) * 4.0)
    adapter.open_pose = np.ones(16) * 3.5
    test_angles = np.ones(16) * 2.0  # 3.5 + (-1)*2.0 = 1.5 (合规)
    pos = adapter.set_angles(test_angles)
    assert (pos >= 0.0).all() and (pos <= 4.0).all()


def test_leap_hand_adapter_relax_step():
    adapter = LeapHandAdapter(port="COM_DUMMY", limits_path="/non_existent_limits.json")
    adapter.open_pose = np.ones(16) * 3.0
    adapter.last_commanded_pose = np.ones(16) * 1.0

    # 阶段 1: 丢失初期的保持 (t < 0.3s)
    p0 = adapter.relax_step(now=100.0, hold_time=0.3, ramp_time=0.6)
    np.testing.assert_allclose(p0, np.ones(16) * 1.0)

    p1 = adapter.relax_step(now=100.2, hold_time=0.3, ramp_time=0.6)
    np.testing.assert_allclose(p1, np.ones(16) * 1.0)

    # 阶段 2: 线性插值过渡中点 (t = 100.3 + 0.3 = 100.6s -> 50% 进度 -> 1.0 + (3.0-1.0)*0.5 = 2.0)
    p2 = adapter.relax_step(now=100.6, hold_time=0.3, ramp_time=0.6)
    np.testing.assert_allclose(p2, np.ones(16) * 2.0)

    # 阶段 3: 恢复完毕 (t >= 100.9s -> 100% 进度 -> 回到 OPEN 3.0)
    p3 = adapter.relax_step(now=101.5, hold_time=0.3, ramp_time=0.6)
    np.testing.assert_allclose(p3, np.ones(16) * 3.0)

    # 重置状态
    adapter.reset_loss_state()
    assert adapter.loss_t0 is None
