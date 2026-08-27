"""arm_adapter 测试 — Simulation/Real 统一接口 (spec §6.1)."""
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from lerobot_robot_massage.zdt.config import F_READ_POS, ZdtConfig  # noqa: E402
from lerobot_robot_massage.zdt.controller import ZdtController  # noqa: E402
from lerobot_robot_massage.zdt.fakes import FakeTransport  # noqa: E402
from lerobot_robot_massage.zdt.safety import MotorState  # noqa: E402
from lerobot_robot_massage.zdt.testutil import FakeClock  # noqa: E402
from lerobot_robot_massage.zdt.types import CartesianCommand  # noqa: E402
from Co_Teleop.adapters import RealArmAdapter, SimArmAdapter as SimulationArmAdapter  # noqa: E402

ADDRS = [0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
READY_ANCHOR = [0.0, 60.0, 50.0, 0.0, 120.0, 0.0]


def _arm_robot(ctrl):
    ctrl.robot.on_connected()
    motors = {a: MotorState(can_id=a, online=True, joint_slot=i)
              for i, a in enumerate(ADDRS)}
    ctrl.robot.on_enumerated(motors)
    ctrl.robot.on_safe_idle()
    ctrl.robot.arm(gravity_confirmed=True)
    return ctrl


def _inject_anchor_pose(t, q_anchor):
    for addr, deg in zip(ADDRS, q_anchor):
        v = int(round(abs(deg) * 65536.0 / 360.0)) & 0xFFFFFFFF
        sign = 0x01 if deg < 0 else 0x00
        t.inject(addr, F_READ_POS,
                 bytes([sign, (v >> 24) & 0xFF, (v >> 16) & 0xFF,
                        (v >> 8) & 0xFF, v & 0xFF]) + b"\x6b")


def _real_adapter():
    t = FakeTransport()
    cfg = ZdtConfig(timeout_s=0.001, retries=0, reduction_ratios=[1.0] * 6,
                    calib=[(1.0, 0.0)] * 6)
    ctrl = ZdtController(config=cfg, transport=t)
    _arm_robot(ctrl)
    clock = FakeClock()
    adapter = RealArmAdapter(ctrl, clock=clock)
    return adapter, t, clock


def test_sim_adapter_maps_velocity_to_end_event():
    arm = mock.Mock()
    a = SimulationArmAdapter(arm)
    a.connect()
    arm.remote_enable.assert_called_once()
    a.move_cartesian_velocity(CartesianCommand((1.0, 2.0, 3.0), (0.1, 0.0, 0.0)))
    arm.end_event.assert_called_once()
    np.testing.assert_allclose(arm.end_event.call_args[0], [1.0, 2.0, 3.0, 0.1, 0.0, 0.0])


def test_sim_adapter_ee_pose_mm_and_rotmat():
    arm = mock.Mock()
    arm.get_ee_pose.return_value = ([0.1, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0])
    a = SimulationArmAdapter(arm)
    ep = a.get_ee_pose()
    np.testing.assert_allclose(ep.position, [100.0, 200.0, 300.0])   # m → mm
    np.testing.assert_allclose(ep.rotation, np.eye(3), atol=1e-9)


def test_sim_adapter_joint_state():
    arm = mock.Mock()
    arm.get_state.return_value = ([0.0, 60.0, 50.0, 0.0, 120.0, 0.0],
                                  [0.0] * 6, [10.0] * 6)
    a = SimulationArmAdapter(arm)
    js = a.get_joint_state()
    assert list(js.q) == [0.0, 60.0, 50.0, 0.0, 120.0, 0.0]
    assert len(js.current_ma) == 6


def test_real_adapter_move_cartesian_sends_fd():
    adapter, t, clock = _real_adapter()
    _inject_anchor_pose(t, READY_ANCHOR)
    clock.tick(0.05)
    adapter.move_cartesian_velocity(
        CartesianCommand((10.0, 0.0, 0.0), (0.0, 0.0, 0.0), timestamp=clock.t))
    assert any(f.data and f.data[0] == 0xFD for f in t.sent)


def test_real_adapter_connect_does_not_arm():
    # P0-①: connect() 只到 SAFE_IDLE, 不使能扭矩; arm() 由调用方显式调用
    t = FakeTransport()
    cfg = ZdtConfig(timeout_s=0.001, retries=0, reduction_ratios=[1.0] * 6,
                    calib=[(1.0, 0.0)] * 6)
    ctrl = ZdtController(config=cfg, transport=t)
    for addr in range(0x01, 0x07):
        t.inject(addr, 0x1F, bytes([0x00, 0x01, 0x01]) + b"\x6b")
    for addr in range(0x01, 0x07):
        t.inject(addr, F_READ_POS, b"\x00\x00\x00\x00\x00" + b"\x6b")
    adapter = RealArmAdapter(ctrl)
    adapter.connect()
    assert ctrl.robot.phase.name == "SAFE_IDLE"
    assert not any(f.data and f.data[0] == 0xF3 for f in t.sent)   # 未使能扭矩
    adapter.arm(gravity_confirmed=True)
    assert ctrl.robot.phase.name == "ARMED"
    assert any(f.data and f.data[0] == 0xF3 for f in t.sent)


def test_real_adapter_get_ee_pose_uses_fk():
    adapter, t, clock = _real_adapter()
    from lerobot_robot_massage.zdt.kinematics import fk_mdh, anchor_to_source
    _inject_anchor_pose(t, READY_ANCHOR)
    ep = adapter.get_ee_pose()
    T = fk_mdh(anchor_to_source(READY_ANCHOR))
    np.testing.assert_allclose(ep.position, T[:3, 3], atol=0.01)
    np.testing.assert_allclose(ep.rotation, T[:3, :3], atol=1e-3)


def test_real_adapter_does_not_touch_driver_directly():
    # RealArmAdapter 只经 cart/ctrl 公共 API — 源码不应引用 _driver
    src = Path(Path(__file__).resolve().parents[1] / "adapters" / "arm_adapter.py").read_text()
    assert "_driver" not in src


def test_real_adapter_state_and_estop():
    adapter, t, clock = _real_adapter()
    assert adapter.state() == "ARMED"
    adapter.e_stop()
    assert t.sent[-1].arbitration_id == 0x0000


def test_no_drive_adapter_lifecycle():
    from Co_Teleop.adapters import NoDriveArmAdapter
    a = NoDriveArmAdapter(ready_pose=[0.0, 60.0, 50.0, 0.0, 120.0, 0.0])
    assert a.state() == "NO_DRIVE"
    a.connect()
    assert a.state() == "SAFE_IDLE"
    a.arm(gravity_confirmed=True)
    assert a.state() == "ARMED"
    a.enter_teleop()
    assert a.state() == "TELEOP"
    a.ready()
    assert list(a.get_joint_state().q) == [0.0, 60.0, 50.0, 0.0, 120.0, 0.0]
    a.home()
    assert list(a.get_joint_state().q) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    a.e_stop()
    assert a.state() == "STOPPED"
    a.disconnect()
    assert a.state() == "DISCONNECTED"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc}")
    print("ALL PASS" if failed == 0 else f"{failed} FAILED")
