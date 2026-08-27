"""real_arm_teleop 管线测试 — 无相机, 用 fake provider 驱动 run_once (spec §6.2)."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lerobot_robot_massage.zdt.recording import EpisodeRecorder  # noqa: E402
from lerobot_robot_massage.zdt.types import CartesianCommand, JointState  # noqa: E402
from Co_Teleop.pipeline.single_arm_teleop import RealArmTeleop  # noqa: E402
from Co_Teleop.safety import VisionWatchdog  # noqa: E402


class _FakeAdapter:
    def __init__(self):
        self.calls = []
        self.e_stop_calls = 0
        self.ready_calls = 0
        self.home_calls = 0
        self.phase = "ARMED"

    def move_cartesian_velocity(self, cmd: CartesianCommand):
        self.calls.append(("move", cmd))

    def ready(self):
        self.ready_calls += 1

    def home(self):
        self.home_calls += 1

    def e_stop(self):
        self.e_stop_calls += 1

    def get_joint_state(self):
        return JointState(q=(0.0,) * 6)

    def state(self):
        return self.phase


def _out(tag):
    p = Path("/tmp") / f"teleop_{tag}"
    if p.exists():
        shutil.rmtree(p)
    return p


def _hand(**kw):
    d = {"hand_present": True, "confidence": 0.9, "depth_valid": True,
         "wrist_mm": (0.0, 0.0, 100.0), "velocity": (5.0, 0.0, 0.0)}
    d.update(kw)
    return d


def test_fresh_command_flows_to_adapter():
    adapter = _FakeAdapter()
    wd = VisionWatchdog()
    rec = EpisodeRecorder(_out("fresh"))
    rec.start_episode()
    teleop = RealArmTeleop(adapter, wd, rec, hand_provider=lambda: _hand(),
                           key_provider=lambda: None)
    out = teleop.run_once(cmd_ts=0.1, now=0.15)
    assert out["action"] == "OK"
    assert adapter.calls and adapter.calls[-1][0] == "move"
    assert list(adapter.calls[-1][1].linear_velocity) == [5.0, 0.0, 0.0]
    rec.finish_episode()


def test_stale_command_no_motion():
    adapter = _FakeAdapter()
    wd = VisionWatchdog()
    rec = EpisodeRecorder(_out("stale"))
    rec.start_episode()
    teleop = RealArmTeleop(adapter, wd, rec, hand_provider=lambda: _hand(),
                           key_provider=lambda: None)
    out = teleop.run_once(cmd_ts=0.0, now=0.9)   # 陈旧
    assert out["action"] == "STOP"
    assert adapter.calls == [] or adapter.calls[-1][0] != "move"
    rec.finish_episode()


def test_estop_calls_adapter_estop():
    adapter = _FakeAdapter()
    wd = VisionWatchdog()
    rec = EpisodeRecorder(_out("estop"))
    rec.start_episode()
    hand = _hand(hand_present=False, confidence=0.0, depth_valid=False, wrist_mm=None)
    teleop = RealArmTeleop(adapter, wd, rec, hand_provider=lambda: hand,
                           key_provider=lambda: None)
    teleop.run_once(cmd_ts=0.1, now=0.1)
    teleop.run_once(cmd_ts=0.5, now=0.5)
    out = teleop.run_once(cmd_ts=1.2, now=1.2)   # 连续丢失 > estop_s
    assert out["action"] == "ESTOP"
    assert adapter.e_stop_calls == 1
    rec.finish_episode()


def test_key_estop_immediate():
    adapter = _FakeAdapter()
    wd = VisionWatchdog()
    rec = EpisodeRecorder(_out("key_estop"))
    rec.start_episode()
    teleop = RealArmTeleop(adapter, wd, rec, hand_provider=lambda: _hand(),
                           key_provider=lambda: ord("y"))
    out = teleop.run_once(cmd_ts=0.1, now=0.15)
    assert out["action"] == "ESTOP"
    assert adapter.e_stop_calls == 1
    rec.finish_episode()


def test_key_ready_and_home():
    adapter = _FakeAdapter()
    wd = VisionWatchdog()
    rec = EpisodeRecorder(_out("key_poses"))
    rec.start_episode()
    # Test 'r' -> READY
    teleop = RealArmTeleop(adapter, wd, rec, hand_provider=lambda: _hand(),
                           key_provider=lambda: ord("r"))
    out = teleop.run_once(cmd_ts=0.1, now=0.15)
    assert out["action"] == "READY"
    assert adapter.ready_calls == 1

    # Test 'o' / 'h' -> HOME
    teleop = RealArmTeleop(adapter, wd, rec, hand_provider=lambda: _hand(),
                           key_provider=lambda: ord("h"))
    out = teleop.run_once(cmd_ts=0.1, now=0.15)
    assert out["action"] == "HOME"
    assert adapter.home_calls == 1
    rec.finish_episode()


def test_key_space_toggle_clutch():
    adapter = _FakeAdapter()
    wd = VisionWatchdog()
    rec = EpisodeRecorder(_out("key_space"))
    rec.start_episode()
    # Initial state: clutch active -> move works
    teleop = RealArmTeleop(adapter, wd, rec, hand_provider=lambda: _hand(),
                           key_provider=lambda: None)
    out1 = teleop.run_once(cmd_ts=0.1, now=0.15)
    assert out1["clutch"] is True
    assert adapter.calls and adapter.calls[-1][0] == "move"

    # Press SPACE -> Toggle to PAUSED
    teleop.key_provider = lambda: 32
    out2 = teleop.run_once(cmd_ts=0.2, now=0.25)
    assert out2["action"] == "PAUSED"
    assert out2["clutch"] is False

    # Next frame with no key: stays PAUSED, no new move calls
    teleop.key_provider = lambda: None
    calls_before = len(adapter.calls)
    out3 = teleop.run_once(cmd_ts=0.3, now=0.35)
    assert out3["action"] == "PAUSED"
    assert len(adapter.calls) == calls_before

    # Press SPACE again -> Toggle back to ACTIVE (with ramp-up)
    teleop.key_provider = lambda: 32
    out4 = teleop.run_once(cmd_ts=0.4, now=0.45)
    assert out4["clutch"] is True
    rec.finish_episode()


def test_deadband_and_ramp_up():
    adapter = _FakeAdapter()
    wd = VisionWatchdog()
    rec = EpisodeRecorder(_out("deadband"))
    rec.start_episode()
    # hand with small tremor velocity (1.0 mm/s < 3.0 mm/s deadband)
    teleop = RealArmTeleop(adapter, wd, rec,
                           hand_provider=lambda: _hand(velocity=(1.0, 0.0, 0.0)),
                           key_provider=lambda: None,
                           deadband_mm_s=3.0)
    out = teleop.run_once(cmd_ts=0.1, now=0.15)
    # Velocity clamped to zero by deadband
    assert list(adapter.calls[-1][1].linear_velocity) == [0.0, 0.0, 0.0]
    rec.finish_episode()


def test_recorder_gets_records():
    adapter = _FakeAdapter()
    wd = VisionWatchdog()
    rec = EpisodeRecorder(_out("recs"))
    rec.start_episode()
    teleop = RealArmTeleop(adapter, wd, rec, hand_provider=lambda: _hand(),
                           key_provider=lambda: None)
    teleop.run_once(cmd_ts=0.1, now=0.15)
    teleop.run_once(cmd_ts=0.2, now=0.25)
    stats = rec.finish_episode()
    assert stats["records"] == 2


def test_massage_modes_and_orientation_constants():
    from Co_Teleop.pipeline.single_arm_teleop import (
        MODE_FULL, MODE_KNEAD, MODE_MAP, MODE_NAMES, MODE_PITCH, MODE_ROLL,
    )
    assert MODE_KNEAD == 1
    assert MODE_ROLL == 2
    assert MODE_PITCH == 3
    assert MODE_FULL == 4
    assert MODE_MAP["knead"] == MODE_KNEAD
    assert MODE_MAP["roll"] == MODE_ROLL
    assert MODE_MAP["pitch"] == MODE_PITCH
    assert MODE_MAP["full"] == MODE_FULL
    assert "垂直点按揉捏" in MODE_NAMES[MODE_KNEAD]
    assert "滚法" in MODE_NAMES[MODE_ROLL]
    assert "俯仰调节" in MODE_NAMES[MODE_PITCH]
    assert "全 6-DOF" in MODE_NAMES[MODE_FULL]


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
