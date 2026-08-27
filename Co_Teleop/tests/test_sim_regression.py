"""仿真回归 — 全栈闭环: SimulationArmAdapter → fake MuJoCo server → watchdog → recorder.

验证依赖链终点: types → kinematics → adapter → watchdog → recording → teleop 逻辑,
在无硬件/无 MuJoCo 二进制下可跑. fake server 用 zdt 运动学做物理积分, 与控制器同源.
"""
import math
import socket
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np

from Co_Teleop.adapters import SimArmAdapter as SimulationArmAdapter  # noqa: E402
from Co_Teleop.adapters.arm_client import ArmClient  # noqa: E402
from lerobot_robot_massage.zdt.kinematics import (  # noqa: E402
    RESET_POSE_DEG, fk_mdh, jacobian,
)
from lerobot_robot_massage.zdt.recording import EpisodeRecorder  # noqa: E402
from lerobot_robot_massage.zdt.types import CartesianCommand  # noqa: E402
from Co_Teleop.pipeline.single_arm_teleop import RealArmTeleop  # noqa: E402
from Co_Teleop.safety import VisionWatchdog  # noqa: E402


class FakeMuJoCoServer:
    """模拟 mujoco_sim.py 文本协议 + FK 积分 (end_event → q += J⁻¹·twist·dt)."""

    def __init__(self):
        self.q = list(RESET_POSE_DEG)                 # source 帧
        self.lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        conn, _ = self._sock.accept()
        buf = b""
        with conn:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    resp = self._handle(line.decode().strip())
                    if resp is not None:
                        conn.sendall(resp + b"\n")

    def _handle(self, line):
        with self.lock:
            if line.startswith("get_ee_pose"):
                from lerobot_robot_massage.zdt.types import rotmat_to_quat
                T = fk_mdh(self.q)
                p = T[:3, 3] / 1000.0                  # mm → m
                q = rotmat_to_quat(T[:3, :3])          # wxyz
                return (f"EEPOSE:{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},"
                        f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}").encode()
            if line.startswith("get_state"):
                zero = ",0" * 12
                return (f"STATE:{','.join(f'{a:.4f}' for a in self.q)}{zero}").encode()
            if line.startswith("end_event"):
                vals = [float(x) for x in line.split()[1:]]
                twist = np.array(vals)                  # mm/s + rad/s
                J = jacobian(self.q)
                dq = np.linalg.solve(J.T, twist * 0.02)   # dt=0.02
                self.q = [self.q[i] + np.degrees(dq[i]) for i in range(6)]
                return None
            return None                                 # remote_enable/disable/soft_reset


def _rotation_angle(R):
    """SO(3) → 转角 (rad) (测试辅助)."""
    return math.acos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))


def _run_sim_sequence(server, adapter, wd, rec, cmd_6, frames=30, dt=0.02):
    """驱动 N 帧固定 6DOF 命令, 返回 (start_T, end_T). 手始终在 → watchdog OK."""
    present = {"hand_present": True, "confidence": 0.9, "depth_valid": True,
               "wrist_mm": (0.0, 0.0, 100.0)}
    teleop = RealArmTeleop(adapter, wd, rec,
                           hand_provider=lambda: dict(present),
                           key_provider=lambda: None)
    teleop._build_command = lambda h, ts: CartesianCommand(
        tuple(cmd_6[:3]), tuple(cmd_6[3:]), timestamp=ts)
    with server.lock:
        T0 = fk_mdh(np.array(server.q))
    for i in range(frames):
        teleop.run_once(cmd_ts=0.0 + i * dt, now=0.0 + i * dt)
    adapter.get_joint_state()   # 同步刷新 TCP 管道, 确保所有 end_event 处理完毕
    with server.lock:
        T1 = fk_mdh(np.array(server.q))
    return T0, T1


def _make_arm_client(port):
    import serial
    url = f"socket://127.0.0.1:{port}"
    ser = serial.serial_for_url(url, timeout=0.1)
    return ArmClient(url, ser=ser)


def test_sim_regression_full_stack():
    """全栈闭环: adapter + watchdog + recorder, 30 帧纯 +X 位移 > 2mm."""
    server = FakeMuJoCoServer()
    arm = _make_arm_client(server.port)
    adapter = SimulationArmAdapter(arm)
    wd = VisionWatchdog()
    rec = EpisodeRecorder("/tmp/sim_reg_1")
    rec.start_episode()
    adapter.connect()
    T0, T1 = _run_sim_sequence(server, adapter, wd, rec, (5.0, 0.0, 0.0, 0, 0, 0))
    # 30 帧 × 5mm/s × 0.02s = 3mm (+x 净位移, FK 积分误差容忍)
    assert T1[0, 3] - T0[0, 3] > 2.0, f"仿真闭环位移过小: {T1[0,3]-T0[0,3]:.2f}mm"
    stats = rec.finish_episode()
    assert stats["records"] == 30
    adapter.disconnect()
    server._sock.close()


def test_sim_regression_six_dof():
    """P1-⑦: 真正 6DOF — 纯 X/Y/Z 验位置、纯 Rx/Ry/Rz 验姿态."""
    cases = [
        ("X",  (5.0, 0.0, 0.0, 0.0, 0.0, 0.0), "pos", 0),
        ("Y",  (0.0, 5.0, 0.0, 0.0, 0.0, 0.0), "pos", 1),
        ("Z",  (0.0, 0.0, 5.0, 0.0, 0.0, 0.0), "pos", 2),
        ("Rx", (0.0, 0.0, 0.0, 0.15, 0.0, 0.0), "rot", 0),
        ("Ry", (0.0, 0.0, 0.0, 0.0, 0.15, 0.0), "rot", 1),
        ("Rz", (0.0, 0.0, 0.0, 0.0, 0.0, 0.15), "rot", 2),
    ]
    for label, cmd6, kind, idx in cases:
        server = FakeMuJoCoServer()
        arm = _make_arm_client(server.port)
        adapter = SimulationArmAdapter(arm)
        wd = VisionWatchdog()
        rec = EpisodeRecorder(f"/tmp/sim_6dof_{label}")
        rec.start_episode()
        adapter.connect()
        T0, T1 = _run_sim_sequence(server, adapter, wd, rec, cmd6)
        dp = T1[:3, 3] - T0[:3, 3]
        dR = _rotation_angle(T0[:3, :3].T @ T1[:3, :3])
        if kind == "pos":
            assert abs(dp[idx]) > 2.0, f"{label} 位置未动: {dp}"
        else:
            assert dR > 0.05, f"{label} 姿态未动: {dR} rad"
        assert rec.finish_episode()["records"] == 30
        adapter.disconnect()
        server._sock.close()


def test_sim_regression_combined_6dof():
    """组合 [vx,vy,vz,wx,wy,wz] 同时移动位置 + 姿态."""
    server = FakeMuJoCoServer()
    arm = _make_arm_client(server.port)
    adapter = SimulationArmAdapter(arm)
    wd = VisionWatchdog()
    rec = EpisodeRecorder("/tmp/sim_6dof_comb")
    rec.start_episode()
    adapter.connect()
    T0, T1 = _run_sim_sequence(server, adapter, wd, rec,
                               (3.0, 2.0, 1.0, 0.08, 0.06, 0.04))
    dp = T1[:3, 3] - T0[:3, 3]
    dR = _rotation_angle(T0[:3, :3].T @ T1[:3, :3])
    assert float(np.linalg.norm(dp)) > 2.0, f"组合位置未动: {dp}"
    assert dR > 0.05, f"组合姿态未动: {dR} rad"
    adapter.disconnect()
    server._sock.close()


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
