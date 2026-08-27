"""机械臂串口薄客户端（真机串口 / socket:// 仿真）。

仅实现视觉遥操所需命令，命令语义与固件一致：
  remote_event p0 p1 p2 p3 p4 p5 [p6]
    vx=-p0, vy=p1, vz=(p4-p5)/2, rx=-p3(J5), ry=p2(J6), p6→J4(仿真扩展)
  end_event vx vy vz wx wy wz   (仿真扩展, 末端 6DOF 线+角速度 → 全 DLS IK)
  get_ee_pose → EEPOSE:xyz+wxyz  (仿真扩展, 末端位姿反馈)
本模块与 Arm-robot_VLA 的 serial_protocol.py 解耦（避免跨仓库运行时依赖）。

历史路径: 此文件原位于 Leap_Hand/python/gesture_mapping/arm_client.py,
2026-08 因归属错误迁移到本仓. Leap_Hand 同名模块于同次提交删除.
"""
import time
from typing import Tuple

import serial


class ArmClient:
    def __init__(self, port: str, baudrate: int = 115200, ser=None):
        self._ser = ser or serial.serial_for_url(
            port, baudrate=baudrate, timeout=0.05, write_timeout=0.1)
        time.sleep(0.3)
        self._ser.reset_input_buffer()
        self.ee_available = True   # 真机固件无 get_ee, 首次超时后置 False 不再轮询

    # ── 命令 ──────────────────────────────────────────────

    def get_state(self) -> Tuple[list, list, list]:
        """读取关节状态, 返回 (angles_deg, vels, loads) 各 6 元列表."""
        self._ser.write(b"get_state\n")
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            line = self._ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith("STATE:"):
                values = [float(v) for v in line[6:].split(",")]
                n = len(values) // 3
                return values[:n], values[n:2 * n], values[2 * n:]
        return [], [], []

    def get_ee(self):
        """读取仿真末端世界坐标 (m). 真机固件无此命令 → 返回 None 并缓存."""
        if not self.ee_available:
            return None
        self._ser.write(b"get_ee\n")
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            line = self._ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith("EE:"):
                vals = [float(v) for v in line[3:].split(",")]
                return vals[:3] if len(vals) >= 3 else None
        self.ee_available = False    # 一次超时即判定不支持, 之后立即返回
        return None

    def get_wrist(self):
        """读取仿真腕心世界坐标 (m). J4/J5/J6 旋转不移动腕心, 位置环反馈用."""
        if not self.ee_available:
            return None
        self._ser.write(b"get_wrist\n")
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            line = self._ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith("WRIST:"):
                vals = [float(v) for v in line[6:].split(",")]
                return vals[:3] if len(vals) >= 3 else None
        self.ee_available = False
        return None

    def get_ee_pose(self):
        """读仿真末端位姿 (m + wxyz四元数). 返回 (pos3, quat4) 或 None."""
        if not self.ee_available:
            return None
        self._ser.write(b"get_ee_pose\n")
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            line = self._ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith("EEPOSE:"):
                vals = [float(v) for v in line[7:].split(",")]
                if len(vals) >= 7:
                    return vals[:3], vals[3:7]   # (pos m, quat wxyz)
        self.ee_available = False    # 一次超时即判定不支持, 之后立即返回
        return None

    def end_event(self, vx, vy, vz, wx, wy, wz) -> None:
        """末端 6DOF 速度 (仿真扩展): 线速度+角速度 ∈[-1,1]."""
        cmd = (f"end_event {vx:.3f} {vy:.3f} {vz:.3f} "
               f"{wx:.3f} {wy:.3f} {wz:.3f}\n")
        self._ser.write(cmd.encode())

    def remote_enable(self) -> None:
        self._ser.write(b"remote_enable\n")

    def remote_disable(self) -> None:
        self._ser.write(b"remote_disable\n")

    def e_stop(self) -> None:
        self._ser.write(b"e_stop\n")

    def remote_event(self, vx: float, vy: float, vz: float,
                     j5: float, j6: float = 0.0, j4: float = 0.0) -> None:
        """发送差分速度命令. 各输入∈[-1,1]. p6→J4 为仿真扩展通道."""
        p0, p1, p2, p3, p4, p5, p6 = -vx, vy, j6, -j5, vz, -vz, j4
        cmd = (f"remote_event {p0:.3f} {p1:.3f} {p2:.3f} "
               f"{p3:.3f} {p4:.3f} {p5:.3f} {p6:.3f}\n")
        self._ser.write(cmd.encode())

    def soft_reset(self) -> None:
        """软复位: 全部关节回预设初始角度."""
        self._ser.write(b"soft_reset\n")

    def set_joints(self, angles_deg) -> None:
        """设置全部关节目标角度(度). 用于手动复位点. 列表长度为 6."""
        cmd = "set_joints " + " ".join(f"{float(a):.3f}" for a in angles_deg[:6])
        self._ser.write((cmd + "\n").encode())

    def rel_rotate(self, joint_id: int, delta_deg: float) -> None:
        """单关节相对旋转 (joint_id: 1-based 1-6). 录制复位点时微调用."""
        self._ser.write(f"rel_rotate {joint_id} {float(delta_deg):.1f}\n".encode())

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass
