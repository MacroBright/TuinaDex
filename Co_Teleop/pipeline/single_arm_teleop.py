"""scripts/teleop/real_arm_teleop.py — 真机 6DOF 视觉遥操入口 (spec TASK-23).

管线: RealSense + HandTracker + WristTracker → CartesianCommand → VisionWatchdog
(分级) → RealArmAdapter → CartesianController → ZdtController → CAN.
控制层是陈旧命令最终权威 (step 的 cmd_ts 单调期限); 本入口只做视觉分级 + 组装.
按键: H=clutch, R=reset/ready, Y=e_stop, Q/ESC=安全退出.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# ─── sys.path 注入 ───
_CURR_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = _CURR_DIR.parents[1]
_ARM_ROOT = _WORKSPACE_ROOT / "Arm-robot_VLA"
_LEAP_ROOT = _WORKSPACE_ROOT / "Leap_Hand" / "python"
_CO_ROOT = _WORKSPACE_ROOT / "Co_Teleop"
for _p in (_CURR_DIR, _WORKSPACE_ROOT, _ARM_ROOT, _LEAP_ROOT, _CO_ROOT):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from Co_Teleop.adapters import RealArmAdapter  # noqa: E402
from gesture_mapping.filter import OneEuroFilter  # noqa: E402
from lerobot_robot_massage.zdt.recording import EpisodeRecorder  # noqa: E402
from lerobot_robot_massage.zdt.types import CartesianCommand  # noqa: E402
from Co_Teleop.safety import VisionWatchdog, WatchdogAction  # noqa: E402


class RealArmTeleop:
    """一帧遥操逻辑 (可无相机单测): hand_provider → cmd → watchdog → clutch/ramp → adapter."""

    def __init__(self, adapter, watchdog, recorder, hand_provider, key_provider,
                 stale_cmd_max_s: float = 0.25,
                 ramp_up_s: float = 0.3,
                 deadband_mm_s: float = 3.0,
                 deadband_rad_s: float = 0.05,
                 default_clutch_active: bool = True):
        self.adapter = adapter
        self.watchdog = watchdog
        self.recorder = recorder
        self.hand_provider = hand_provider      # () -> dict | None
        self.key_provider = key_provider        # () -> key or None
        self.stale_cmd_max_s = stale_cmd_max_s
        self.ramp_up_s = ramp_up_s
        self.deadband_mm_s = deadband_mm_s
        self.deadband_rad_s = deadband_rad_s
        self.clutch_active = default_clutch_active
        self._clutch_on_time = -1e9             # 初始默认已完成热身, toggle 后从当前时间起算
        self._cmd = CartesianCommand((0.0, 0.0, 0.0))

    def toggle_clutch(self, now: float) -> bool:
        """切换离合器状态 (True=接合运动, False=暂停离合)."""
        # 若处于急停状态，按空格键尝试重新使能 (Re-arm) 并恢复
        if self.adapter.state() == "STOPPED":
            try:
                self.adapter.arm(gravity_confirmed=True)
                self.adapter.enter_teleop()
            except Exception:  # noqa: BLE001
                pass
            self.watchdog.reset()
            self.clutch_active = True
            self._clutch_on_time = now
            return True

        self.clutch_active = not self.clutch_active
        if self.clutch_active:
            self._clutch_on_time = now
            self.watchdog.reset()
        return self.clutch_active

    def run_once(self, cmd_ts: float, now: float) -> dict:
        """跑一帧: 返回 {action, cmd, phase, clutch}. 调用方提供时间 (单调)."""
        # 每帧最开始采集相机画面与手势，确保任何状态下画面都不中断/黑屏
        hand = self.hand_provider()

        key = self.key_provider()
        if key in (ord("y"), ord("Y")):
            self.adapter.e_stop()
            self.clutch_active = False
            return {"action": "ESTOP", "cmd": self._cmd,
                    "phase": self.adapter.state(),
                    "clutch": False}
        if key in (ord("q"), ord("Q"), 27):
            return {"action": "QUIT", "cmd": self._cmd,
                    "phase": self.adapter.state(),
                    "clutch": self.clutch_active}
        if key in (ord("r"), ord("R")):
            try:
                if hasattr(self.adapter, "re_arm") and self.adapter.state() == "STOPPED":
                    self.adapter.re_arm(gravity_confirmed=True)
                self.adapter.ready()
            except Exception as exc:  # noqa: BLE001
                print(f"[READY] {exc}")
            self.watchdog.reset()
            self.clutch_active = False
            return {"action": "READY", "cmd": CartesianCommand((0.0, 0.0, 0.0)),
                    "phase": self.adapter.state(),
                    "clutch": False}
        if key in (ord("o"), ord("O"), ord("0"), ord("h"), ord("H")):
            try:
                if hasattr(self.adapter, "re_arm") and self.adapter.state() == "STOPPED":
                    self.adapter.re_arm(gravity_confirmed=True)
                self.adapter.home()
            except Exception as exc:  # noqa: BLE001
                print(f"[HOME] {exc}")
            self.watchdog.reset()
            self.clutch_active = False
            return {"action": "HOME", "cmd": CartesianCommand((0.0, 0.0, 0.0)),
                    "phase": self.adapter.state(),
                    "clutch": False}
        if key == 32:  # SPACE bar: Toggle clutch / Re-arm from E-stop
            if self.adapter.state() == "STOPPED":
                if hasattr(self.adapter, "re_arm"):
                    self.adapter.re_arm(gravity_confirmed=True)
                self.watchdog.reset()
                print("[恢复] 机械臂已解除急停锁定并重新使能 (Re-armed)")
            self.toggle_clutch(now)

        if not self.clutch_active:
            scaled = CartesianCommand((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), timestamp=cmd_ts)
            self._record(hand, scaled, "PAUSED")
            return {"action": "PAUSED", "cmd": scaled,
                    "phase": self.adapter.state(),
                    "clutch": False}

        if (now - cmd_ts) > self.stale_cmd_max_s:
            scaled = CartesianCommand((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), timestamp=cmd_ts)
            self._record(hand, scaled, "STOP")
            return {"action": "STOP", "cmd": scaled,
                    "phase": self.adapter.state(),
                    "clutch": True}

        cmd = self._build_command(hand, cmd_ts)
        action, wd_scale = self.watchdog.update(
            hand_present=bool(hand and hand.get("hand_present")),
            hand_confidence=float(hand.get("confidence", 0.0)) if hand else 0.0,
            depth_valid=bool(hand and hand.get("depth_valid")),
            wrist_mm=hand.get("wrist_mm") if hand else None,
            now=now)

        # 开启缓起动 (Ramp-up) 线性增益
        dt_on = max(0.0, now - self._clutch_on_time)
        ramp_scale = min(1.0, dt_on / self.ramp_up_s) if self.ramp_up_s > 0 else 1.0
        total_scale = wd_scale * ramp_scale

        scaled = CartesianCommand(
            tuple(float(v) * total_scale for v in cmd.linear_velocity),
            tuple(float(w) * total_scale for w in cmd.angular_velocity),
            timestamp=cmd.timestamp)
        if action == WatchdogAction.ESTOP:
            self.adapter.e_stop()
            self.clutch_active = False
        elif action != WatchdogAction.STOP and self.adapter.state() != "STOPPED":
            self.adapter.move_cartesian_velocity(scaled)

        self._record(hand, scaled, action.name)
        return {"action": action.name, "cmd": scaled,
                "phase": self.adapter.state(),
                "clutch": self.clutch_active}

    def _build_command(self, hand, cmd_ts: float) -> CartesianCommand:
        """由手部信息合成 CartesianCommand (速度增量直接积分模式 + 死区过滤)."""
        if hand is None:
            return CartesianCommand((0.0, 0.0, 0.0), timestamp=cmd_ts)
        v = list(hand.get("velocity") or (0.0, 0.0, 0.0))
        w = list(hand.get("angular_velocity") or (0.0, 0.0, 0.0))
        v_norm = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        if v_norm < self.deadband_mm_s:
            v = [0.0, 0.0, 0.0]
        w_norm = (w[0]**2 + w[1]**2 + w[2]**2)**0.5
        if w_norm < self.deadband_rad_s:
            w = [0.0, 0.0, 0.0]
        return CartesianCommand(tuple(float(x) for x in v),
                                tuple(float(x) for x in w),
                                timestamp=cmd_ts)

    def _record(self, hand, cmd: CartesianCommand, action: str) -> None:
        obs = {
            "q": [0.0] * 6, "dq": [0.0] * 6, "current": [],
            "ee_pose": {"position": [0.0, 0.0, 0.0],
                        "quaternion": [1.0, 0.0, 0.0, 0.0]},
            "hand_pose": ({"position": list(hand.get("wrist_mm") or (0, 0, 0)),
                           "orientation": [0.0, 0.0, 0.0],
                           "confidence": hand.get("confidence", 0.0)}
                          if hand else {"position": [], "orientation": [],
                                        "confidence": 0.0}),
        }
        act = {"cartesian_command": {"linear_velocity": list(cmd.linear_velocity),
                                     "angular_velocity": list(cmd.angular_velocity),
                                     "timestamp": cmd.timestamp},
               "commanded_joint_target": [0.0] * 6}
        saf = {"phase": self.adapter.state(), "action": action}
        self.recorder.add_record(obs, act, saf)


MODE_KNEAD = 1
MODE_ROLL = 2
MODE_PITCH = 3
MODE_FULL = 4

MODE_NAMES = {
    MODE_KNEAD: "1. 垂直点按揉捏 (姿态全锁定)",
    MODE_ROLL:  "2. 滚法推法 (单轴Roll / 锁Pitch)",
    MODE_PITCH: "3. 俯仰调节 (单轴Pitch / 锁Roll)",
    MODE_FULL:  "4. 全 6-DOF (全姿态跟随)",
}
MODE_MAP = {"knead": MODE_KNEAD, "roll": MODE_ROLL, "pitch": MODE_PITCH, "full": MODE_FULL}

# 灵敏度档位系统 (Sensitivity Speed Gears):
# 从集中配置文件 teleop_config.py / teleop_config.yaml 动态构建
from Co_Teleop.config import TeleopConfig, build_gear_configs  # noqa: E402

DEFAULT_TELEOP_CONFIG = TeleopConfig.default()
GEAR_FINE = 1
GEAR_STANDARD = 2
GEAR_FAST = 3


def build_gear_configs(cfg: TeleopConfig) -> dict:
    """由 TeleopConfig 数据类动态构建 HUD 与速度计算字典."""
    return {
        GEAR_FINE: {
            "name": cfg.gear.gear_1_low.name,
            "badge": cfg.gear.gear_1_low.badge,
            "color": cfg.gear.gear_1_low.color,
            "lin_scale": cfg.gear.gear_1_low.lin_scale,
            "gain_xyz": cfg.gear.gear_1_low.gain_xyz,
            "max_omega": cfg.gear.gear_1_low.max_omega,
        },
        GEAR_STANDARD: {
            "name": cfg.gear.gear_2_mid.name,
            "badge": cfg.gear.gear_2_mid.badge,
            "color": cfg.gear.gear_2_mid.color,
            "lin_scale": cfg.gear.gear_2_mid.lin_scale,
            "gain_xyz": cfg.gear.gear_2_mid.gain_xyz,
            "max_omega": cfg.gear.gear_2_mid.max_omega,
        },
        GEAR_FAST: {
            "name": cfg.gear.gear_3_high.name,
            "badge": cfg.gear.gear_3_high.badge,
            "color": cfg.gear.gear_3_high.color,
            "lin_scale": cfg.gear.gear_3_high.lin_scale,
            "gain_xyz": cfg.gear.gear_3_high.gain_xyz,
            "max_omega": cfg.gear.gear_3_high.max_omega,
        },
    }


GEAR_CONFIGS = build_gear_configs(DEFAULT_TELEOP_CONFIG)


def _draw_joint_status_table(bgr, joint_state, no_drive: bool = False) -> None:
    """在 OpenCV 画面右下角绘制 6 关节 all status 实时监控表 (pos + current + flag)."""
    if joint_state is None:
        return
    import cv2
    h, w = bgr.shape[:2]
    box_w, box_h = 295, 148
    x0 = w - box_w - 12
    y0 = h - box_h - 12

    # 半透明深色底框
    overlay = bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (18, 18, 24), -1)
    cv2.addWeighted(overlay, 0.82, bgr, 0.18, 0, bgr)
    cv2.rectangle(bgr, (x0, y0), (x0 + box_w, y0 + box_h), (65, 65, 80), 1)

    # 标题与表头
    title = "[6-AXIS JOINTS TELEMETRY]" if not no_drive else "[6-AXIS JOINTS (SIM)]"
    cv2.putText(bgr, title, (x0 + 10, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 230, 255), 1)
    header = "Jnt  Addr   Pos(deg)   Current   Flag"
    cv2.putText(bgr, header, (x0 + 10, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 170), 1)
    cv2.line(bgr, (x0 + 8, y0 + 37), (x0 + box_w - 8, y0 + 37), (65, 65, 80), 1)

    addrs = [0x02, 0x03, 0x04, 0x05, 0x06, 0x07]
    q = list(joint_state.q) if joint_state and joint_state.q else [0.0] * 6
    cur = list(joint_state.current_ma) if joint_state and joint_state.current_ma else [0.0] * 6
    flags = list(joint_state.flags) if joint_state and joint_state.flags else [0] * 6

    for i in range(6):
        row_y = y0 + 53 + i * 15
        deg_str = f"{q[i]:+6.1f}°" if i < len(q) else " N/A "
        cur_str = f"{int(cur[i]):4d}mA" if (i < len(cur) and not no_drive) else " 0mA"
        flg_val = flags[i] if i < len(flags) else 0
        flg_str = f"0x{flg_val:02X} OK" if flg_val == 0 else f"0x{flg_val:02X} WRN"
        flg_color = (0, 255, 0) if flg_val == 0 else (0, 165, 255)

        line_txt = f" J{i+1}  0x{addrs[i]:02X}  {deg_str:8s}  {cur_str:7s}"
        cv2.putText(bgr, line_txt, (x0 + 10, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
        cv2.putText(bgr, flg_str, (x0 + 218, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, flg_color, 1)


def _draw_overlay(bgr, out: dict, hand_info: dict | None, clutch_active: bool,
                  no_drive: bool = False, lin_scale: float = 1.0, ang_scale: float = 1.0,
                  joint_state=None, mode: int = MODE_KNEAD, gear: int = GEAR_STANDARD) -> None:
    """在 OpenCV 画面上绘制丰富图元 (状态横幅 + 离合徽标 + 锚定球 + 速度矢量 + 数值反馈 + 关节全状态表 + 灵敏度档位)."""
    import cv2
    h, w = bgr.shape[:2]
    tag = "[NO-DRIVE] " if no_drive else ""
    g_info = GEAR_CONFIGS.get(gear, GEAR_CONFIGS[GEAR_STANDARD])
    gear_name = g_info["name"]
    phase = out.get("phase", "N/A")
    action = out.get("action", "N/A")

    # 顶部状态横幅 (显示推拿模式与灵敏度档位)
    mode_str = MODE_NAMES.get(mode, "")
    if action == "ESTOP" or phase == "STOPPED":
        cv2.rectangle(bgr, (0, 0), (w, 42), (0, 0, 200), -1)
        txt = f" {tag}[EMERGENCY STOP] Robot Locked! Press SPACE to Re-arm | Q: Quit"
        cv2.putText(bgr, txt, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif clutch_active:
        if mode == MODE_KNEAD:
            bar_color = (0, 130, 40)
        elif mode == MODE_ROLL:
            bar_color = (140, 90, 0)
        elif mode == MODE_PITCH:
            bar_color = (160, 50, 0)
        else:
            bar_color = (100, 0, 130)
        cv2.rectangle(bgr, (0, 0), (w, 42), bar_color, -1)
        txt = f" {tag}[遥操活跃] {mode_str} | 档位:[{gear_name}] | [S/TAB]换档 | [M]切模式 | [SPACE]暂停"
        cv2.putText(bgr, txt, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)
    else:
        cv2.rectangle(bgr, (0, 0), (w, 42), (0, 140, 220), -1)
        txt = f" {tag}[CLUTCH PAUSED] {mode_str} | 档位:[{gear_name}] | [S/TAB]换档 | [SPACE]激活"
        cv2.putText(bgr, txt, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 2)

    # 动作与状态机信息
    status_color = (0, 255, 0) if action == "OK" else ((0, 0, 255) if action == "ESTOP" else (0, 200, 255))
    act_str = f"Action: {action} | Phase: {phase} | Mode: {mode_str}"
    cv2.putText(bgr, act_str, (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2)

    # 速度指令反馈
    cmd = out.get("cmd")
    vel = cmd.linear_velocity if cmd else (0.0, 0.0, 0.0)
    ang = cmd.angular_velocity if cmd else (0.0, 0.0, 0.0)

    if clutch_active and action != "ESTOP":
        x_dir = "左" if vel[0] > 4 else ("右" if vel[0] < -4 else "-")
        y_dir = "后" if vel[1] > 4 else ("前" if vel[1] < -4 else "-")
        z_dir = "上" if vel[2] > 4 else ("下" if vel[2] < -4 else "-")
        spd_txt = f"v_lin: [X(左右):{vel[0]:+4.0f}({x_dir}), Y(前后):{vel[1]:+4.0f}({y_dir}), Z(上下):{vel[2]:+4.0f}({z_dir})] mm/s"
        cv2.putText(bgr, spd_txt, (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2)

        pitch_dir = "低头" if ang[0] < -0.05 else ("抬头" if ang[0] > 0.05 else "")
        roll_dir = "左滚" if ang[1] > 0.05 else ("右滚" if ang[1] < -0.05 else "")
        is_rot = np.linalg.norm(ang) > 0.02
        ang_color = (0, 255, 255) if is_rot else (200, 200, 200)
        tags = [t for t in [pitch_dir, roll_dir] if t]
        if tags:
            rot_tag = f" [{' '.join(tags)}]"
        elif mode == MODE_KNEAD:
            rot_tag = " [姿态全锁定]"
        elif mode == MODE_ROLL:
            rot_tag = " [锁Pitch/仅Roll]"
        elif mode == MODE_PITCH:
            rot_tag = " [锁Roll/仅Pitch]"
        else:
            rot_tag = ""
        ang_txt = f"w_ang: [Pitch:{ang[0]:+4.2f}, Roll:{ang[1]:+4.2f}] rad/s{rot_tag}"
        cv2.putText(bgr, ang_txt, (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ang_color, 2)
    else:
        pause_hint = "PAUSED (Press SPACE)" if action != "ESTOP" else "LOCKED (ESTOP)"
        spd_txt = f"v_lin: [ +0.0,  +0.0,  +0.0] mm/s  [{pause_hint}]"
        cv2.putText(bgr, spd_txt, (15, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)
        ang_txt = f"w_ang: [ +0.00,  +0.00,  +0.00] rad/s [{pause_hint}]"
        cv2.putText(bgr, ang_txt, (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)

    # 手部目标偏角显示 (Joystick Tilt Angles)
    if hand_info and "d_pitch_deg" in hand_info and "d_roll_deg" in hand_info:
        tilt_status = "" if clutch_active else " (PAUSED)"
        hand_roll_tag = "左倾" if hand_info['d_roll_deg'] > 4.0 else ("右倾" if hand_info['d_roll_deg'] < -4.0 else "平")
        hand_pitch_tag = "下压" if hand_info['d_pitch_deg'] < -4.0 else ("上抬" if hand_info['d_pitch_deg'] > 4.0 else "平")
        pose_txt = (f"Tilt: Roll: {hand_info['d_roll_deg']:+5.1f}°({hand_roll_tag}) | "
                    f"Pitch: {hand_info['d_pitch_deg']:+5.1f}°({hand_pitch_tag}){tilt_status}")
        cv2.putText(bgr, pose_txt, (15, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 230, 255), 1)

    # 灵敏度档位指示
    gear_txt = f"Gear: {gear_name} (Press [S] or [TAB] to Shift Speed)"
    cv2.putText(bgr, gear_txt, (15, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.50, g_info["color"], 2)

    # 右下角绘制 6 关节 all status 实时监测表
    if joint_state is not None:
        _draw_joint_status_table(bgr, joint_state, no_drive=no_drive)

    # 绘制手腕锚点与速度矢量线
    if hand_info and hand_info.get("px_coord") is not None:
        u, v = hand_info["px_coord"]
        cv2.circle(bgr, (u, v), 8, (0, 255, 0), -1)
        cv2.circle(bgr, (u, v), 12, (255, 255, 255), 2)
        if clutch_active and cmd:
            dx = int(np.clip(cmd.linear_velocity[1] * -1.2, -60, 60))
            dy = int(np.clip(cmd.linear_velocity[0] * -1.2, -60, 60))
            if abs(dx) > 3 or abs(dy) > 3:
                cv2.arrowedLine(bgr, (u, v), (u + dx, v + dy), (0, 255, 255), 2, tipLength=0.3)


def main():
    ap = argparse.ArgumentParser(description="真机 6DOF 视觉遥操 (支持 --no-drive 空跑测试)")
    ap.add_argument("--iface", default="can0", help="SocketCAN 接口")
    ap.add_argument("--calib", default=str(Path(__file__).parent / "handeye_calib.json"))
    ap.add_argument("--out", default="datasets/teleop_real", help="录制输出目录")
    ap.add_argument("-y", "--gravity-confirm", action="store_true",
                    help="确认重力关节 J2/J3 二次确认 (真机驱动必须)")
    ap.add_argument("--no-drive", action="store_true", help="只做视觉与UI测试，不连接机械臂与CAN总线")
    ap.add_argument("--mode", choices=["knead", "roll", "pitch", "full"], default="knead",
                    help="推拿遥操姿态模式: knead(点按揉捏锁定), roll(滚法单轴Roll), pitch(俯仰单轴Pitch), full(全6DOF自由)")
    ap.add_argument("--config", default=str(Path(__file__).parent / "teleop_config.yaml"),
                    help="集中配置文件路径 (.yaml / .json, 默认 scripts/teleop/teleop_config.yaml)")
    ap.add_argument("--speed-scale", type=float, default=None,
                    help="全局平移线速度缩放比例 (覆盖配置文件)")
    ap.add_argument("--gain-xyz", type=float, default=None,
                    help="XYZ 平移扫掠动态加速度增益 (覆盖配置文件)")
    ap.add_argument("--ang-scale", type=float, default=None,
                    help="全局旋转角速度独立缩放比例 (覆盖配置文件)")
    ap.add_argument("--max-omega", type=float, default=None,
                    help="摇杆模式最大角速度 (rad/s, 覆盖配置文件)")
    ap.add_argument("--deadband-angle", type=float, default=None,
                    help="摇杆模式倾斜死区角度 (deg, 覆盖配置文件)")
    ap.add_argument("--joint-factors", default=None,
                    help="各关节独立速度倍率因子 J1..J6 (逗号分隔, 默认读取配置文件)")
    args = ap.parse_args()
    if not args.gravity_confirm and not args.no_drive:
        sys.exit("遥操前必须 -y/--gravity-confirm 确认重力关节 (J2/J3) (空跑测试请加 --no-drive)")

    # 载入集中配置文件 (优先使用 YAML/JSON，若不存在使用内置默认数据类)
    teleop_cfg = TeleopConfig.load(args.config)
    GEAR_CONFIGS.update(build_gear_configs(teleop_cfg))

    import numpy as np  # noqa: E402
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Leap_Hand" / "python"))
    import cv2  # noqa: E402
    from gesture_mapping.camera import open_realsense  # noqa: E402
    from gesture_mapping.filter import OneEuroFilter  # noqa: E402
    from gesture_mapping.hand_tracker import HandTracker  # noqa: E402
    from gesture_mapping.wrist_tracker import build_palm_pts  # noqa: E402

    cam = open_realsense()
    if cam is None:
        sys.exit("未检测到 RealSense (D455) 相机")
    tracker = HandTracker(max_num_hands=1)

    # 手眼标定旋转矩阵 (相机系 -> 基座系)
    calib_path = Path(args.calib)
    if calib_path.exists():
        from Co_Teleop.calibration import load_calib  # noqa: E402
        r_cam_to_base = load_calib(calib_path)
    else:
        r_cam_to_base = np.eye(3)

    # 解析各关节速度因子 (CLI 显式传入优先，否则使用配置文件)
    if args.joint_factors is not None:
        try:
            joint_factors = [float(x.strip()) for x in args.joint_factors.split(",") if x.strip()]
            if len(joint_factors) != 6:
                joint_factors = teleop_cfg.joint_factor.as_list()
        except Exception:
            joint_factors = teleop_cfg.joint_factor.as_list()
    else:
        joint_factors = teleop_cfg.joint_factor.as_list()

    joint_limits = teleop_cfg.joint_limits.as_list()
    joint_margin = teleop_cfg.joint_limits.joint_limit_margin_deg

    if args.no_drive:
        from Co_Teleop.adapters import NoDriveArmAdapter  # noqa: E402
        adapter = NoDriveArmAdapter(
            ready_pose=teleop_cfg.pose.ready_pose_deg,
            home_pose=teleop_cfg.pose.home_pose_deg,
            joint_limits=joint_limits,
        )
        print("[模式] 已启用 --no-drive 空跑测试模式 (仅做视觉追踪与显示计算，不连接 CAN 总线)")
    else:
        from lerobot_robot_massage.zdt.config import ZdtConfig  # noqa: E402
        from lerobot_robot_massage.zdt.controller import ZdtController  # noqa: E402
        ctrl = ZdtController(ZdtConfig(channel=args.iface,
                                       speed_rpm=teleop_cfg.motor.speed_rpm,
                                       position_acc=teleop_cfg.motor.position_acc,
                                       joint_speed_factors=joint_factors,
                                       limits=joint_limits,
                                       joint_limit_margin_deg=joint_margin,
                                       max_vel_mm_s=teleop_cfg.motor.max_vel_mm_s,
                                       max_ang_rad_s=teleop_cfg.motor.max_ang_rad_s,
                                       max_joint_vel_deg_s=teleop_cfg.motor.max_joint_vel_deg_s,
                                       max_joint_acc_deg_s2=teleop_cfg.motor.max_joint_acc_deg_s2))
        adapter = RealArmAdapter(ctrl, max_dq_deg=teleop_cfg.motor.max_dq_deg,
                                 joint_factors=joint_factors,
                                 joint_limits=joint_limits,
                                 joint_limit_margin_deg=joint_margin,
                                 ready_pose=teleop_cfg.pose.ready_pose_deg,
                                 home_pose=teleop_cfg.pose.home_pose_deg)

    watchdog = VisionWatchdog()
    recorder = EpisodeRecorder(args.out)

    # 3D 腕部位置滤波 (n_joints=3) 与 3D 姿态李群正交平滑滤波 (n_joints=9)
    pts_filter = OneEuroFilter(n_joints=3, min_cutoff=teleop_cfg.vision.pts_min_cutoff, beta=teleop_cfg.vision.pts_beta)
    rot_filter = OneEuroFilter(n_joints=9, min_cutoff=teleop_cfg.vision.rot_min_cutoff, beta=teleop_cfg.vision.rot_beta)

    latest_frame = [None]
    latest_hand = [None]
    last_wrist = [None]
    last_t = [None]

    # 时域惯性缓冲与丢帧容错 (最多保持 3 帧 / ~100ms 瞬态阴影遮挡)
    loss_count = [0]
    last_valid_info = [None]

    # 姿态控制与推拿模态跟踪变量
    current_mode = [MODE_MAP.get(args.mode, MODE_KNEAD)]
    current_gear = [teleop_cfg.gear.default_gear]  # 从配置读取默认档位
    anchor_r_hand = [None]

    # 速度独立解耦配置: 平移线速度 lin_scale 与旋转角速度 ang_scale / 摇杆参数
    lin_scale = max(0.01, min(6.0, float(args.speed_scale))) if args.speed_scale is not None else 1.0
    gain_xyz = max(1.0, min(6.0, float(args.gain_xyz))) if args.gain_xyz is not None else 1.0
    ang_scale = max(0.01, min(3.0, float(args.ang_scale))) if args.ang_scale is not None else 1.0
    max_omega = max(0.2, min(12.0, float(args.max_omega))) if args.max_omega is not None else 3.0
    deadband_angle = max(1.0, min(15.0, float(args.deadband_angle))) if args.deadband_angle is not None else teleop_cfg.vision.deadband_angle_deg
    smooth_v_base = [np.zeros(3)]
    smooth_w_base = [np.zeros(3)]

    def _joystick_rate(angle_deg: float, deadband_deg: float = 5.0, max_angle_deg: float = 28.0, max_omega_val: float = 3.0) -> float:
        """虚拟手势摇杆速率响应:
        - 倾斜角度在死区内 (|angle| <= deadband_deg): 返回 0.0 (锁定保持当前姿态)
        - 倾斜角度超过死区: 平滑输出正比于倾斜幅度的角速度 (持续持续转动)
        - 倾斜越大转动越快 (最高达 max_omega_val rad/s)
        """
        abs_ang = abs(angle_deg)
        if abs_ang <= deadband_deg:
            return 0.0
        ratio = min(1.0, (abs_ang - deadband_deg) / max(1.0, max_angle_deg - deadband_deg))
        smooth_ratio = ratio ** 1.4
        return float(np.sign(angle_deg) * smooth_ratio * max_omega_val)

    def hand_provider():
        ok, bgr, depth, K = cam.read_with_depth()
        if not ok or bgr is None:
            latest_frame[0] = None
            loss_count[0] += 1
            if loss_count[0] > 10:
                latest_hand[0] = None
                pts_filter.reset()
                rot_filter.reset()
                last_wrist[0] = None
                last_t[0] = None
            return None

        hands = tracker.detect(bgr)
        if hands:
            bgr = tracker.draw_landmarks(bgr, hands)
        latest_frame[0] = bgr

        pts = build_palm_pts(hands[0], depth, K) if hands else None
        px = (int(hands[0].landmarks[0].x * bgr.shape[1]),
              int(hands[0].landmarks[0].y * bgr.shape[0])) if hands else None

        if not hands or pts is None:
            loss_count[0] += 1
            # 优化: 1~3 帧时域惯性缓冲 (<100ms 瞬时遮挡/阴影)，维持平滑输出，防止电机顿挫
            if loss_count[0] <= 3 and last_valid_info[0] is not None:
                held = dict(last_valid_info[0])
                held["velocity"] = tuple(x * 0.95 for x in held.get("velocity", (0.0, 0.0, 0.0)))
                held["angular_velocity"] = tuple(x * 0.95 for x in held.get("angular_velocity", (0.0, 0.0, 0.0)))
                held["confidence"] = 0.75
                held["depth_valid"] = True
                latest_hand[0] = held
                return held

            # 真正持续丢失 (>100ms)
            info = {"hand_present": bool(hands), "confidence": 0.0,
                    "depth_valid": False, "wrist_mm": None, "px_coord": px}
            latest_hand[0] = info
            if loss_count[0] > 10:
                pts_filter.reset()
                rot_filter.reset()
                last_wrist[0] = None
                last_t[0] = None
            return info

        # 成功捕获有效帧: 重置丢帧计数
        loss_count[0] = 0

        # 灵敏度档位动态参数装载 (Gear Parameters)
        g_cfg = GEAR_CONFIGS.get(current_gear[0], GEAR_CONFIGS[GEAR_STANDARD])
        g_lin_scale = g_cfg["lin_scale"] * lin_scale
        g_gain_xyz = g_cfg["gain_xyz"]
        g_max_omega = g_cfg["max_omega"] * ang_scale

        # 1. 位置提取 (纯手腕 3D 深度，彻底免疫手指遮挡)
        wrist_raw = np.array(pts[0], dtype=float)
        wrist_cam = pts_filter(wrist_raw)

        # 2. 姿态提取 (解剖学刚体掌骨基底 0-5-17，基于 3D World Landmarks，彻底解耦手指弯曲)
        if hands[0].world_landmarks and len(hands[0].world_landmarks) >= 18:
            wl = hands[0].world_landmarks
            w0 = np.array([wl[0].x, wl[0].y, wl[0].z], dtype=float)
            m5 = np.array([wl[5].x, wl[5].y, wl[5].z], dtype=float)
            m17 = np.array([wl[17].x, wl[17].y, wl[17].z], dtype=float)
        else:
            lm = hands[0].landmarks
            w0 = np.array([lm[0].x, lm[0].y, lm[0].z], dtype=float)
            m5 = np.array([lm[5].x, lm[5].y, lm[5].z], dtype=float)
            m17 = np.array([lm[17].x, lm[17].y, lm[17].z], dtype=float)

        f = 0.5 * (m5 + m17) - w0
        f = f / (np.linalg.norm(f) + 1e-9)
        across = m17 - m5
        across = across / (np.linalg.norm(across) + 1e-9)
        n = np.cross(across, f)
        n = n / (np.linalg.norm(n) + 1e-9)
        lat = np.cross(f, n)
        r_raw = np.stack([f, n, lat], axis=1)

        # 李群旋转矩阵低通平滑 + SVD 严格正交重整化 (消除高频微震，保证严格正交旋转群)
        r_filtered = rot_filter(r_raw.flatten()).reshape(3, 3)
        u_svd, _, vt_svd = np.linalg.svd(r_filtered)
        r_hand = u_svd @ vt_svd
        if np.linalg.det(r_hand) < 0:
            u_svd[:, -1] *= -1
            r_hand = u_svd @ vt_svd

        pitch_deg = float(np.degrees(np.arcsin(np.clip(r_hand[1, 0], -1.0, 1.0))))
        roll_deg = float(np.degrees(np.arctan2(r_hand[2, 1], r_hand[1, 1])))

        now = time.monotonic()
        v_base = (0.0, 0.0, 0.0)
        w_base = (0.0, 0.0, 0.0)

        # 1. 笛卡尔线速度 (平移): 帧间物理差分 + 10mm/s 灵敏死区 + 跨轴正交抑制
        v_target = np.zeros(3)
        dt = 0.05
        if last_wrist[0] is not None and last_t[0] is not None:
            measured_dt = now - last_t[0]
            if 0.001 < measured_dt < 0.5:
                dt = measured_dt
                v_cam = (wrist_cam - last_wrist[0]) / dt
                v_b_raw = r_cam_to_base @ v_cam
                # 10 mm/s 单轴灵敏死区过滤 (滤除生理微颤，提升起步灵敏度)
                v_b_clamped = np.zeros(3)
                for i in range(3):
                    if abs(v_b_raw[i]) > 10.0:
                        v_b_clamped[i] = v_b_raw[i]
                # 跨轴正交抑制 (Cross-Axis Rejection): 主轴明显移动时，次轴低于 25% 视为微震耦合并清零
                max_axis_val = float(np.max(np.abs(v_b_clamped)))
                if max_axis_val > 18.0:
                    for i in range(3):
                        if abs(v_b_clamped[i]) < 0.25 * max_axis_val:
                            v_b_clamped[i] = 0.0
                v_target = v_b_clamped

        # 自适应动态滤波 (Adaptive EMA): 慢速微调稳定 (α=0.30)，快速扫掠零延迟 (α 动态提升至 0.55 超跟手)
        raw_speed = float(np.linalg.norm(v_target))
        alpha = 0.30 + 0.25 * min(1.0, max(0.0, (raw_speed - 12.0) / 50.0))
        smooth_v_base[0] = alpha * v_target + (1.0 - alpha) * smooth_v_base[0]
        if np.linalg.norm(smooth_v_base[0]) < 1.0:
            smooth_v_base[0] = np.zeros(3)

        # 鼠标级动力学加速度增益 (Dynamic Velocity Acceleration):
        # 慢动 1.0x 细腻对准，快挥 2.2x 迅速大跨度覆盖，单次挥手直达推拿目标，告别反复踩离合
        spd_filtered = float(np.linalg.norm(smooth_v_base[0]))
        if spd_filtered > 10.0:
            spd_ratio = min(1.0, (spd_filtered - 10.0) / 60.0)
            dynamic_gain = 1.0 + (g_gain_xyz - 1.0) * (spd_ratio ** 1.3)
            v_amplified = smooth_v_base[0] * dynamic_gain
        else:
            v_amplified = smooth_v_base[0]

        # 乘以当前档位平移线速度缩放比例 g_lin_scale
        v_base = tuple(float(x * g_lin_scale) for x in v_amplified)

        last_wrist[0] = wrist_cam
        last_t[0] = now

        # 2. 虚拟手势摇杆速率姿态控制 (Virtual Hand Joystick Rate Teleop) + 推拿模态解耦
        if anchor_r_hand[0] is None or not teleop.clutch_active:
            anchor_r_hand[0] = r_hand.copy()
            smooth_w_base[0] = np.zeros(3)
            w_base = (0.0, 0.0, 0.0)
            d_pitch = 0.0
            d_roll = 0.0
        else:
            # 相对旋转矩阵 R_rel (从锚点姿态到当前姿态的精确 3D 旋转)
            r_diff = r_hand @ anchor_r_hand[0].T
            axis = np.array([r_diff[2, 1] - r_diff[1, 2],
                             r_diff[0, 2] - r_diff[2, 0],
                             r_diff[1, 0] - r_diff[0, 1]])
            ax_norm = np.linalg.norm(axis)
            cos_ang = np.clip((np.trace(r_diff) - 1.0) / 2.0, -1.0, 1.0)
            ang_rad = float(np.arccos(cos_ang))

            theta_cam = np.zeros(3)
            if ax_norm > 1e-6 and ang_rad > 1e-4:
                theta_cam = (ang_rad / ax_norm) * axis

            # 将人手旋转矢量转换至机械臂基座系
            theta_base_raw = r_cam_to_base @ theta_cam
            # 提取人手当前实测俯仰与滚转倾角 (度)
            # Pitch (绕 X_base): 手腕下压为负(低头)，手腕上抬为正(抬头)
            hand_pitch_deg = float(np.degrees(theta_base_raw[0]))
            # Roll (绕 Y_base): 顺时针右翻为负，逆时针左翻为正
            hand_roll_deg = float(np.degrees(theta_base_raw[1]))

            # 方案 2: 推拿模态解耦约束 (Tuina Modes) 下的虚拟摇杆速率生成
            m = current_mode[0]
            if m == MODE_KNEAD:
                # 模式 1: 垂直点按揉捏 (姿态全锁定，角速度严格为 0，彻底免疫五指揉捏时的一切干扰)
                w_pitch = 0.0
                w_roll = 0.0
            elif m == MODE_ROLL:
                # 模式 2: 滚法推法 (单轴 Roll 摇杆，Pitch 强制锁定为 0)
                # 倾斜手腕持续滚转，手放平立即锁定保持当前角度
                w_pitch = 0.0
                w_roll = _joystick_rate(hand_roll_deg, deadband_deg=deadband_angle,
                                       max_angle_deg=28.0, max_omega_val=g_max_omega)
            elif m == MODE_PITCH:
                # 模式 3: 俯仰调节 (单轴 Pitch 摇杆，Roll 强制锁定为 0)
                # 下压手腕持续低头，上抬手腕持续抬头，手放平立即锁定保持当前倾角
                w_pitch = _joystick_rate(hand_pitch_deg, deadband_deg=deadband_angle,
                                         max_angle_deg=22.0, max_omega_val=g_max_omega)
                w_roll = 0.0
            else:
                # 模式 4: 全 6-DOF 自由摇杆姿态调节
                w_pitch = _joystick_rate(hand_pitch_deg, deadband_deg=deadband_angle,
                                         max_angle_deg=22.0, max_omega_val=g_max_omega)
                w_roll = _joystick_rate(hand_roll_deg, deadband_deg=deadband_angle,
                                       max_angle_deg=28.0, max_omega_val=g_max_omega)

            w_target = np.array([w_pitch, w_roll, 0.0])

            # EMA 低通平滑输出 (α=0.35)，消除采样微跳，实现丝滑连续转动
            smooth_w_base[0] = 0.35 * w_target + 0.65 * smooth_w_base[0]
            if np.linalg.norm(smooth_w_base[0]) < 0.01:
                smooth_w_base[0] = np.zeros(3)
            w_base = tuple(float(x) for x in smooth_w_base[0])

            d_pitch = hand_pitch_deg
            d_roll = hand_roll_deg

        info = {"hand_present": True, "confidence": 0.9, "depth_valid": True,
                "wrist_mm": tuple(float(v) for v in pts[0]),
                "velocity": v_base,
                "angular_velocity": w_base,
                "pitch_deg": pitch_deg,
                "roll_deg": roll_deg,
                "d_pitch_deg": d_pitch,
                "d_roll_deg": d_roll,
                "px_coord": px}
        last_valid_info[0] = info
        latest_hand[0] = info
        return info

    curr_key = -1

    # 默认以离合暂停态 (default_clutch_active=False) 启动，操作员准备好后按 SPACE 激活
    teleop = RealArmTeleop(adapter, watchdog, recorder, hand_provider,
                           key_provider=lambda: curr_key,
                           deadband_mm_s=0.05,
                           deadband_rad_s=0.0005,
                           default_clutch_active=False)
    adapter.connect()
    adapter.arm(gravity_confirmed=True)
    adapter.enter_teleop()
    adapter.ready()
    recorder.start_episode()
    WIN_NAME = "RealArmTeleop 6DOF"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 720)

    cached_joint_state = [None]
    last_joint_poll = [0.0]

    try:
        while True:
            now = time.monotonic()
            out = teleop.run_once(cmd_ts=now, now=now)
            if out["action"] == "QUIT":
                break
            if out["action"] == "ESTOP":
                # 急停触发后保持窗口运行，不主动 break，显示红色警告提示并允许按键恢复/退出
                pass
            elif out["action"] == "READY":
                print("[姿态] 正在安全同步运动至按摩准备姿态 (READY)...")
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
            elif out["action"] == "HOME":
                print("[姿态] 正在安全同步运动至上电初始姿态 (HOME)...")
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)

            # ~12 Hz 周期轮询 6 关节 all status (pos + current + flag)
            if now - last_joint_poll[0] >= 0.08:
                try:
                    cached_joint_state[0] = adapter.get_joint_state()
                except Exception:  # noqa: BLE001
                    pass
                last_joint_poll[0] = now

            if latest_frame[0] is not None:
                _draw_overlay(latest_frame[0], out, latest_hand[0], teleop.clutch_active,
                              no_drive=args.no_drive, lin_scale=lin_scale, ang_scale=ang_scale,
                              joint_state=cached_joint_state[0], mode=current_mode[0],
                              gear=current_gear[0])
                cv2.imshow(WIN_NAME, latest_frame[0])

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("s"), ord("S"), 9):  # 9 是 TAB 键，S 也是 Speed/Shift 快捷键
                current_gear[0] = (current_gear[0] % 3) + 1
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                print(f"[档位切换] 当前灵敏度档位: {GEAR_CONFIGS[current_gear[0]]['name']}")
            elif k in (ord("m"), ord("M")):
                current_mode[0] = (current_mode[0] % 4) + 1
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
                print(f"[模式切换] 当前推拿姿态模式: {MODE_NAMES[current_mode[0]]}")
            elif k == ord("1"):
                current_mode[0] = MODE_KNEAD
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
                print(f"[模式切换] 当前推拿姿态模式: {MODE_NAMES[MODE_KNEAD]}")
            elif k == ord("2"):
                current_mode[0] = MODE_ROLL
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
                print(f"[模式切换] 当前推拿姿态模式: {MODE_NAMES[MODE_ROLL]}")
            elif k == ord("3"):
                current_mode[0] = MODE_PITCH
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
                print(f"[模式切换] 当前推拿姿态模式: {MODE_NAMES[MODE_PITCH]}")
            elif k == ord("4"):
                current_mode[0] = MODE_FULL
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
                print(f"[模式切换] 当前推拿姿态模式: {MODE_NAMES[MODE_FULL]}")
            elif k in (ord("c"), ord("C")):
                # 按 C 键即时重校准姿态零点
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
                print("[校准] 手势姿态零点已重新校准 (Re-centered)")
            curr_key = k if k != 255 else -1
            try:
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:  # noqa: BLE001
                pass
    finally:
        try:
            adapter.e_stop()
        except Exception:  # noqa: BLE001
            pass
        recorder.finish_episode()
        adapter.disconnect()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
