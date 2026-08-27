"""机械臂视觉遥操 demo: 人手 6DOF → 末端位姿目标 → 位置/姿态环 → end_event 速度.

用法:
  # 仿真臂 (先另开终端: conda activate smolvla && python scripts/mujoco_sim.py --ik --no-camera)
  conda activate leap_hand
  python Arm-robot_VLA/scripts/demo_arm_teleop.py --port socket://localhost:5555

  # 真机臂 (M3: 需固件支持 get_ee_pose/FK, 当前无反馈 → 不出命令)
  python Arm-robot_VLA/scripts/demo_arm_teleop.py --port /dev/ttyUSB0

控制范式: 末端 6DOF 位姿跟随 (方案 C 动态锚点).
按住 H 时捕获手锚点+末端锚点; 按住期间手位置增量 → 末端位置目标 (位置环→v_lin),
手姿态增量 → 末端姿态目标 (姿态环→w_ang), 经 end_event vx vy vz wx wy wz 驱动全 IK.
仿真经 get_ee_pose 读末端位姿 (m+wxyz) 作反馈; 真机无反馈时返回全 0 (不差分降级).

按键:
  H (按住)   离合器: 按住跟随, 松开=重锚定 (走哪停哪)
  M          录制复位点: 1-6选关节, W/S微调(±5°), S保存到 home_pose.json, M退出
  R          复位: 回手动复位点(若有)或默认初始位, 等待归位后重新锚定
  C          重载 handeye_calib.json
  K          轴对齐校准向导 (手沿3方向挥动+选1-6方向码, 自动求解手眼R并保存)
  Y          e_stop
  Q/ESC      退出

历史路径: 此文件原位于 Leap_Hand/python/gesture_mapping/demo_arm_teleop.py,
2026-08 因归属错误迁移到本仓. Leap_Hand 同名 demo 于同次提交删除.

依赖结构 (跨仓库):
  本仓 (同目录 scripts/):     arm_client (串口薄客户端), handeye_calib
  Leap_Hand 仓 (sys.path):    gesture_mapping.{camera, filter,
                              hand_tracker, wrist_tracker} (视觉 + 手跟踪共享模块)
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ─── sys.path 注入 ───
# 1. 本仓 scripts/teleop/ (含 arm_client / handeye_calib)
sys.path.insert(0, str(Path(__file__).resolve().parent))
# 2. Leap_Hand 仓的视觉 + 手跟踪共享模块 (跨仓库)
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Leap_Hand" / "python"))

# arm-only (本仓)
from arm_client import ArmClient  # noqa: E402
from handeye_calib import load_calib, save_calib, solve_handeye  # noqa: E402

# Leap_Hand shared (跨仓库 sys.path 注入)
from gesture_mapping.camera import open_realsense  # noqa: E402
from gesture_mapping.filter import OneEuroFilter  # noqa: E402
from gesture_mapping.hand_tracker import HandTracker  # noqa: E402
from gesture_mapping.wrist_tracker import WristTracker, build_palm_pts  # noqa: E402

_KEYS = {
    ord("h"): "clutch", ord("H"): "clutch",
    ord("r"): "reset", ord("R"): "reset",
    ord("c"): "calib", ord("C"): "calib",
    ord("k"): "calib", ord("K"): "calib",
    ord("m"): "record", ord("M"): "record",
    ord("y"): "estop", ord("Y"): "estop",
    ord("q"): "quit", 27: "quit",
}

_CALIB_STEP_HINTS = [
    "步骤1: 手沿画面\"向右\"移动一段距离(按住H), 完成后按 SPACE, 再按 1-6 选臂应去方向",
    "步骤2: 手沿画面\"向上\"移动一段距离(按住H), 完成后按 SPACE, 再按 1-6",
    "步骤3: 手\"向前(靠近相机)\"移动一段距离(按住H), 完成后按 SPACE, 再按 1-6",
]
_DIR_CODE_HINT = "方向码: 1=+X 2=-X 3=+Y 4=-Y 5=+Z(上) 6=-Z(下)"

# 软复位目标: 各关节初始位 J1..J6 (与 zdt/config.py JOINT_INIT_ANGLE_DEG 权威值一致)
INIT_POSE_DEG = [90.0, 90.0, -90.0, 0.0, 90.0, 0.0]
HOME_POSE_PATH = Path(__file__).resolve().parent / "home_pose.json"
REC_STEP_DEG = 5.0        # 录制模式单关节步进角度


def _load_home_pose():
    """读手动录制复位点 (home_pose.json) → [j1..j6] 或 None."""
    if HOME_POSE_PATH.exists():
        try:
            data = json.loads(HOME_POSE_PATH.read_text())
            return [float(a) for a in data["angles"][:6]]
        except Exception:
            pass
    return None


def _save_home_pose(angles):
    """保存当前关节角为复位点."""
    HOME_POSE_PATH.write_text(json.dumps(
        {"angles": [round(float(a), 2) for a in angles[:6]]}))
    print(f"[录制] 复位点已保存到 {HOME_POSE_PATH}: "
          f"{[round(float(a), 1) for a in angles[:6]]}")


def main():
    ap = argparse.ArgumentParser(description="机械臂视觉遥操 (末端6DOF位姿跟随)")
    ap.add_argument("--port", default="socket://localhost:5555",
                    help="串口或 socket:// (默认仿真 5555)")
    ap.add_argument("--calib", default=str(
        Path(__file__).resolve().parent / "handeye_calib.json"))
    ap.add_argument("--no-drive", action="store_true",
                    help="只显示速度, 不发送命令")
    args = ap.parse_args()

    cam = open_realsense()
    if cam is None:
        sys.exit("未检测到 RealSense (D455) 相机")
    tracker = HandTracker(max_num_hands=1)

    R = None
    if Path(args.calib).exists():
        R = load_calib(args.calib)
        print(f"[标定] 已加载 handeye: {args.calib}")
    else:
        R = np.eye(3)
        print("[标定] 未找到 handeye_calib.json, 使用单位旋转 (仅测试)")

    wt = WristTracker(R=R)
    cmd_smoother = OneEuroFilter(6, min_cutoff=8.0, beta=0.08)
    arm = None if args.no_drive else ArmClient(args.port)
    if arm is not None:
        arm.remote_enable()
        try:
            angles, _, _ = arm.get_state()
            wrist0 = None
            w = arm.get_wrist()
            if w is not None:
                wrist0 = np.array(w) * 1000.0
            ee_pose0 = None
            ep = arm.get_ee_pose()
            if ep is not None:
                ee_pose0 = (np.array(ep[0]) * 1000.0, ep[1])   # 位置米→mm, 四元数不变
            if len(angles) >= 6:
                wt.capture(None, wrist0, ee_pose0, angles[4], angles[3])
        except Exception:
            pass
        print(f"[臂] 已连接 {args.port}")

    clutch = False
    reset_hold = 0        # 复位等待帧计数 (~5s @30fps)

    # 录制复位点状态 (M 进入)
    home_pose = _load_home_pose()
    if home_pose is not None:
        print(f"[复位点] 已加载手动复位点: {[round(a,1) for a in home_pose]}")
    record_mode = False
    record_joint = 0

    # 校准状态机 (K 进入轴对齐向导)
    calib_step = 0            # 0=off, 1/2/3=收集第几步
    calib_buf = []            # 滚动 wrist_cam 缓冲
    calib_cam = []            # 已确认的相机系单位方向
    calib_codes = []          # 对应基座方向码
    calib_pending = None      # 待选方向的 cam 单位向量
    CALIB_BUF_MAX = 20

    print("\n按键: H=离合器(按住跟随,松开重锚定)  R=复位  C=重载标定  K=轴对齐校准向导  Y=急停  Q=退出\n")
    print("[控制范式] 末端6DOF位姿跟随: 按住H后, 手相对锚点的位置/姿态增量 → 末端目标位姿, "
          "位置环(v_lin)+姿态环(w_ang)将臂末端驱动到目标; "
          "松开H重锚定当前手位+末端位姿, 走哪停哪. "
          "手在画面中的运动方向经手眼标定R映射到机械臂基座系.\n")

    try:
        while True:
            ok, bgr, depth, K = cam.read_with_depth()
            if not ok or bgr is None:
                continue
            hands = tracker.detect(bgr)
            hand = hands[0] if hands else None
            pts = build_palm_pts(hand, depth, K) if hand is not None else None

            # 读腕心+末端位姿+关节反馈 (仿真 get_wrist/get_ee_pose; 无 arm 时用 None/零)
            wrist_mm = None
            ee_pose = None
            j4c = j5c = 0.0
            if arm is not None:
                w = arm.get_wrist()
                if w is not None:
                    wrist_mm = np.array(w) * 1000.0
                ep = arm.get_ee_pose()
                if ep is not None:
                    ee_pose = (np.array(ep[0]) * 1000.0, ep[1])   # 位置米→mm, 四元数不变
                angles, _, _ = arm.get_state()
                if len(angles) >= 6:
                    j4c, j5c = angles[3], angles[4]

            key = cv2.waitKey(1) & 0xFF

            # 校准帧采集: 记录相机系 wrist 位移缓冲 (未过 R)
            if 1 <= calib_step <= 3 and pts is not None:
                calib_buf.append(pts[0])
                if len(calib_buf) > CALIB_BUF_MAX:
                    calib_buf.pop(0)

            # 校准向导按键 (独立于 _KEYS)
            if 1 <= calib_step <= 3 and key == ord(" ") and pts is not None and len(calib_buf) >= 5:
                d = calib_buf[-1] - calib_buf[0]
                if np.linalg.norm(d) < 30.0:
                    print("位移太小, 重试 (沿该方向移动更远距离)")
                else:
                    calib_pending = d / np.linalg.norm(d)
                    print(f"采集到相机系方向 {np.round(calib_pending, 3)}. " + _DIR_CODE_HINT)
            elif calib_pending is not None and ord("1") <= key <= ord("6"):
                code = key - ord("1") + 1
                calib_cam.append(calib_pending)
                calib_codes.append(code)
                calib_pending = None
                calib_buf = []      # 步骤间清空, 下步 SPACE 只采当前方向位移
                if len(calib_codes) == 3:
                    R = solve_handeye(calib_cam, calib_codes)
                    save_calib(args.calib, R)
                    wt.R = R
                    wt.capture(None, None, None, 0.0, 0.0)   # 清旧参考, 避免 R 系混用
                    print(f"[校准] R 已保存到 {args.calib}:\n{R}")
                    print("CALIB: 验证 Z 方向 - 手向相机移动, 确认臂朝期望方向; 反了按 Z 翻转, 正常按 SPACE 完成")
                    calib_step = 4
                    calib_buf = []
                else:
                    calib_step += 1
                    print(_CALIB_STEP_HINTS[calib_step - 1])
            elif calib_step == 4 and key == ord(" "):
                calib_step = 0
                calib_buf = []
                print("校准完成")
            elif calib_step == 4 and key in (ord("z"), ord("Z")):
                R = np.asarray(R, float) @ np.diag([1.0, 1.0, -1.0])
                save_calib(args.calib, R)
                wt.R = R
                wt.capture(None, None, None, 0.0, 0.0)
                print("已翻转 Z 方向并保存")
            # ── 录制复位点模式 (M 进入/退出) ──
            elif record_mode:
                if ord("1") <= key <= ord("6"):
                    record_joint = key - ord("1")
                    print(f"[录制] 选中 J{record_joint+1} "
                          f"(W/↑=正转, X/↓=反转 {REC_STEP_DEG:g}°, S=保存复位点)")
                elif key in (ord("w"), ord("W"), 82):      # W / ↑: 正转
                    if arm is not None:
                        arm.rel_rotate(record_joint + 1, +REC_STEP_DEG)
                    print(f"[录制] J{record_joint+1} +{REC_STEP_DEG:g}°")
                elif key in (ord("x"), ord("X"), 84):      # X / ↓: 反转
                    if arm is not None:
                        arm.rel_rotate(record_joint + 1, -REC_STEP_DEG)
                    print(f"[录制] J{record_joint+1} -{REC_STEP_DEG:g}°")
                elif key in (ord("s"), ord("S")):          # S: 保存复位点
                    angles, _, _ = arm.get_state()
                    if len(angles) >= 6:
                        _save_home_pose(angles)
                        home_pose = [float(a) for a in angles[:6]]
                elif key in (ord("m"), ord("M")):
                    record_mode = False
                    print("[录制] 退出录制模式, 恢复遥操")
            elif key in _KEYS:
                action = _KEYS[key]
                if action == "clutch":
                    clutch = not clutch
                    wt.capture(pts, wrist_mm, ee_pose, j5c, j4c)   # 按下/松开都重锚定(手参考+末端锚点)
                elif action == "calib" and key in (ord("k"), ord("K")):
                    if calib_step == 0:
                        calib_step = 1
                        calib_buf = []
                        calib_cam = []
                        calib_codes = []
                        calib_pending = None
                        print(_DIR_CODE_HINT)
                        print(_CALIB_STEP_HINTS[0])
                    else:
                        calib_step = 0
                        calib_pending = None
                        calib_buf = []
                        print("[校准] 已退出")
                elif action == "calib" and Path(args.calib).exists():
                    R = load_calib(args.calib)
                    wt.R = R
                    wt.capture(None, None, None, 0.0, 0.0)   # 清旧参考, 避免 R 系混用
                    print("[标定] 已重载 handeye")
                elif action == "estop" and arm is not None:
                    arm.e_stop()
                    print("[急停] e_stop")
                elif action == "record":
                    record_mode = not record_mode
                    record_joint = 0
                    if record_mode:
                        print("[录制] 进入复位点录制: 1-6选关节, W/↑正转, X/↓反转, "
                              "S保存复位点, M退出")
                    else:
                        print("[录制] 退出录制模式")
                elif action == "reset":
                    if arm is not None:
                        if home_pose is not None:
                            arm.set_joints(home_pose)
                            print(f"[复位] 回手动复位点 {[round(a,1) for a in home_pose]}")
                        else:
                            arm.soft_reset()
                            print("[复位] 回默认初始位")
                        reset_hold = 150    # ~5s @30fps, 让仿真归位
                    else:
                        print("[复位] 无臂连接 (--no-drive), 忽略")
                elif action == "quit":
                    break

            if reset_hold > 0:
                reset_hold -= 1
                if reset_hold == 0:
                    ep = arm.get_ee_pose()
                    angles, _, _ = arm.get_state()
                    ee_pose = (np.array(ep[0]) * 1000.0, ep[1]) if ep is not None else None
                    j5c = angles[4] if len(angles) >= 6 else 0.0
                    j4c = angles[3] if len(angles) >= 6 else 0.0
                    wt.capture(pts, wrist_mm, ee_pose, j5c, j4c)
                    print("[复位] 完成, 已重新锚定")
                    if len(angles) >= 6:
                        init_ref = home_pose if home_pose is not None else INIT_POSE_DEG
                        dev = max(abs(a - init) for a, init in zip(angles, init_ref))
                        if dev > 20.0:
                            print(f"[复位] 警告: 臂未归到初始位 (最大偏差 {dev:.0f}°)")
                cmd = wt.no_hand()
            elif pts is None:
                cmd = wt.update(None, wrist_mm, ee_pose, j5c, j4c)
            elif clutch:
                cmd = wt.update(pts, wrist_mm, ee_pose, j5c, j4c)
            else:
                cmd = wt.no_hand()
                wt.capture(pts, wrist_mm, ee_pose, j5c, j4c)   # 未按住时也持续重锚定(手参考+末端锚点)

            cmd = cmd_smoother(np.array(cmd))
            if arm is not None:
                arm.end_event(*cmd)

            # HUD
            h, w = bgr.shape[:2]
            if calib_step > 0:
                if calib_step == 4:
                    cv2.putText(bgr, "CALIB: 验证 Z 方向 - 手向相机移动, 反了按 Z 翻转, 正常按 SPACE 完成",
                                (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 200, 255), 2)
                else:
                    cv2.putText(bgr,
                                f"CALIB: step {calib_step} | pending: "
                                f"{'Y' if calib_pending is not None else 'N'} "
                                f"| pairs: {len(calib_codes)}/3",
                                (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 200, 255), 2)
            cv2.putText(bgr, f"CLUTCH:{'ON' if clutch else 'OFF'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if clutch else (0, 0, 255), 2)
            if record_mode:
                cur = (f"[{', '.join(f'{a:.0f}' for a in angles[:6])}]"
                       if "angles" in dir() and len(angles) >= 6 else "")
                cv2.putText(bgr, f"REC J{record_joint+1} {cur} (W/X转 S存)",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
            cv2.putText(bgr, f"v=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                             f"W=({cmd[3]:+.2f},{cmd[4]:+.2f},{cmd[5]:+.2f})",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(bgr, f"roll={wt.last_roll_deg:+.1f}deg pitch={wt.last_pitch_deg:+.1f}deg",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            d = wt.last_delta_base
            cv2.putText(bgr, f"d=({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f})mm ANCHOR:set",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if ee_pose is not None:
                t = wt.last_target_ee
                e = t - ee_pose[0]
                cv2.putText(bgr, f"tgt=({t[0]:+.0f},{t[1]:+.0f},{t[2]:+.0f})mm "
                                 f"err=({e[0]:+.0f},{e[1]:+.0f},{e[2]:+.0f})mm",
                            (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
            if pts is not None:
                cv2.putText(bgr,
                            f"depth={pts[0][2]:.0f}mm roll={wt.last_roll_deg:+.1f}deg "
                            f"pitch={wt.last_pitch_deg:+.1f}deg",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)
            cv2.imshow("Arm Teleop", bgr)
    finally:
        if arm is not None:
            arm.remote_disable()
            arm.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[退出] 已安全断开")


if __name__ == "__main__":
    main()