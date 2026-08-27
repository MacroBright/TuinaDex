"""Co_Teleop.pipeline.unified_teleop — 机械臂-灵巧手协同视觉遥操统一主控管线.

【架构特性】
1. 单目视觉复用 (Single Perception Stream):
   - 单台 RealSense D455 相机, 单次 MediaPipe 3D 手势检测;
   - 同时解算手腕 3D 宏观轨迹 (驱动 6DOF 机械臂) 与五指 21 关节点微观几何 (驱动 16DOF 灵巧手);
2. 臂-手解耦控制 (Decoupled Arm-Hand Kinematics):
   - 手腕位移 + 掌面倾角 -> 机械臂空间平移与推拿模态 (点按/滚法/俯仰/全自由);
   - 五指屈伸 -> 灵巧手 16 舵机抓握与揉捏;
3. 统一安全看门狗与离合器 (Unified Watchdog & Clutch):
   - 丢帧/遮挡时: 机械臂平稳减速悬停, 灵巧手执行 relax_step 平滑回全开位 (OPEN_POSE);
4. 一体化综合 HUD 监控看板 (Single Unified Dashboard):
   - 1280x720 宽屏 HUD: 手部骨骼 + 虚拟摇杆 + 机械臂 6 轴状态表 + 灵巧手 16 舵机弯曲柱状图;
5. 集中配置即改即用: 自动加载 teleop_config.yaml 并执行起飞前安全检查.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional

import cv2
import numpy as np

# 根目录与子系统模块路径注入
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
for p in (WORKSPACE_ROOT, WORKSPACE_ROOT / "Arm-robot_VLA", WORKSPACE_ROOT / "Leap_Hand" / "python", WORKSPACE_ROOT / "Co_Teleop"):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Co_Teleop 模块导入
from Co_Teleop.adapters import (
    ArmAdapter,
    RealArmAdapter,
    SimArmAdapter,
    NoDriveArmAdapter,
    LeapHandAdapter,
    NoDriveHandAdapter,
    JointState,
    EEPose,
    CartesianCommand,
)
from Co_Teleop.config import (
    TeleopConfig,
    build_gear_configs,
)
from Co_Teleop.safety import VisionWatchdog, WatchdogAction
from Co_Teleop.calibration import load_calib
from Co_Teleop.gui import (
    draw_unified_dashboard,
    MODE_KNEAD,
    MODE_ROLL,
    MODE_PITCH,
    MODE_FULL,
    MODE_NAMES,
    FINGER_NAMES,
)

# 灵巧手视觉算法组件
from gesture_mapping import Calibrator, FingerIdentifier, HandTracker, JointMapper
from gesture_mapping.camera import open_realsense
from gesture_mapping.filter import OneEuroFilter
from gesture_mapping.hamer_3d import HaMeR3D, hand_bbox_from_landmarks
from gesture_mapping.wrist_tracker import build_palm_pts

MODE_MAP = {"knead": MODE_KNEAD, "roll": MODE_ROLL, "pitch": MODE_PITCH, "full": MODE_FULL}
SOURCE_NAMES = {0: "HAMER 3D", 1: "WORLD 3D", 2: "MP PSEUDO-3D"}
_MIRRORED_LABEL = {"right": "left", "left": "right"}


def _smoothed_frame(pts, smoother):
    """计算掌心参考系并对 normal/mid_dir/lateral 进行时域正交平滑."""
    wrist, normal, mid_dir, lateral = JointMapper._palm_frame(pts)
    fvec = smoother(np.concatenate([normal, mid_dir, lateral]))
    normal, mid_dir, lateral = fvec[:3], fvec[3:6], fvec[6:9]
    for v in (normal, mid_dir):
        n = np.linalg.norm(v)
        if n > 1e-9:
            v /= n
    lateral = lateral - np.dot(lateral, mid_dir) * mid_dir
    n = np.linalg.norm(lateral)
    if n > 1e-9:
        lateral /= n
    else:
        lateral = np.cross(normal, mid_dir)
        n = np.linalg.norm(lateral)
        if n > 1e-9:
            lateral /= n
    return (wrist, normal, mid_dir, lateral)


def main():
    default_cfg = Path(__file__).resolve().parents[1] / "config" / "teleop_config.yaml"
    default_calib = Path(__file__).resolve().parents[1] / "calibration" / "handeye_calib.json"

    ap = argparse.ArgumentParser(description="真机 6DOF 机械臂 + 16DOF 灵巧手视觉遥操统一管线")
    ap.add_argument("--iface", default="can0", help="SocketCAN 接口 (机械臂)")
    ap.add_argument("--hand-port", default=None, help="灵巧手串口路径 (默认从配置文件读取或自动扫描)")
    ap.add_argument("--config", default=str(default_cfg), help="集中配置文件路径 (默认 teleop_config.yaml)")
    ap.add_argument("--calib", default=str(default_calib), help="手眼标定矩阵路径")
    ap.add_argument("-y", "--gravity-confirm", action="store_true", help="确认重力关节 J2/J3 (机械臂真机必须)")
    ap.add_argument("--no-drive-arm", action="store_true", help="机械臂空跑测试 (不连 CAN 总线)")
    ap.add_argument("--no-drive-hand", action="store_true", help="灵巧手空跑测试 (不连 Dynamixel 串口)")
    ap.add_argument("--mode", choices=["knead", "roll", "pitch", "full"], default="roll",
                    help="推拿遥操姿态模式: knead, roll, pitch, full")
    args = ap.parse_args()

    if not args.gravity_confirm and not args.no_drive_arm:
        sys.exit("遥操前必须 -y/--gravity-confirm 确认重力关节 (J2/J3) (空跑测试请加 --no-drive-arm)")

    # 1. 载入并验证全局集中配置 (起飞前安全检查)
    teleop_cfg = TeleopConfig.load(args.config)
    gear_configs = build_gear_configs(teleop_cfg)

    # 2. 机械臂适配器初始化
    joint_factors = teleop_cfg.joint_factor.as_list()
    joint_limits = teleop_cfg.joint_limits.as_list()
    joint_margin = teleop_cfg.joint_limits.joint_limit_margin_deg

    if args.no_drive_arm:
        arm_adapter = NoDriveArmAdapter(
            ready_pose=teleop_cfg.pose.ready_pose_deg,
            home_pose=teleop_cfg.pose.home_pose_deg,
            joint_limits=joint_limits,
        )
        print("[机械臂] 已启用 --no-drive-arm 空跑测试模式 (按 R 运动至准备姿态)")
    else:
        from lerobot_robot_massage.zdt.config import ZdtConfig
        from lerobot_robot_massage.zdt.controller import ZdtController
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
        arm_adapter = RealArmAdapter(ctrl, max_dq_deg=teleop_cfg.motor.max_dq_deg,
                                     joint_factors=joint_factors,
                                     joint_limits=joint_limits,
                                     joint_limit_margin_deg=joint_margin,
                                     ready_pose=teleop_cfg.pose.ready_pose_deg,
                                     home_pose=teleop_cfg.pose.home_pose_deg)
        print(f"[机械臂] 正在连接 SocketCAN ({args.iface}) 并初始化 6 个电机...")
        arm_adapter.connect()
        arm_adapter.arm(gravity_confirmed=True)
        arm_adapter.enter_teleop()
        print(f"[机械臂] 6-DOF 电机已成功上电使能 (状态: {arm_adapter.state()}, 请按 R 键开始运动至准备姿态)")

    # 3. 灵巧手适配器与映射器初始化
    if args.no_drive_hand:
        hand_adapter = NoDriveHandAdapter()
        hand_adapter.connect()
        print("[灵巧手] 已启用 --no-drive-hand 空跑测试模式 (不连接串口)")
    else:
        target_port = args.hand_port or teleop_cfg.hand.port
        hand_adapter = LeapHandAdapter(
            port=target_port,
            kP=teleop_cfg.hand.kP,
            kI=teleop_cfg.hand.kI,
            kD=teleop_cfg.hand.kD,
            curr_lim=teleop_cfg.hand.curr_lim,
        )
        print(f"[灵巧手] 实体驱动就绪 (按 K 键校准并上电): port={target_port} (kP={teleop_cfg.hand.kP}, kD={teleop_cfg.hand.kD}, curr_lim={teleop_cfg.hand.curr_lim}mA)")

    mapper = JointMapper()
    calibrator = Calibrator(mapper)
    finger_id = FingerIdentifier(mapper, bend_threshold=teleop_cfg.hand.bend_threshold)
    h3d = HaMeR3D()

    # 4. 视觉感知与手眼标定
    cam = open_realsense()
    if cam is None:
        sys.exit("[错误] 未检测到 RealSense D455 深度相机")
    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.35)

    calib_path = Path(args.calib)
    if calib_path.exists():
        r_cam_to_base = load_calib(calib_path)
    else:
        r_cam_to_base = np.eye(3)

    # 5. 滤波器与安全看门狗
    pts_filter = OneEuroFilter(n_joints=3, min_cutoff=teleop_cfg.vision.pts_min_cutoff, beta=teleop_cfg.vision.pts_beta)
    rot_filter = OneEuroFilter(n_joints=9, min_cutoff=teleop_cfg.vision.rot_min_cutoff, beta=teleop_cfg.vision.rot_beta)
    hand_angle_filter = OneEuroFilter(n_joints=16, min_cutoff=teleop_cfg.hand.filter_min_cutoff, beta=teleop_cfg.hand.filter_beta)
    pseudo_smoother = OneEuroFilter(n_joints=63, min_cutoff=1.5, beta=0.004)
    world_smoother = OneEuroFilter(n_joints=63, min_cutoff=1.2, beta=0.004)
    frame_smoother = OneEuroFilter(n_joints=9, min_cutoff=1.0, beta=0.005)

    watchdog = VisionWatchdog()

    # 6. 控制循环状态变量
    current_mode = [MODE_MAP.get(args.mode, MODE_ROLL)]
    current_gear = [teleop_cfg.gear.default_gear]
    clutch_active = [False]         # 机械臂跟随离合器状态 (默认以 PAUSE 安全离合态启动，按 SPACE 激活)
    hand_clutch_active = [True]     # 灵巧手跟随离合器状态 (L 键控制)
    source_mode = [teleop_cfg.hand.source_mode]

    last_wrist = [None]
    last_t = [None]
    anchor_r_hand = [None]
    smooth_v_base = [np.zeros(3)]
    smooth_w_base = [np.zeros(3)]
    loss_count = [0]

    cached_joint_state = [arm_adapter.get_joint_state()]
    last_joint_poll = [0.0]
    hand_angles = np.zeros(16, dtype=np.float64)
    hand_bent = [False, False, False, False]

    prev_loop_t = time.monotonic()
    fps = 0.0

    WIN_NAME = "TuinaDex — Arm & LeapHand Unified Visual Teleoperation"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    print("\n" + "=" * 80)
    print("  TuinaDex 机械臂-灵巧手协同视觉遥操统一系统 (全视野 1280x720 宽屏 HUD)")
    print("  SPACE: 机械臂跟随 | R: 准备姿态(READY) | H: 初始复位(HOME) | L: 灵巧手锁定 | Z: 姿态回零 | K: 灵巧手校准 | Q: 退出")
    print("=" * 80 + "\n")

    try:
        while True:
            now = time.monotonic()
            dt = now - prev_loop_t
            prev_loop_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

            # ── 1. 单目视觉采集与感知 ────────────────────────────────
            ok, bgr, depth, K = cam.read_with_depth()
            if not ok or bgr is None:
                time.sleep(0.01)
                continue

            # 同步镜像翻转 BGR 与 Depth，确保全局视野深度反投影完全对齐
            frame_native = cv2.flip(bgr, 1)
            depth_flipped = cv2.flip(depth, 1) if depth is not None else None

            # 内参主点 cx 对应水平镜像变换: cx_mirrored = depth_w - cx
            if K is not None and depth is not None:
                fx, fy, cx, cy = K
                K_mirrored = (fx, fy, float(depth.shape[1] - cx), cy)
            else:
                K_mirrored = K

            # 在原始无形变分辨率上运行 MediaPipe 检测
            results = tracker.detect(frame_native)

            # 构建 1280x720 宽屏显示画布
            if frame_native.shape[1] != 1280 or frame_native.shape[0] != 720:
                frame = cv2.resize(frame_native, (1280, 720), interpolation=cv2.INTER_LINEAR)
            else:
                frame = frame_native.copy()
            h, w = frame.shape[:2]

            hand_detected = False
            arm_action = "MOVE"
            scaled_v = np.zeros(3)
            scaled_w = np.zeros(3)
            roll_deg = 0.0
            pitch_deg = 0.0
            wrist_px = None
            palm_pts = None

            if results:
                # 匹配物理操作手 (默认右手)
                target_hand_label = _MIRRORED_LABEL.get(teleop_cfg.hand.hand_type, teleop_cfg.hand.hand_type)
                matched_hand = None
                for r in results:
                    if teleop_cfg.hand.hand_type == "first" or r.handedness.lower() == target_hand_label:
                        matched_hand = r
                        break

                # 边界容错降级: 若视野中仅检测到 1 只手，避免因边界裁剪导致左右手判别跳变而丢帧
                if matched_hand is None and len(results) == 1:
                    matched_hand = results[0]

                if matched_hand is not None:
                    hand_detected = True
                    wrist_px = (int(matched_hand.landmarks[0].x * w), int(matched_hand.landmarks[0].y * h))
                    hand_adapter.reset_loss_state()
                    frame = tracker.draw_landmarks(frame, [matched_hand])
                    mp_pts = tracker.landmark_xy(matched_hand, (h, w))

                    # ── 2. 机械臂手腕 6DOF 解算 ──
                    palm_pts = build_palm_pts(matched_hand, depth_flipped, K_mirrored)
                    if palm_pts is not None:
                        palm_wrist_mm = palm_pts[0]
                        # 3D 腕部位置滤波与速度积分
                        filt_wrist = pts_filter(palm_wrist_mm)
                        if last_wrist[0] is not None and last_t[0] is not None:
                            v_dt = max(1e-4, now - last_t[0])
                            v_cam = (filt_wrist - last_wrist[0]) / v_dt
                            v_base = r_cam_to_base @ v_cam

                            # 档位比例与增益
                            curr_g = gear_configs[current_gear[0]]
                            v_scale = curr_g["lin_scale"]
                            v_norm = float(np.linalg.norm(v_base))
                            if v_norm < teleop_cfg.vision.deadband_vel_mm_s:
                                v_base = np.zeros(3)
                            else:
                                if v_norm > 40.0 and curr_g.get("gain_xyz", 1.0) > 1.0:
                                    g_ratio = min(1.0, (v_norm - 40.0) / 120.0)
                                    dyn_gain = 1.0 + (curr_g["gain_xyz"] - 1.0) * (g_ratio ** 1.2)
                                    v_base = v_base * dyn_gain
                                v_base = v_base * v_scale

                            smooth_v_base[0] = 0.50 * smooth_v_base[0] + 0.50 * v_base
                        last_wrist[0] = filt_wrist
                        last_t[0] = now

                        # 掌面倾角解算姿态摇杆角速度
                        p_wrist = palm_pts[0]
                        p_idx = palm_pts[5]
                        p_mid = palm_pts[9]
                        p_pky = palm_pts[17]
                        v_mid = p_mid - p_wrist
                        v_lat = p_pky - p_idx
                        n_mid = np.linalg.norm(v_mid)
                        n_lat = np.linalg.norm(v_lat)
                        if n_mid > 1e-6 and n_lat > 1e-6:
                            y_dir = v_mid / n_mid
                            z_norm = np.cross(v_lat / n_lat, y_dir)
                            nz = np.linalg.norm(z_norm)
                            if nz > 1e-6:
                                z_norm /= nz
                                x_dir = np.cross(y_dir, z_norm)
                                r_raw = np.column_stack([x_dir, y_dir, z_norm])
                                r_filt = rot_filter(r_raw.reshape(-1)).reshape(3, 3)
                                u_mat, _, vt_mat = np.linalg.svd(r_filt)
                                r_palm = u_mat @ vt_mat

                                # 姿态回零与相对旋转基准 (Wrist Attitude Zero Calibration)
                                if anchor_r_hand[0] is None or not clutch_active[0]:
                                    anchor_r_hand[0] = r_palm.copy()
                                    r_rel = np.eye(3)
                                else:
                                    r_rel = r_palm @ anchor_r_hand[0].T

                                # 提取相对于中立基准的偏角 (度) — 下压为负, 上抬为正
                                roll_deg = float(np.degrees(np.arctan2(r_rel[2, 0], r_rel[2, 2])))
                                pitch_deg = - float(np.degrees(np.arctan2(r_rel[2, 1], r_rel[2, 2])))

                                # 模式角度解算
                                curr_g = gear_configs[current_gear[0]]
                                max_omega_val = curr_g["max_omega"]
                                deadband_deg = teleop_cfg.vision.deadband_angle_deg

                                # 虚拟摇杆速率响应
                                def _joy_rate(ang_deg: float) -> float:
                                    abs_a = abs(ang_deg)
                                    if abs_a <= deadband_deg:
                                        return 0.0
                                    ratio = min(1.0, (abs_a - deadband_deg) / max(1.0, 28.0 - deadband_deg))
                                    return float(np.sign(ang_deg) * (ratio ** 1.4) * max_omega_val)

                                w_cmd = np.zeros(3)
                                if current_mode[0] == MODE_ROLL:
                                    w_cmd[0] = _joy_rate(roll_deg)
                                elif current_mode[0] == MODE_PITCH:
                                    w_cmd[1] = _joy_rate(pitch_deg)
                                elif current_mode[0] == MODE_FULL:
                                    w_cmd[0] = _joy_rate(roll_deg)
                                    w_cmd[1] = _joy_rate(pitch_deg)

                                smooth_w_base[0] = 0.50 * smooth_w_base[0] + 0.50 * (r_cam_to_base @ w_cmd)

                    # ── 3. 灵巧手五指 16DOF 关节角解算 ──
                    if source_mode[0] == 1 and matched_hand.world_landmarks is not None:
                        wpts = np.array([[lm.x, lm.y, lm.z] for lm in matched_hand.world_landmarks], dtype=np.float64)
                        pts = world_smoother(wpts.reshape(-1)).reshape(21, 3)
                        p_frame = _smoothed_frame(pts, frame_smoother)
                        raw_angles = calibrator.map_points(pts, frame=p_frame)
                        hand_bent, _ = finger_id.identify_points(pts)
                    else:
                        # 默认 MediaPipe 伪 3D 模式 (跟手性与握拳最佳)
                        npts = np.array([[lm.x, lm.y, lm.z] for lm in matched_hand.landmarks], dtype=np.float64)
                        pts = pseudo_smoother(npts.reshape(-1)).reshape(21, 3)
                        p_frame = _smoothed_frame(pts, frame_smoother)
                        raw_angles = calibrator.map_points(pts, frame=p_frame)
                        hand_bent, _ = finger_id.identify_points(pts)

                    if hand_clutch_active[0]:
                        hand_angles = hand_angle_filter(raw_angles)
                        # 下发灵巧手舵机目标
                        hand_adapter.set_angles(hand_angles)

            # ── 4. 手部丢失/遮挡平滑缓冲 ──
            if not hand_detected or palm_pts is None:
                loss_count[0] += 1
                if loss_count[0] <= 3 and last_wrist[0] is not None:
                    smooth_v_base[0] = smooth_v_base[0] * 0.90
                    smooth_w_base[0] = smooth_w_base[0] * 0.90
                else:
                    last_wrist[0] = None
                    last_t[0] = None
                    smooth_v_base[0] = np.zeros(3)
                    smooth_w_base[0] = np.zeros(3)
                    pseudo_smoother.reset()
                    world_smoother.reset()
                    frame_smoother.reset()
                    hand_angle_filter.reset()
                    hand_adapter.relax_step(now)
            else:
                loss_count[0] = 0

            # ── 5. 看门狗检测与机械臂下发 ────────────────────────────
            if clutch_active[0]:
                action, wd_scale = watchdog.update(
                    hand_present=hand_detected,
                    hand_confidence=1.0 if hand_detected else 0.0,
                    depth_valid=True if hand_detected else False,
                    wrist_mm=last_wrist[0],
                    now=now,
                )

                if action == WatchdogAction.ESTOP:
                    if arm_adapter.state() != "STOPPED":
                        arm_adapter.e_stop()
                    clutch_active[0] = False
                    scaled_v = np.zeros(3)
                    scaled_w = np.zeros(3)
                elif action != WatchdogAction.STOP and arm_adapter.state() != "STOPPED":
                    scaled_v = smooth_v_base[0] * wd_scale
                    scaled_w = smooth_w_base[0] * wd_scale
                    arm_cmd = CartesianCommand(
                        (float(scaled_v[0]), float(scaled_v[1]), float(scaled_v[2])),
                        (float(scaled_w[0]), float(scaled_w[1]), float(scaled_w[2])),
                        timestamp=now,
                    )
                    arm_adapter.move_cartesian_velocity(arm_cmd)
                else:
                    scaled_v = np.zeros(3)
                    scaled_w = np.zeros(3)
            else:
                # 机械臂暂停离合态: 重置看门狗，下发零速度保持，不误触发急停
                watchdog.reset()
                action = WatchdogAction.OK
                wd_scale = 1.0
                scaled_v = np.zeros(3)
                scaled_w = np.zeros(3)

            # ── 6. 状态轮询与界面渲染 ────────────────────────────────
            if now - last_joint_poll[0] >= 0.08:
                try:
                    cached_joint_state[0] = arm_adapter.get_joint_state()
                except Exception:
                    pass
                last_joint_poll[0] = now

            draw_unified_dashboard(
                frame=frame,
                arm_out={
                    "action": "PAUSED" if not clutch_active[0] else action.name,
                    "v": scaled_v,
                    "w": scaled_w,
                    "d_roll_deg": roll_deg,
                    "d_pitch_deg": pitch_deg,
                    "wrist_px": wrist_px,
                    "wd_scale": wd_scale,
                },
                gear_info=gear_configs.get(current_gear[0], gear_configs[2]),
                clutch_active=clutch_active[0],
                arm_mode=current_mode[0],
                joint_state=cached_joint_state[0],
                no_drive_arm=args.no_drive_arm,
                hand_state_str=("PAUSED (HOLD)" if not hand_clutch_active[0] else hand_adapter.state()) if hand_adapter.is_connected() else "UNPOWERED",
                source_name=SOURCE_NAMES.get(source_mode[0], "PSEUDO-3D"),
                hand_angles=hand_angles,
                hand_bent=hand_bent,
                fps=fps,
            )

            cv2.imshow(WIN_NAME, frame)

            # ── 7. 统一键盘交互分发 ──────────────────────────────────
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q"), 27):  # Q or ESC
                print("[系统] 用户请求退出，安全停机...")
                break
            elif k == ord(" "):
                # SPACE: 机械臂遥操暂停 / 恢复跟随 (Arm Clutch Toggle / Re-arm)
                if arm_adapter.state() == "STOPPED":
                    if hasattr(arm_adapter, "re_arm"):
                        arm_adapter.re_arm(gravity_confirmed=True)
                    watchdog.reset()
                    print("[恢复] 机械臂已解除急停锁定并重新使能 (Re-armed)")
                clutch_active[0] = not clutch_active[0]
                anchor_r_hand[0] = None
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                if clutch_active[0]:
                    watchdog.reset()
                print(f"[机械臂遥操] {'已恢复跟随 (RUN)' if clutch_active[0] else '已暂停锁定 (PAUSE)'}")
            elif k in (ord("l"), ord("L")):
                # L: 灵巧手遥操暂停锁定当前姿态 / 恢复跟随 (Hand Clutch Toggle)
                hand_clutch_active[0] = not hand_clutch_active[0]
                if hand_clutch_active[0]:
                    hand_angle_filter.reset()
                print(f"[灵巧手遥操] {'已恢复跟随 (RUN)' if hand_clutch_active[0] else '已暂停锁定当前姿态 (PAUSE/HOLD)'}")
            elif k in (ord("z"), ord("Z")):
                # Z: 机械臂手腕姿态回零校准
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
                print("\n  *** 机械臂手腕姿态已回零校准 (当前手势设为 0° 中立基准)! ***\n")
            elif k in (ord("w"), ord("W")):
                # W: 视觉看门狗手动复位并重新使能
                watchdog.reset()
                anchor_r_hand[0] = None
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                last_wrist[0] = None
                if arm_adapter.state() == "STOPPED":
                    arm_adapter.arm(gravity_confirmed=True)
                print("\n  *** 视觉看门狗已手动复位并重新使能 (Watchdog Reset & Re-armed)! ***\n")
            elif k in (ord("k"), ord("K")):
                # K: 灵巧手五指全开校准并使能上电
                if hand_detected and 'pts' in locals() and 'p_frame' in locals():
                    calibrator.calibrate_points(pts, frame=p_frame)
                    hand_angle_filter.reset()
                    if not args.no_drive_hand and not hand_adapter.is_connected():
                        print("[灵巧手] 正在连接串口并使能 Dynamixel 舵机...")
                        if hand_adapter.connect():
                            print("  *** 灵巧手已成功连接使能上电，并同步为人手全开姿态! ***")
                        else:
                            print("  [警告] 灵巧手串口连接失败，请检查 USB 串口连接与电源")
                    print("\n  *** 灵巧手五指全开校准完成 (已同步为全开姿态)! ***\n")
                else:
                    print("[提示] 请将手部置于相机视野中完全张开五指后再按 K 进行校准与上电")
            elif k in (ord("p"), ord("P")):
                # P: 灵巧手延迟上电 / 断开切换
                if not args.no_drive_hand and not hand_adapter.is_connected():
                    hand_adapter.connect()
            elif k in (ord("s"), ord("S"), 9):  # TAB or S: 切换机械臂档位
                current_gear[0] = (current_gear[0] % 3) + 1
                anchor_r_hand[0] = None
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                print(f"[档位切换] 机械臂灵敏度: {gear_configs[current_gear[0]]['name']}")
            elif k in (ord("m"), ord("M")):
                current_mode[0] = (current_mode[0] % 4) + 1
                anchor_r_hand[0] = None
                smooth_w_base[0] = np.zeros(3)
                print(f"[模式切换] 机械臂推拿模式: {MODE_NAMES[current_mode[0]]}")
            elif k in (ord("c"), ord("C"), ord("f"), ord("F")):
                clutch_active[0] = not clutch_active[0]
                anchor_r_hand[0] = None
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                print(f"[离合切换] 离合器状态: {'已激活 (CLUTCH ON)' if clutch_active[0] else '已冻结 (FREEZE)'}")
            elif k in (ord("r"), ord("R")):
                print("[姿态] 机械臂安全运动至按摩准备姿态 (READY)，灵巧手张开...")
                clutch_active[0] = False
                anchor_r_hand[0] = None
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                if hasattr(arm_adapter, "re_arm") and arm_adapter.state() == "STOPPED":
                    arm_adapter.re_arm(gravity_confirmed=True)
                arm_adapter.ready()
                hand_adapter.set_open()
                cached_joint_state[0] = arm_adapter.get_joint_state()
            elif k in (ord("h"), ord("H"), ord("o"), ord("O")):
                print("[姿态] 机械臂安全运动至上电初始姿态 (HOME)，灵巧手张开...")
                clutch_active[0] = False
                anchor_r_hand[0] = None
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                if hasattr(arm_adapter, "re_arm") and arm_adapter.state() == "STOPPED":
                    arm_adapter.re_arm(gravity_confirmed=True)
                arm_adapter.home()
                hand_adapter.set_open()
                cached_joint_state[0] = arm_adapter.get_joint_state()

    except KeyboardInterrupt:
        print("\n[系统] 检测到 Ctrl+C 中断信号，正在优雅停机...")
    finally:
        try:
            arm_adapter.disconnect()
        except Exception:
            pass
        try:
            hand_adapter.disconnect()
        except Exception:
            pass
        try:
            tracker.close()
        except Exception:
            pass
        try:
            cam.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[系统] 机械臂与灵巧手已完全安全断开，系统退出完成。")


if __name__ == "__main__":
    main()
