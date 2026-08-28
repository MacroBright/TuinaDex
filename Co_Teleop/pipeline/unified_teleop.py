"""Co_Teleop.pipeline.unified_teleop — 机械臂-灵巧手协同视觉遥操与 LeRobot 多模态产数主控管线.

【核心特性】
1. 单目视觉复用 (Single Perception Stream):
   - 单台 RealSense D455 相机, 单次 MediaPipe 3D 手势检测;
   - 同时解算手腕 3D 宏观轨迹 (驱动 6DOF 机械臂) 与五指 21 关节点微观几何 (驱动 16DOF 灵巧手);
2. 臂-手解耦与控制仲裁 (Decoupled Arm-Hand Kinematics & ControlArbiter):
   - 手腕位移 + 掌面倾角 -> 机械臂空间平移与推拿模态 (点按/滚法/俯仰/全自由);
   - 五指屈伸 -> 灵巧手 16 舵机抓握与揉捏;
   - ControlArbiter 控制租约 (HUMAN_TELEOP / VLA_POLICY) 保证动作源唯一与人类最高抢占权;
3. 统一安全看门狗与运动监管 (Unified Watchdog & MotionSafetySupervisor):
   - 丢帧/遮挡时: 机械臂平稳减速悬停, 灵巧手执行 relax_step 平滑回全开位 (OPEN_POSE);
   - 22 轴软限位、单步最大跳变 (max_dq) 与速度硬限幅;
4. LeRobotDataset v3 与 RawRecorder 双模产数管线:
   - 实时采集 22D 观测、22D 执行目标、工业相机工作区视频与 37 穴位结构化特征;
   - 空格离合自动清洗 (PAUSED 帧不写入训练集), 支持 G 键一键录制/保存, X 键一键放弃;
5. 一体化综合 HUD 监控看板 (Single Unified Dashboard):
   - 1280x720 宽屏 HUD: 手部骨骼 + 虚拟摇杆 + 机械臂 6 轴状态表 + 灵巧手 16 舵机弯曲柱状图 + 录制指示器.
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
for p in (WORKSPACE_ROOT, WORKSPACE_ROOT / "Arm-robot_VLA", WORKSPACE_ROOT / "Leap_Hand" / "python", WORKSPACE_ROOT / "Co_Teleop", WORKSPACE_ROOT / "packages" / "lerobot_robot_tuinadex"):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Co_Teleop 模块导入
from Co_Teleop.adapters import (
    ArmAdapter,
    CartesianCommand,
    EEPose,
    JointState,
    LatestStateCache,
    LeapHandAdapter,
    NoDriveArmAdapter,
    NoDriveHandAdapter,
    RealArmAdapter,
    SimArmAdapter,
)
from Co_Teleop.config import (
    TeleopConfig,
    build_gear_configs,
)
from Co_Teleop.calibration import load_calib
from Co_Teleop.gui import (
    FINGER_NAMES,
    MODE_FULL,
    MODE_KNEAD,
    MODE_NAMES,
    MODE_PITCH,
    MODE_ROLL,
    draw_unified_dashboard,
)
from Co_Teleop.perception.acupoint_adapter import AcuPointAdapter
from Co_Teleop.perception.camera import CameraConfig, IndustrialCamera
from Co_Teleop.pipeline.observation_aggregator import ObservationAggregator
from Co_Teleop.recording import (
    ArmObservation,
    FrameValidity,
    HandObservation,
    LeRobotDatasetWriter,
    RawRecorder,
    TaskMetadata,
    TeleopFrame,
)
from Co_Teleop.safety import (
    ControlArbiter,
    ControlSource,
    MotionSafetySupervisor,
    VisionWatchdog,
    WatchdogAction,
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

    ap = argparse.ArgumentParser(description="真机 6DOF 机械臂 + 16DOF 灵巧手视觉遥操与 LeRobot 产数统一管线")
    ap.add_argument("--iface", default="can0", help="SocketCAN 接口 (机械臂)")
    ap.add_argument("--hand-port", default=None, help="灵巧手串口路径 (默认从配置文件读取或自动扫描)")
    ap.add_argument("--config", default=str(default_cfg), help="集中配置文件路径 (默认 teleop_config.yaml)")
    ap.add_argument("--calib", default=str(default_calib), help="手眼标定矩阵路径")
    ap.add_argument("-y", "--gravity-confirm", action="store_true", help="确认重力关节 J2/J3 (机械臂真机必须)")
    ap.add_argument("--no-drive-arm", action="store_true", help="机械臂空跑测试 (不连 CAN 总线)")
    ap.add_argument("--no-drive-hand", action="store_true", help="灵巧手空跑测试 (不连 Dynamixel 串口)")
    ap.add_argument("--mode", choices=["knead", "roll", "pitch", "full"], default="roll",
                    help="推拿遥操姿态模式: knead, roll, pitch, full")
    ap.add_argument("--record", action="store_true", help="启用 LeRobotDataset v3 与 RawRecorder 产数录制")
    ap.add_argument("--dataset-root", default="./datasets/lerobot_tuina", help="LeRobot 数据集存储根目录")
    ap.add_argument("--mock-overhead", action="store_true", help="启用推拿工作区工业相机 Mock 模式")
    args = ap.parse_args()

    if not args.gravity_confirm and not args.no_drive_arm:
        sys.exit("遥操前必须 -y/--gravity-confirm 确认重力关节 (J2/J3) (空跑测试请加 --no-drive-arm)")

    # 1. 载入并验证全局集中配置 (起飞前安全检查)
    teleop_cfg = TeleopConfig.load(args.config)
    gear_configs = build_gear_configs(teleop_cfg)

    # 2. 机械臂与灵巧手适配器初始化
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
                                       max_joint_acc_deg_s2=teleop_cfg.motor.max_joint_acc_deg_s2,
                                       max_dq_deg=teleop_cfg.motor.max_dq_deg))
        arm_adapter = RealArmAdapter(ctrl=ctrl, ready_pose=teleop_cfg.pose.ready_pose_deg,
                                     home_pose=teleop_cfg.pose.home_pose_deg)

    if not arm_adapter.connect():
        sys.exit(f"机械臂连接失败 (接口: {args.iface})")

    # 显式上电使能门禁
    if not args.no_drive_arm:
        arm_adapter.arm(gravity_confirmed=True)

    hand_port = args.hand_port or teleop_cfg.motor.hand_serial_port
    if args.no_drive_hand:
        print("[灵巧手] 启动 NoDrive 虚拟仿真测试模式...")
        hand_adapter = NoDriveHandAdapter()
    else:
        print(f"[灵巧手] 正在连接 Dynamixel 串口 ({hand_port})...")
        hand_adapter = LeapHandAdapter(port=hand_port)

    # 3. 数据契约、状态缓存、聚合器与录制器初始化
    cache = LatestStateCache()
    aggregator = ObservationAggregator(cache)
    arbiter = ControlArbiter()
    supervisor = MotionSafetySupervisor()

    overhead_cfg = CameraConfig(camera_id="overhead_cam", width=640, height=480, fps=30)
    overhead_cam = IndustrialCamera(overhead_cfg, mock_mode=args.mock_overhead or args.no_drive_arm)
    overhead_cam.start_stream()

    acupoint_adapter = AcuPointAdapter(mock_mode=True)
    acupoint_adapter.initialize()

    lerobot_writer = LeRobotDatasetWriter(
        repo_id="tuina_massage_22dof",
        fps=30,
        root=args.dataset_root,
    )
    if args.record:
        lerobot_writer.initialize()

    raw_recorder = RawRecorder()
    is_recording = [False]
    episode_idx = [1]
    recorded_frames = [0]

    TECHNIQUES = {
        1: ("按揉大椎穴", "kneading", "dazhui"),
        2: ("滚法肩井穴", "rolling", "jianjing"),
        3: ("点按肺俞穴", "point_press", "feishu"),
        4: ("抚摩肾俞穴", "stroking", "shenshu"),
    }
    current_tech_idx = [1]

    # 4. 手眼标定与视觉跟踪器
    calib_mat = load_calib(args.calib)
    print(f"[标定] 成功载入手眼标定矩阵 R_cam->base (来自: {args.calib})")

    watchdog = VisionWatchdog()
    calibrator = Calibrator()
    joint_mapper = JointMapper()
    finger_id = FingerIdentifier()
    pseudo_smoother = OneEuroFilter(freq=30.0, mincutoff=1.5, beta=0.08)
    world_smoother = OneEuroFilter(freq=30.0, mincutoff=1.5, beta=0.08)
    frame_smoother = OneEuroFilter(freq=30.0, mincutoff=2.0, beta=0.10)
    hand_angle_filter = OneEuroFilter(freq=30.0, mincutoff=1.5, beta=0.05)

    cam = open_realsense(width=1280, height=720, fps=30)
    tracker = HandTracker(max_hands=1)
    hamer = HaMeR3D()

    WIN_NAME = "TuinaDex 22-DOF LeRobot Teleoperation & Data Acquisition Pipeline"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    current_gear = [teleop_cfg.gear.default_gear]
    current_mode = [MODE_MAP.get(args.mode, MODE_ROLL)]
    clutch_active = [False]
    hand_clutch_active = [True]
    source_mode = [2]  # 默认 MP 3D

    last_wrist = [None]
    last_t = [None]
    anchor_r_hand = [None]
    smooth_v_base = [np.zeros(3)]
    smooth_w_base = [np.zeros(3)]
    loss_count = [0]
    last_joint_poll = [0.0]
    cached_joint_state = [None]

    t_prev = time.monotonic()
    fps_smooth = 30.0

    print("\n" + "=" * 70)
    print("  TuinaDex 22-DOF LeRobot 视觉遥操与多模态产数主控就绪!")
    print("  - 空格键 [SPACE]: 启动/暂停机械臂跟随 (PAUSED 时自动过滤录制)")
    print("  - [G] 键: 开始 / 完成录制当前 Episode 并保存为 LeRobotDataset v3")
    print("  - [X] 键: 放弃并丢弃当前未完成 Episode")
    print("  - [1~4] 键: 切换推拿手法 (1:按揉大椎, 2:滚法肩井, 3:点穴肺俞, 4:抚摩肾俞)")
    print("  - [R] 键: 回按摩准备位 (READY) | [H] 键: 回上电初始位 (HOME)")
    print("=" * 70 + "\n")

    try:
        while True:
            t_now = time.monotonic()
            dt_loop = max(1e-4, t_now - t_prev)
            t_prev = t_now
            fps = 0.9 * fps_smooth + 0.1 * (1.0 / dt_loop)
            fps_smooth = fps

            color_bgr, depth_img = cam.read()
            if color_bgr is None:
                continue

            frame = color_bgr.copy()
            now = time.monotonic()

            # ── 1. 操作员手部检测与追踪 ───────────────────────────────
            res = tracker.process(color_bgr)
            hand_detected = False
            wrist_px = None
            hand_angles = None
            hand_bent = None
            roll_deg, pitch_deg = 0.0, 0.0

            if res.multi_hand_landmarks:
                hand_detected = True
                loss_count[0] = 0
                lm = res.multi_hand_landmarks[0]
                pts = np.array([[p.x * 1280, p.y * 720, p.z * 1280] for p in lm.landmark], dtype=np.float64)
                wrist_px = (int(pts[0, 0]), int(pts[0, 1]))

                angles_16, bent = joint_mapper.map_hand(pts)
                hand_angles = hand_angle_filter(angles_16)
                hand_bent = bent

                if hand_clutch_active[0]:
                    if hasattr(hand_adapter, "set_angles"):
                        hand_adapter.set_angles(hand_angles)
                    elif hasattr(hand_adapter, "send_angles"):
                        hand_adapter.send_angles(hand_angles)

                p_pts = build_palm_pts(pts)
                p_frame = _smoothed_frame(p_pts, frame_smoother)
                w_pos = p_frame[0]

                if last_wrist[0] is not None and last_t[0] is not None:
                    dt_w = max(1e-4, now - last_t[0])
                    v_cam = (w_pos - last_wrist[0]) / dt_w
                    v_base = calib_mat @ v_cam

                    g_cfg = gear_configs.get(current_gear[0], gear_configs[2])
                    lin_scale = g_cfg["lin_scale"]
                    v_scaled = v_base * lin_scale

                    smooth_v_base[0] = 0.6 * smooth_v_base[0] + 0.4 * v_scaled
                last_wrist[0] = w_pos
                last_t[0] = now

                if anchor_r_hand[0] is None:
                    anchor_r_hand[0] = p_frame[1].copy()

                d_norm = p_frame[1] - anchor_r_hand[0]
                roll_deg = float(np.clip(d_norm[0] * 45.0, -45.0, 45.0))
                pitch_deg = float(np.clip(d_norm[1] * 45.0, -45.0, 45.0))

                w_cmd = np.array([np.radians(pitch_deg) * 0.2, np.radians(roll_deg) * 0.2, 0.0])
                smooth_w_base[0] = 0.6 * smooth_w_base[0] + 0.4 * w_cmd
            else:
                loss_count[0] += 1
                if loss_count[0] > 10:
                    smooth_v_base[0] *= 0.8
                    smooth_w_base[0] *= 0.8
                    if not hand_clutch_active[0] and hasattr(hand_adapter, "relax_step"):
                        hand_adapter.relax_step(now)

            # ── 2. 运动控制仲裁与下发 ────────────────────────────────
            scaled_v = np.zeros(3)
            scaled_w = np.zeros(3)

            if clutch_active[0] and hand_detected:
                arbiter.request_lease(ControlSource.HUMAN_TELEOP, now=now)

                scaled_v = smooth_v_base[0]
                scaled_w = smooth_w_base[0]
                arm_cmd = CartesianCommand(
                    (float(scaled_v[0]), float(scaled_v[1]), float(scaled_v[2])),
                    (float(scaled_w[0]), float(scaled_w[1]), float(scaled_w[2])),
                    timestamp=now,
                )
                arm_adapter.move_cartesian_velocity(arm_cmd)
            else:
                scaled_v = np.zeros(3)
                scaled_w = np.zeros(3)

            # ── 3. 状态轮询、环境感知与快照聚合 ──────────────────────
            cur_q_deg = arm_adapter.current_q_deg if hasattr(arm_adapter, "current_q_deg") else np.zeros(6)
            cur_q_rad = np.radians(cur_q_deg).astype(np.float32)
            cur_dq = np.zeros(6, dtype=np.float32)
            cur_curr = np.zeros(6, dtype=np.float32)

            arm_obs = ArmObservation(q=cur_q_rad, dq=cur_dq, current=cur_curr, timestamp=now)
            cache.update_arm(arm_obs)

            if hasattr(hand_adapter, "get_current_angles"):
                h_q = np.asarray(hand_adapter.get_current_angles(), dtype=np.float32)
            elif hasattr(hand_adapter, "curr_angles"):
                h_q = np.asarray(hand_adapter.curr_angles, dtype=np.float32)
            else:
                h_q = np.asarray(hand_angles if hand_angles is not None else np.zeros(16), dtype=np.float32)

            hand_obs = HandObservation(q=h_q, currents=np.zeros(16, dtype=np.float32), timestamp=now)
            cache.update_hand(hand_obs)

            oh_frame = overhead_cam.get_latest_frame()
            if oh_frame is not None:
                cache.update_camera(oh_frame)
                acu_obs = acupoint_adapter.detect(oh_frame.image, timestamp=oh_frame.timestamp)
                if acu_obs is not None:
                    cache.update_acupoints(acu_obs)

            act_arm_rad = cur_q_rad.copy()
            act_hand_rad = h_q.copy()
            act_22d = np.concatenate([act_arm_rad, act_hand_rad]).astype(np.float32)
            cart_cmd_6d = np.concatenate([scaled_v, scaled_w]).astype(np.float32)

            aggregator.set_pause(not clutch_active[0])
            aggregator.set_hand_locked(not hand_clutch_active[0])

            t_info = TECHNIQUES[current_tech_idx[0]]
            task_meta = TaskMetadata(
                task_name=t_info[0],
                technique=t_info[1],
                target_acupoint=t_info[2],
                episode_id=f"ep_{episode_idx[0]:03d}",
                step_idx=recorded_frames[0],
            )

            teleop_frame = aggregator.aggregate(
                action_22d=act_22d,
                cartesian_cmd=cart_cmd_6d,
                task=task_meta,
                now=now,
            )

            # ── 4. 数据录制写入 ──────────────────────────────────────
            if is_recording[0]:
                accepted = lerobot_writer.add_frame(teleop_frame)
                raw_recorder.record_frame(teleop_frame)
                if accepted:
                    recorded_frames[0] += 1

            # ── 5. 渲染 HUD 看板 ─────────────────────────────────────
            if now - last_joint_poll[0] >= 0.08:
                try:
                    cached_joint_state[0] = arm_adapter.get_joint_state()
                except Exception:
                    pass
                last_joint_poll[0] = now

            lease_str = arbiter.get_active_source(now=now).value
            rec_info = {
                "is_recording": is_recording[0],
                "episode_idx": episode_idx[0],
                "frames": recorded_frames[0],
                "task_name": f"{t_info[0]} ({t_info[2]})",
            }

            draw_unified_dashboard(
                frame=frame,
                arm_out={
                    "action": "PAUSED" if not clutch_active[0] else "RUN",
                    "v": scaled_v,
                    "w": scaled_w,
                    "d_roll_deg": roll_deg,
                    "d_pitch_deg": pitch_deg,
                    "wrist_px": wrist_px,
                    "wd_scale": 1.0,
                },
                gear_info=gear_configs.get(current_gear[0], gear_configs[2]),
                clutch_active=clutch_active[0],
                arm_mode=current_mode[0],
                joint_state=cached_joint_state[0],
                no_drive_arm=args.no_drive_arm,
                hand_state_str=("PAUSED" if not hand_clutch_active[0] else (hand_adapter.state() if hasattr(hand_adapter, "state") else "RUN")) if hand_adapter.is_connected() else "UNPOWERED",
                source_name=SOURCE_NAMES.get(source_mode[0], "PSEUDO-3D"),
                hand_angles=hand_angles,
                hand_bent=hand_bent,
                fps=fps,
                recording_info=rec_info,
                lease_info="HUMAN" if lease_str == "HUMAN_TELEOP" else lease_str,
            )

            cv2.imshow(WIN_NAME, frame)

            # ── 6. 统一键盘交互分发 ──────────────────────────────────
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q"), 27):
                print("[系统] 用户请求退出，正在安全停机...")
                break
            elif k in (ord("g"), ord("G")):
                if not is_recording[0]:
                    print(f"\n>>> [REC] 开始录制 Episode #{episode_idx[0]} ({t_info[0]} - {t_info[2]}) <<<")
                    lerobot_writer.start_episode()
                    raw_recorder.start_episode(f"ep_{episode_idx[0]:03d}")
                    is_recording[0] = True
                    recorded_frames[0] = 0
                else:
                    print(f"\n<<< [SAVE] 正在保存 Episode #{episode_idx[0]} (共 {recorded_frames[0]} 帧)... >>>")
                    lerobot_writer.save_episode()
                    raw_recorder.save_episode()
                    is_recording[0] = False
                    print(f"  *** Episode #{episode_idx[0]} 成功保存并写入 LeRobotDataset v3! ***\n")
                    episode_idx[0] += 1
                    recorded_frames[0] = 0
            elif k in (ord("x"), ord("X")):
                if is_recording[0]:
                    print(f"\n[DISCARD] 正在放弃并丢弃 Episode #{episode_idx[0]} 缓存...")
                    lerobot_writer.clear_episode_buffer()
                    raw_recorder.discard_episode()
                    is_recording[0] = False
                    recorded_frames[0] = 0
                    print("  *** 当前 Episode 已安全丢弃，不污染数据集 ***\n")
            elif k in (ord("1"), ord("2"), ord("3"), ord("4")):
                idx = int(chr(k))
                current_tech_idx[0] = idx
                print(f"[手法切换] 当前推拿任务: {TECHNIQUES[idx][0]} (目标穴位: {TECHNIQUES[idx][2]})")
            elif k == ord(" "):
                clutch_active[0] = not clutch_active[0]
                anchor_r_hand[0] = None
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                if clutch_active[0]:
                    watchdog.reset()
                print(f"[机械臂遥操] {'已恢复跟随 (RUN)' if clutch_active[0] else '已暂停锁定 (PAUSE)'}")
            elif k in (ord("l"), ord("L")):
                hand_clutch_active[0] = not hand_clutch_active[0]
                if hand_clutch_active[0]:
                    hand_angle_filter.reset()
                print(f"[灵巧手遥操] {'已恢复跟随 (RUN)' if hand_clutch_active[0] else '已暂停锁定当前姿态 (PAUSE/HOLD)'}")
            elif k in (ord("s"), ord("S"), 9):
                current_gear[0] = (current_gear[0] % 3) + 1
                anchor_r_hand[0] = None
                smooth_v_base[0] = np.zeros(3)
                smooth_w_base[0] = np.zeros(3)
                print(f"[档位切换] 机械臂灵敏度: {gear_configs[current_gear[0]]['name']}")
            elif k in (ord("r"), ord("R")):
                print("[姿态] 机械臂平稳运动至准备姿态 (READY)...")
                clutch_active[0] = False
                arm_adapter.ready()
                hand_adapter.set_open()
            elif k in (ord("h"), ord("H")):
                print("[姿态] 机械臂平稳运动至初始姿态 (HOME)...")
                clutch_active[0] = False
                arm_adapter.home()
                hand_adapter.set_open()

    except KeyboardInterrupt:
        print("\n[系统] 收到 Ctrl+C 中断信号，正在安全退出...")
    finally:
        overhead_cam.stop_stream()
        overhead_cam.disconnect()
        if is_recording[0]:
            lerobot_writer.clear_episode_buffer()
            raw_recorder.discard_episode()
        lerobot_writer.finalize()
        arm_adapter.disconnect()
        hand_adapter.disconnect()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[系统] 硬件已断开，产数主控退出完成。")


if __name__ == "__main__":
    main()
