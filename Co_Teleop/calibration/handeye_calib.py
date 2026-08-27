"""手眼标定：相机系→机器人基座系的 3D 正交旋转变换 R (6DOF 完整标定向导 & 实时沙盒).

标定流程:
  1. 3 轴平移标定: 右方平移(-Y)、前方推移(+X)、上方抬高(+Z)
  2. 2 轴旋转标定: 右翻掌(Roll)、手掌下扣(Pitch)
  3. Procrustes SVD 正交求解最优 SO(3) 旋转矩阵 R
  4. 实时 6DOF 沙盒验证 (Live Sandbox): 实时直观测试 6 方向运动映射
  5. 确认无误后按 Y / SPACE 写入 handeye_calib.json
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Union

import numpy as np

_Path = Union[str, Path]


def rot_from_euler(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """绕 X→Y→Z（相机系）的旋转矩阵 R(3,3)。列向量应用: v_base = R @ v_cam。"""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def procrustes_rotation(src_pts, dst_pts) -> np.ndarray:
    """最小化 Σ||R@p_i − q_i||² 的正交旋转矩阵 R (det=1). src/dst: (N,3). 返回 R(3,3)."""
    src = np.asarray(src_pts, float).T   # (3,N)
    dst = np.asarray(dst_pts, float).T   # (3,N)
    H = src @ dst.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def apply_rotation(R: np.ndarray, pts) -> np.ndarray:
    """R(3,3) 作用于 (N,3) 点集（每行一个列向量）。"""
    pts = np.asarray(pts, float)
    return (R @ pts.T).T


def save_calib(path: _Path, R: np.ndarray) -> None:
    Path(path).write_text(json.dumps({"R": np.asarray(R).tolist()}))


def load_calib(path: _Path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    return np.array(data["R"])


# 基座方向码: 1=+X(前) 2=-X(后) 3=+Y(左) 4=-Y(右) 5=+Z(上) 6=-Z(下)
_BASE_DIR_CODES = {
    1: np.array([1.0, 0.0, 0.0]),   # +X (机械臂前方/推拿床方向)
    2: np.array([-1.0, 0.0, 0.0]),  # -X (机械臂后方/立柱方向)
    3: np.array([0.0, 1.0, 0.0]),   # +Y (机械臂左方)
    4: np.array([0.0, -1.0, 0.0]),  # -Y (机械臂右方)
    5: np.array([0.0, 0.0, 1.0]),   # +Z (机械臂上方/抬高)
    6: np.array([0.0, 0.0, -1.0]),  # -Z (机械臂下方/下压)
}


def solve_handeye(cam_dirs, base_codes):
    """从 (相机系单位方向, 基座方向码) 配对解手眼旋转 R."""
    src = np.asarray(cam_dirs, float)
    dst = np.array([_BASE_DIR_CODES[c] for c in base_codes], float)
    return procrustes_rotation(src, dst)


def main():
    import cv2

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Leap_Hand" / "python"))
    from gesture_mapping.camera import open_realsense
    from gesture_mapping.filter import OneEuroFilter
    from gesture_mapping.hand_tracker import HandTracker
    from gesture_mapping.wrist_tracker import build_palm_pts

    ap = argparse.ArgumentParser(description="TuinaDex 6DOF 视觉手眼交互标定向导 & 实时沙盒")
    ap.add_argument("--out", default=str(Path(__file__).parent / "handeye_calib.json"),
                    help="标定输出文件路径")
    ap.add_argument("--sandbox", action="store_true",
                    help="直接加载现有标定矩阵，进入 6DOF 实时沙盒挥手测试")
    args = ap.parse_args()

    calib_out_path = Path(args.out)
    solved_R = None
    in_sandbox = False

    if args.sandbox:
        if calib_out_path.exists():
            solved_R = load_calib(calib_out_path)
            in_sandbox = True
            print(f"[沙盒模式] 已加载现有标定矩阵: {calib_out_path}")
        else:
            print(f"[提示] 未找到现有标定文件 {calib_out_path}，将先运行 5 步标定向导。")

    print("=" * 70)
    print("【TuinaDex 6DOF 视觉手眼标定向导】")
    print("本向导通过 5 步简易手势采样，求解 RealSense D455 相机系到机械臂基座系的正交旋转矩阵 R。")
    print("=" * 70)

    cam = open_realsense()
    if cam is None:
        sys.exit("错误: 未检测到 RealSense 相机，请检查 USB 3.0 连接。")

    tracker = HandTracker(max_num_hands=1)
    pts_filter = OneEuroFilter(n_joints=3, min_cutoff=0.4, beta=0.005)
    rot_filter = OneEuroFilter(n_joints=9, min_cutoff=0.3, beta=0.003)

    win_name = "TuinaDex 6DOF Hand-Eye Calibration & Sandbox"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1024, 768)

    # 5 步标定定义:
    # (步骤名称, 详细中文动作指引, 机械臂末端对应运动, 基座方向码, 模式)
    steps = [
        ("Step 1/5 [左右平移 (水平横向)]", "请将手掌在水平面上【向右平移】约 15~20cm", "机械臂末端向【右方】平移 (-Y)", 4, "lin"),
        ("Step 2/5 [前后推拉 (水平纵向)]", "请将手掌在水平面上【向前推移】约 15~20cm (远离身体向前推)", "机械臂末端向【前方】延伸 (+X)", 1, "lin"),
        ("Step 3/5 [上下升降 (垂直高度)]", "请将手掌垂直【向上抬高/远离相机】约 15~20cm (向天花板抬手)", "机械臂末端向【上方】抬起 (+Z)", 5, "lin"),
        ("Step 4/5 [旋转: 左右翻掌]", "请将手掌向【右侧倾斜翻掌/顺时针翻腕】约 25°", "机械臂末端向【右顺时针滚转】(Roll)", 1, "rot"),
        ("Step 5/5 [旋转: 手掌下扣]", "请将手腕向【下弯曲/手掌向下扣】约 25°", "机械臂末端向【下低头点头】(Pitch)", 4, "rot"),
    ]

    current_step = 0
    calib_cam_dirs = []
    calib_task_codes = []
    history_pts = []
    history_rot = []
    is_recording = False

    # 沙盒实时物理量跟踪
    last_wrist = [None]
    last_t = [None]
    anchor_r_hand = [None]
    smooth_v_base = np.zeros(3)
    sandbox_mode = [1]

    try:
        while True:
            ok, bgr, depth, K = cam.read_with_depth()
            if not ok or bgr is None:
                continue

            hands = tracker.detect(bgr)
            wrist_cam = None
            r_hand = None
            px_coord = None
            if hands:
                bgr = tracker.draw_landmarks(bgr, hands)
                pts = build_palm_pts(hands[0], depth, K)
                if pts is not None:
                    wrist_raw = pts[0]
                    wrist_cam = pts_filter(wrist_raw)

                    # 姿态提取: 严格使用解剖学刚体掌骨基底 0-5-17 (基于 MediaPipe 3D World Landmarks，免疫手指弯曲与像素深度边缘噪声)
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

                    # 李群旋转矩阵低通平滑 + SVD 严格正交重整化
                    r_filtered = rot_filter(r_raw.flatten()).reshape(3, 3)
                    u_svd, _, vt_svd = np.linalg.svd(r_filtered)
                    r_hand = u_svd @ vt_svd

                    px_coord = (int(hands[0].landmarks[0].x * bgr.shape[1]),
                                int(hands[0].landmarks[0].y * bgr.shape[0]))

            h, w = bgr.shape[:2]
            now = time.monotonic()

            # -------------------------------------------------------------
            # 模式 A: 5 步交互式引导采样
            # -------------------------------------------------------------
            if not in_sandbox:
                step_title, action_guide, robot_guide, task_code, mode_type = steps[current_step]

                # 顶部信息横幅
                cv2.rectangle(bgr, (0, 0), (w, 80), (25, 25, 32), -1)
                cv2.putText(bgr, f"{step_title}", (18, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 230, 255), 2)
                cv2.putText(bgr, f"人手动作: {action_guide}", (18, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                cv2.putText(bgr, f"映射目标: {robot_guide}", (18, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 120), 1)

                # 底部操作提示
                cv2.rectangle(bgr, (0, h - 50), (w, h), (20, 20, 25), -1)
                if not is_recording:
                    hint_txt = "按 [SPACE 空格键] 开始记录手势轨迹 | [Q/ESC] 退出"
                    cv2.putText(bgr, hint_txt, (18, h - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 2)
                else:
                    samples_cnt = len(history_pts) if mode_type == "lin" else len(history_rot)
                    hint_txt = f"● 正在录制手势中... 采样帧数: {samples_cnt} | 完成后再次按 [SPACE] 确认"
                    cv2.putText(bgr, hint_txt, (18, h - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0), 2)

                    if mode_type == "lin" and wrist_cam is not None:
                        history_pts.append(wrist_cam)
                    elif mode_type == "rot" and r_hand is not None:
                        history_rot.append(r_hand)

                # 绘制轨迹拖尾点
                if is_recording and len(history_pts) > 1:
                    for i in range(1, len(history_pts)):
                        p1 = tuple(np.int32(history_pts[i - 1][:2]))
                        p2 = tuple(np.int32(history_pts[i][:2]))
                        cv2.line(bgr, p1, p2, (0, 255, 255), 2)

            # -------------------------------------------------------------
            # 模式 B: 实时 6DOF 沙盒验证 (Live Sandbox Preview)
            # -------------------------------------------------------------
            else:
                cv2.rectangle(bgr, (0, 0), (w, 90), (18, 35, 18), -1)
                cv2.putText(bgr, "[实时 6DOF 运动沙盒验证 Sandbox Mode]", (18, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 0), 2)
                cv2.putText(bgr, "请自由向各个方向挥手/翻腕: [Y/SPACE] 保存标定 | [C] 姿态零位重置 | [R] 重做标定 | [Q] 退出",
                            (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

                v_b = np.zeros(3)
                if wrist_cam is not None and last_wrist[0] is not None and last_t[0] is not None:
                    dt = now - last_t[0]
                    if 0.001 < dt < 0.5:
                        v_cam = (wrist_cam - last_wrist[0]) / dt
                        v_b_raw = solved_R @ v_cam
                        # 1. 10 mm/s 单轴灵敏死区门限 (滤除生理微颤，提升起步灵敏度)
                        v_b_clamped = np.zeros(3)
                        for i in range(3):
                            if abs(v_b_raw[i]) > 10.0:
                                v_b_clamped[i] = v_b_raw[i]
                        # 2. 跨轴正交抑制 (Cross-Axis Rejection): 主轴明显移动时，次轴低于 25% 视为微震耦合并清零
                        max_axis_val = float(np.max(np.abs(v_b_clamped)))
                        if max_axis_val > 18.0:
                            for i in range(3):
                                if abs(v_b_clamped[i]) < 0.25 * max_axis_val:
                                    v_b_clamped[i] = 0.0
                        raw_speed = float(np.linalg.norm(v_b_clamped))
                        alpha = 0.30 + 0.25 * min(1.0, max(0.0, (raw_speed - 12.0) / 50.0))
                        smooth_v_base = alpha * v_b_clamped + (1.0 - alpha) * smooth_v_base
                        spd_filt = float(np.linalg.norm(smooth_v_base))
                        if spd_filt > 10.0:
                            spd_r = min(1.0, (spd_filt - 10.0) / 60.0)
                            v_b = smooth_v_base * (1.0 + 1.2 * (spd_r ** 1.3))
                        else:
                            v_b = smooth_v_base

                last_wrist[0] = wrist_cam
                last_t[0] = now

                # 姿态旋转解算 (绝对姿态跟随 + 推拿模态解耦)
                d_roll_raw, d_pitch_raw = 0.0, 0.0
                d_roll, d_pitch = 0.0, 0.0
                if r_hand is not None:
                    if anchor_r_hand[0] is None:
                        anchor_r_hand[0] = r_hand.copy()
                    else:
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
                        theta_b = solved_R @ theta_cam

                        # 虚拟手势摇杆速率响应 (倾斜持续转动，回平死区保持锁定)
                        def _joy(angle_deg: float, deadband: float = 5.0, max_ang: float = 28.0, max_w: float = 3.0) -> float:
                            abs_ang = abs(angle_deg)
                            if abs_ang <= deadband:
                                return 0.0
                            r = min(1.0, (abs_ang - deadband) / max(1.0, max_ang - deadband))
                            return float(np.sign(angle_deg) * (r ** 1.4) * max_w)

                        hand_pitch = float(np.degrees(theta_b[0]))
                        hand_roll = float(np.degrees(theta_b[1]))
                        w_pitch_raw = _joy(hand_pitch, deadband=5.0, max_ang=22.0, max_w=3.0)
                        w_roll_raw = _joy(hand_roll, deadband=5.0, max_ang=28.0, max_w=3.0)

                        # 模态过滤
                        if sandbox_mode[0] == 1:
                            # 模态 1: 垂直点按揉捏 (完全锁定)
                            w_roll, w_pitch = 0.0, 0.0
                        elif sandbox_mode[0] == 2:
                            # 模态 2: 滚法推法 (仅开放 Roll, 锁 Pitch)
                            w_roll, w_pitch = w_roll_raw, 0.0
                        elif sandbox_mode[0] == 3:
                            # 模态 3: 俯仰调节 (仅开放 Pitch, 锁 Roll)
                            w_roll, w_pitch = 0.0, w_pitch_raw
                        else:
                            # 模态 4: 全 6-DOF
                            w_roll, w_pitch = w_roll_raw, w_pitch_raw

                # 中文动作映射实时解析
                mode_names = {1: "1.点按揉捏(锁定)", 2: "2.滚法(单轴Roll)", 3: "3.俯仰调节(单轴Pitch)", 4: "4.全6DOF(全姿态)"}
                x_tag = "【+X 向左平移】" if v_b[0] > 10 else ("【-X 向右平移】" if v_b[0] < -10 else "静止")
                y_tag = "【-Y 前进延伸】" if v_b[1] < -10 else ("【+Y 后退收缩】" if v_b[1] > 10 else "静止")
                z_tag = "【+Z 向上抬高】" if v_b[2] > 10 else ("【-Z 向下压低】" if v_b[2] < -10 else "静止")
                roll_tag = "【向左持续滚转】" if w_roll > 0.05 else ("【向右持续滚转】" if w_roll < -0.05 else "回平保持 (死区内)")
                pitch_tag = "【向下持续低头】" if w_pitch < -0.05 else ("【向上持续抬头】" if w_pitch > 0.05 else "回平保持 (死区内)")
                if sandbox_mode[0] == 1:
                    roll_tag, pitch_tag = "【姿态锁定】", "【姿态锁定】"
                elif sandbox_mode[0] == 2:
                    pitch_tag = "【姿态锁定】"
                elif sandbox_mode[0] == 3:
                    roll_tag = "【姿态锁定】"

                # 左侧数据面板
                cv2.rectangle(bgr, (15, 95), (460, 275), (20, 20, 28), -1)
                cv2.rectangle(bgr, (15, 95), (460, 275), (60, 60, 80), 1)

                cv2.putText(bgr, f"推拿模态: {mode_names[sandbox_mode[0]]} (按M切换)", (25, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1)
                cv2.putText(bgr, f"平移 X(左右): {v_b[0]:+5.1f} mm/s -> {x_tag}", (25, 148),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1)
                cv2.putText(bgr, f"平移 Y(前后): {v_b[1]:+5.1f} mm/s -> {y_tag}", (25, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
                cv2.putText(bgr, f"平移 Z(上下): {v_b[2]:+5.1f} mm/s -> {z_tag}", (25, 202),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 0), 1)
                cv2.putText(bgr, f"摇杆 Roll : {w_roll:+5.2f} rad/s -> {roll_tag}", (25, 232),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 255), 1)
                cv2.putText(bgr, f"摇杆 Pitch: {w_pitch:+5.2f} rad/s -> {pitch_tag}", (25, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 255), 1)

                # 手腕圆环与牵引线
                if px_coord is not None:
                    u, v = px_coord
                    cv2.circle(bgr, (u, v), 10, (0, 255, 0), -1)
                    cv2.circle(bgr, (u, v), 14, (255, 255, 255), 2)
                    dx = int(np.clip(v_b[1] * -1.5, -80, 80))
                    dy = int(np.clip(v_b[0] * -1.5, -80, 80))
                    if abs(dx) > 4 or abs(dy) > 4:
                        cv2.arrowedLine(bgr, (u, v), (u + dx, v + dy), (255, 50, 50), 3, tipLength=0.3)

            cv2.imshow(win_name, bgr)
            k = cv2.waitKey(1) & 0xFF

            if k in (ord("q"), ord("Q"), 27):
                print("\n[退出] 标定向导已安全退出。")
                return

            if in_sandbox:
                if k in (ord("m"), ord("M")):
                    sandbox_mode[0] = (sandbox_mode[0] % 4) + 1
                    anchor_r_hand[0] = None
                    print(f"[沙盒] 切换推拿姿态模式: {mode_names[sandbox_mode[0]]}")
                elif k == ord("1"):
                    sandbox_mode[0] = 1
                    anchor_r_hand[0] = None
                elif k == ord("2"):
                    sandbox_mode[0] = 2
                    anchor_r_hand[0] = None
                elif k == ord("3"):
                    sandbox_mode[0] = 3
                    anchor_r_hand[0] = None
                elif k == ord("4"):
                    sandbox_mode[0] = 4
                    anchor_r_hand[0] = None
                elif k in (ord("c"), ord("C")):
                    anchor_r_hand[0] = None
                    print("[沙盒] 姿态零位已重置 (Re-centered)。")
                elif k in (ord("y"), ord("Y")):
                    save_calib(args.out, solved_R)
                    print("\n" + "=" * 60)
                    print(f"✅ 标定矩阵已成功保存至: {args.out}")
                    print("求解所得 3D 正交旋转矩阵 R:")
                    print(np.round(solved_R, 4))
                    print("=" * 60)
                    return
                elif k in (ord("r"), ord("R")):
                    in_sandbox = False
                    current_step = 0
                    calib_cam_dirs = []
                    calib_task_codes = []
                    history_pts = []
                    history_rot = []
                    is_recording = False
                    print("\n[重新标定] 重置向导至第 1 步。")
                    continue

            # 采样阶段按 SPACE 处理
            if not in_sandbox and k == ord(" "):
                if not is_recording:
                    history_pts = []
                    history_rot = []
                    is_recording = True
                    print(f"\n▶ 开始记录 [{step_title}]，请按提示动作平移/旋转手部...")
                else:
                    is_recording = False
                    mode_type = steps[current_step][4]
                    task_code = steps[current_step][3]

                    if mode_type == "lin":
                        if len(history_pts) < 5:
                            print("⚠️ 采样点过少，请重新按 SPACE 录制。")
                            continue
                        d = history_pts[-1] - history_pts[0]
                        norm = np.linalg.norm(d)
                        if norm < 20.0:
                            print(f"⚠️ 手部位移过小 ({norm:.1f}mm < 20mm)，请重新平移手部。")
                            continue
                        unit_dir = d / norm
                        calib_cam_dirs.append(unit_dir)
                        calib_task_codes.append(task_code)
                        print(f"✓ [{step_title}] 采样成功: 位移 {norm:.1f}mm, 矢量: {np.round(unit_dir, 3)}")

                    elif mode_type == "rot":
                        if len(history_rot) < 5:
                            print("⚠️ 采样点过少，请重新按 SPACE 录制。")
                            continue
                        r_diff = history_rot[-1] @ history_rot[0].T
                        axis = np.array([r_diff[2, 1] - r_diff[1, 2],
                                         r_diff[0, 2] - r_diff[2, 0],
                                         r_diff[1, 0] - r_diff[0, 1]])
                        ax_norm = np.linalg.norm(axis)
                        if ax_norm < 1e-4:
                            print("⚠️ 旋转偏角过小，请重新做翻腕/压腕动作。")
                            continue
                        unit_rot_axis = axis / ax_norm
                        calib_cam_dirs.append(unit_rot_axis)
                        calib_task_codes.append(task_code)
                        print(f"✓ [{step_title}] 采样成功: 旋转轴矢量: {np.round(unit_rot_axis, 3)}")

                    current_step += 1
                    if current_step >= len(steps):
                        # 5 步全部完成，求解 R 并进入沙盒验证
                        solved_R = solve_handeye(calib_cam_dirs, calib_task_codes)
                        in_sandbox = True
                        print("\n" + "=" * 60)
                        print("🎉 5 步采样全部完成！已计算出正交变换矩阵 R:")
                        print(np.round(solved_R, 4))
                        print("现在进入【实时沙盒验证模式】，可在窗口中自由挥手确认方向。")
                        print("确认满意后，按 [Y] 或 [SPACE] 键保存，按 [R] 键重试。")
                        print("=" * 60)

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
