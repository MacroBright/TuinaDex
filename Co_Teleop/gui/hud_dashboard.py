"""Co_Teleop.gui.hud_dashboard — 统一 1280x720 宽屏 HUD 仪表盘渲染器.

提供机械臂 6-DOF 状态表、灵巧手 16-DOF 弯曲条形图、速度/姿态矢量、
看门狗状态与操作提示栏的半透明专业渲染。
"""
from __future__ import annotations

import cv2
import numpy as np

FINGER_NAMES = ["Index", "Middle", "Ring", "Thumb"]

MODE_KNEAD = 1
MODE_ROLL = 2
MODE_PITCH = 3
MODE_FULL = 4

MODE_NAMES = {
    MODE_KNEAD: "Knead (Lock)",
    MODE_ROLL: "Roll (Roll Only)",
    MODE_PITCH: "Pitch (Pitch Only)",
    MODE_FULL: "Full (Pitch+Roll)",
}


def draw_unified_dashboard(
    frame: np.ndarray,
    arm_out: dict,
    gear_info: dict,
    clutch_active: bool,
    arm_mode: int,
    joint_state=None,
    no_drive_arm: bool = False,
    hand_state_str: str = "OFF",
    source_name: str = "PSEUDO-3D",
    hand_angles: np.ndarray | None = None,
    hand_bent: list[bool] | None = None,
    fps: float = 0.0,
) -> None:
    """在全视野 1280x720 画面上绘制专业现代科技风 HUD 仪表盘."""
    h, w = frame.shape[:2]

    # 1. 绘制半透明黑色背景遮罩 (HUD 卡片底色)
    overlay = frame.copy()

    # 顶部 Header 状态栏底色 (0~44px)
    cv2.rectangle(overlay, (0, 0), (w, 44), (16, 18, 24), -1)

    # 左上动作面板底色 (lx0, ly0, w=235, h=145)
    lx0, ly0 = 14, 52
    l_box_w, l_box_h = 235, 145
    cv2.rectangle(overlay, (lx0, ly0), (lx0 + l_box_w, ly0 + l_box_h), (16, 18, 24), -1)

    # 右上机械臂 6 轴状态面板底色 (rx0, ry0, w=225, h=150)
    rx0, ry0 = w - 240, 52
    box_w, box_h = 225, 150
    cv2.rectangle(overlay, (rx0, ry0), (rx0 + box_w, ry0 + box_h), (16, 18, 24), -1)

    # 左下灵巧手 16 关节状态面板底色 (hx0, hy0, w=260, h=170)
    hx0, hy0 = 14, h - 30 - 175
    h_box_w, h_box_h = 260, 170
    cv2.rectangle(overlay, (hx0, hy0), (hx0 + h_box_w, hy0 + h_box_h), (16, 18, 24), -1)

    # 底部快捷键提示栏底色
    cv2.rectangle(overlay, (0, h - 28), (w, h), (16, 18, 24), -1)

    # 高透明度混合: 35% 黑色遮罩 + 65% 相机原画 (用户手部清晰透见)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # 2. 绘制面板边框线
    cv2.line(frame, (0, 44), (w, 44), (60, 70, 80), 1)
    cv2.line(frame, (0, h - 28), (w, h - 28), (60, 70, 80), 1)
    cv2.rectangle(frame, (lx0, ly0), (lx0 + l_box_w, ly0 + l_box_h), (0, 220, 255), 1)
    cv2.rectangle(frame, (rx0, ry0), (rx0 + box_w, ry0 + box_h), (0, 220, 255), 1)
    cv2.rectangle(frame, (hx0, hy0), (hx0 + h_box_w, hy0 + h_box_h), (0, 255, 180), 1)

    # 3. 顶部 Header 状态栏内容
    if clutch_active:
        cv2.rectangle(frame, (10, 7), (140, 37), (0, 180, 80), -1)
        cv2.putText(frame, "[SPACE] RUN", (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame, (10, 7), (140, 37), (40, 40, 200), -1)
        cv2.putText(frame, "[SPACE] PAUSE", (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)

    # 机械臂模式
    cv2.putText(frame, f"ARM: {MODE_NAMES.get(arm_mode, 'Knead')}", (155, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 2)

    # 档位徽标
    badge_col = gear_info.get("color", (0, 220, 255))
    cv2.rectangle(frame, (510, 7), (595, 37), (30, 30, 30), -1)
    cv2.rectangle(frame, (510, 7), (595, 37), badge_col, 2)
    cv2.putText(frame, f"[S] {gear_info.get('badge', 'MID')}", (516, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_col, 2)

    # 灵巧手状态
    hand_col = (0, 255, 120) if "POWERED" in hand_state_str else (160, 160, 160)
    cv2.putText(frame, f"HAND: {hand_state_str}", (615, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, hand_col, 2)
    cv2.putText(frame, f"3D: {source_name}", (845, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)
    cv2.putText(frame, f"{fps:3.0f} FPS", (w - 95, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 180) if fps >= 45 else (220, 220, 220), 2)

    # 4. 左上紧凑动作姿态面板 (Motion & Attitude Panel)
    cv2.putText(frame, "MOTION & ATTITUDE", (lx0 + 8, ly0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1)

    vel = arm_out.get("v", np.zeros(3))
    ang = arm_out.get("w", np.zeros(3))
    if clutch_active:
        x_dir = "左" if vel[0] > 4 else ("右" if vel[0] < -4 else "-")
        y_dir = "后" if vel[1] > 4 else ("前" if vel[1] < -4 else "-")
        z_dir = "上" if vel[2] > 4 else ("下" if vel[2] < -4 else "-")
        spd_txt = f"v:[X:{vel[0]:+3.0f}({x_dir}) Y:{vel[1]:+3.0f}({y_dir}) Z:{vel[2]:+3.0f}({z_dir})]"
        cv2.putText(frame, spd_txt, (lx0 + 8, ly0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 120), 1)

        p_tag = "抬" if ang[0] > 0.04 else ("低" if ang[0] < -0.04 else "-")
        r_tag = "左" if ang[1] > 0.04 else ("右" if ang[1] < -0.04 else "-")
        ang_txt = f"w:[P:{ang[0]:+4.2f}({p_tag}) R:{ang[1]:+4.2f}({r_tag})]"
        cv2.putText(frame, ang_txt, (lx0 + 8, ly0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 230, 0), 1)
    else:
        cv2.putText(frame, "v:[X:+0 Y:+0 Z:+0] [PAUSED]", (lx0 + 8, ly0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)
        cv2.putText(frame, "w:[P:+0.00 R:+0.00] [PAUSED]", (lx0 + 8, ly0 + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 180, 255), 1)

    # 掌面虚拟摇杆角度 (Tilt Angles)
    roll_deg = float(arm_out.get("d_roll_deg", 0.0))
    pitch_deg = float(arm_out.get("d_pitch_deg", 0.0))
    h_roll_tag = "左" if roll_deg > 4.0 else ("右" if roll_deg < -4.0 else "-")
    h_pitch_tag = "抬" if pitch_deg > 4.0 else ("压" if pitch_deg < -4.0 else "-")
    tilt_txt = f"Tilt: R:{roll_deg:+4.1f}°({h_roll_tag}) P:{pitch_deg:+4.1f}°({h_pitch_tag})"
    cv2.putText(frame, tilt_txt, (lx0 + 8, ly0 + 96), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 230, 255), 1)

    # 看门狗与安全状态
    act_str = str(arm_out.get("action", "OK"))
    act_col = (0, 255, 120) if act_str in ("OK", "MOVE") else ((0, 160, 255) if act_str == "DECAY" else (0, 0, 255))
    status_txt = f"WD: {act_str} ({float(arm_out.get('wd_scale', 1.0))*100:.0f}%) | {gear_info.get('name', 'MID')}"
    cv2.putText(frame, status_txt, (lx0 + 8, ly0 + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.35, act_col, 1)

    # 5. 右上紧凑机械臂 6 轴状态表 (220px 宽)
    arm_title = "ARM (6-DOF) [SIM]" if no_drive_arm else "ARM (6-DOF) [REAL]"
    cv2.putText(frame, arm_title, (rx0 + 8, ry0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1)

    if joint_state is not None:
        for idx in range(6):
            col = idx % 2
            row = idx // 2
            x = rx0 + 8 + col * 105
            y = ry0 + 44 + row * 34
            q_val = float(joint_state.q[idx]) if idx < len(joint_state.q) else 0.0
            cur_val = float(joint_state.current_ma[idx]) if idx < len(joint_state.current_ma) else 0.0
            cv2.putText(frame, f"J{idx+1}: {q_val:+5.1f}°", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (255, 255, 255), 1)
            cv2.putText(frame, f"   {cur_val:3.0f}mA", (x, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 220, 160), 1)

    # 6. 左下紧凑灵巧手 16 关节弯曲条形图 (260x170px)
    hand_title = "LEAP HAND (16-DOF)"
    cv2.putText(frame, hand_title, (hx0 + 8, hy0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1)

    if hand_angles is not None:
        for f_idx in range(4):
            fy = hy0 + 36 + f_idx * 33
            f_name = FINGER_NAMES[f_idx][:3]
            is_bent = hand_bent[f_idx] if hand_bent and f_idx < len(hand_bent) else False
            f_col = (0, 255, 120) if is_bent else (180, 180, 180)
            cv2.putText(frame, f_name, (hx0 + 8, fy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, f_col, 1)

            for j_idx in range(4):
                motor_id = f_idx * 4 + j_idx
                angle_val = float(hand_angles[motor_id]) if motor_id < len(hand_angles) else 0.0
                bar_w = int(np.clip(angle_val / 2.0, 0.0, 1.0) * 44)
                bx = hx0 + 42 + j_idx * 52
                by = fy
                cv2.rectangle(frame, (bx, by), (bx + 44, by + 8), (45, 50, 55), -1)
                cv2.rectangle(frame, (bx, by), (bx + bar_w, by + 8), (0, 220, 255), -1)
                cv2.putText(frame, f"M{motor_id}", (bx, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.24, (160, 160, 160), 1)

    # 7. 手腕锚定标记与动态运动引导箭头 (Wrist Movement Arrow)
    wrist_px = arm_out.get("wrist_px")
    if wrist_px is not None:
        u, v = int(wrist_px[0]), int(wrist_px[1])
        cv2.circle(frame, (u, v), 8, (0, 255, 0), -1)
        cv2.circle(frame, (u, v), 12, (255, 255, 255), 2)

        v_mag = float(np.linalg.norm(vel))
        if clutch_active and v_mag > 4.0:
            dx = int(np.clip(vel[0] * 1.5, -80, 80))
            dy = int(np.clip(vel[1] * 1.5, -80, 80))
            if abs(dx) > 3 or abs(dy) > 3:
                cv2.arrowedLine(frame, (u, v), (u + dx, v + dy), (0, 255, 255), 3, tipLength=0.25)
                cv2.putText(frame, f"{v_mag:.0f} mm/s", (u + dx + 6, v + dy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)

    # 8. 底部操作提示栏
    cv2.putText(frame, "SPACE: Arm Pause | R: Ready | H: Home | L: Hand Pause | Z: Zero | W: WD Reset | K: Calib | S/TAB: Gear | Q: Quit",
                (12, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
