"""Co_Teleop.tests.test_unified_teleop — 臂手协同统一遥操管线单元测试."""
import numpy as np
import pytest

from lerobot_robot_massage.zdt.types import CartesianCommand, JointState
from Co_Teleop.adapters import NoDriveArmAdapter, NoDriveHandAdapter
from Co_Teleop.config import TeleopConfig, build_gear_configs
from Co_Teleop.gui import (
    MODE_FULL,
    MODE_KNEAD,
    MODE_NAMES,
    MODE_PITCH,
    MODE_ROLL,
    draw_unified_dashboard,
)


def test_unified_dashboard_render():
    """测试一体化 HUD 渲染逻辑不报错且正确在画面上绘制所有子面板与手腕导引箭头."""
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    teleop_cfg = TeleopConfig.default()
    gear_configs = build_gear_configs(teleop_cfg)

    joint_state = JointState(
        q=(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        dq=(0.0,) * 6,
        current_ma=(120.0, 150.0, 180.0, 80.0, 90.0, 70.0),
        flags=(0,) * 6,
        status="ARMED",
    )
    hand_angles = np.ones(16, dtype=np.float64) * 0.4
    hand_bent = [True, False, False, True]

    draw_unified_dashboard(
        frame=dummy_frame,
        arm_out={
            "action": "MOVE",
            "v": np.array([25.0, -15.0, 10.0]),
            "w": np.array([0.1, -0.2, 0.0]),
            "d_roll_deg": 12.5,
            "d_pitch_deg": -5.0,
            "wrist_px": (640, 360),
            "wd_scale": 1.0,
        },
        gear_info=gear_configs[2],
        clutch_active=True,
        arm_mode=MODE_ROLL,
        joint_state=joint_state,
        no_drive_arm=True,
        hand_state_str="POWERED",
        source_name="MP PSEUDO-3D",
        hand_angles=hand_angles,
        hand_bent=hand_bent,
        fps=30.0,
    )

    # 画面不应全黑且手腕箭头处应有渲染像素
    assert dummy_frame.sum() > 0
    assert dummy_frame[360, 640].sum() > 0


def test_unified_arm_hand_coordination():
    """测试臂-手解耦执行器协同工作."""
    arm = NoDriveArmAdapter()
    hand = NoDriveHandAdapter()

    arm.connect()
    arm.arm(gravity_confirmed=True)
    hand.connect()

    # 机械臂发平移指令
    arm_cmd = CartesianCommand((20.0, 0.0, 0.0), (0.0, 0.0, 0.0), timestamp=1.0)
    arm.move_cartesian_velocity(arm_cmd)
    assert arm.state() in ("SAFE_IDLE", "ARMED", "TELEOP")

    # 灵巧手发五指弯曲指令
    hand_angles = np.zeros(16, dtype=np.float64)
    hand_angles[0] = 0.5  # 食指侧摆
    hand_angles[1] = 0.8  # 食指弯曲
    hand.set_angles(hand_angles)
    assert hand.state() == "TELEOP"
    np.testing.assert_allclose(hand.get_current_angles(), hand_angles)

    # 统一复位
    arm.ready()
    hand.set_open()
    np.testing.assert_allclose(hand.get_current_angles(), np.zeros(16))

    arm.disconnect()
    hand.disconnect()
    assert not hand.is_connected()
