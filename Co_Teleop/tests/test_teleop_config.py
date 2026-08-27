"""Co_Teleop.tests.test_teleop_config — 集中配置文件单元测试 (包含关节限位验证)."""
import tempfile
from pathlib import Path
import pytest

from Co_Teleop.config import (
    TeleopConfig,
    GearConfig,
    JointFactorConfig,
    JointLimitsConfig,
    MotorLimitConfig,
    PresetPoseConfig,
    VisionFilterConfig,
    HandConfig,
)


def test_default_teleop_config_structure():
    cfg = TeleopConfig.default()
    assert cfg.gear.default_gear == 2
    assert cfg.gear.gear_1_low.lin_scale == 0.030
    assert cfg.gear.gear_2_mid.lin_scale == 0.080

    factors = cfg.joint_factor.as_list()
    assert len(factors) == 6
    assert factors[0] == 2.5

    limits = cfg.joint_limits.as_list()
    assert len(limits) == 6
    assert limits[0] == (-1.0, 360.0)
    assert limits[1] == (-1.0, 150.0)
    assert limits[2] == (-1.0, 120.0)
    assert limits[3] == (-90.0, 90.0)
    assert limits[4] == (-1.0, 180.0)
    assert limits[5] == (-1.0, 360.0)
    assert cfg.joint_limits.joint_limit_margin_deg == 2.0

    assert cfg.motor.speed_rpm == 2000.0
    assert cfg.vision.deadband_angle_deg == 5.0


def test_teleop_config_to_dict_and_from_dict():
    cfg = TeleopConfig.default()
    cfg.gear.gear_2_mid.lin_scale = 0.08
    cfg.joint_factor.j1_base_yaw = 2.5
    cfg.joint_limits.j2_shoulder_pitch = [-5.0, 140.0]
    d = cfg.to_dict()
    assert isinstance(d, dict)
    assert d["gear"]["gear_2_mid"]["lin_scale"] == 0.08
    assert d["joint_limits"]["j2_shoulder_pitch"] == [-5.0, 140.0]

    restored = TeleopConfig.from_dict(d)
    assert restored.gear.gear_2_mid.lin_scale == 0.08
    assert restored.joint_factor.j1_base_yaw == 2.5
    assert restored.joint_limits.j2_shoulder_pitch == [-5.0, 140.0]


def test_teleop_config_load_yaml():
    yaml_path = Path(__file__).parents[1] / "config" / "teleop_config.yaml"
    if yaml_path.exists():
        cfg = TeleopConfig.load(yaml_path)
        assert cfg.gear.default_gear == 2
        assert cfg.joint_factor.j5_wrist_pitch == 1.0
        assert cfg.joint_factor.j1_base_yaw == 2.0
        assert cfg.joint_factor.j4_wrist_roll_1 == 1.5
        assert cfg.joint_limits.j1_base_yaw == [-90.0, 270.0]
        assert cfg.joint_limits.j2_shoulder_pitch == [-5.0, 150.0]
        assert cfg.joint_limits.j3_elbow_pitch == [-5.0, 120.0]
        assert cfg.joint_limits.joint_limit_margin_deg == 2.0
        assert cfg.motor.speed_rpm == 1500.0
        assert cfg.motor.position_acc == 0
        assert cfg.motor.max_dq_deg == 10.0
        assert cfg.gear.gear_1_low.lin_scale == 0.15
        assert cfg.gear.gear_2_mid.lin_scale == 0.18
        assert cfg.gear.gear_3_high.lin_scale == 0.20
        assert cfg.pose.ready_pose_deg == [0.0, 75.0, 55.0, 0.0, 130.0, 0.0]
        assert cfg.hand.port == "/dev/ttyUSB0"
        assert cfg.hand.kP == 300
        assert cfg.hand.curr_lim == 150
        assert cfg.hand.source_mode == 2
        assert cfg.vision.pts_min_cutoff == 0.8
        assert cfg.vision.pts_beta == 0.020
        assert cfg.vision.deadband_vel_mm_s == 15.0
        assert cfg.vision.deadband_angle_deg == 5.0


def test_teleop_config_save_and_load_temp():
    cfg = TeleopConfig.default()
    cfg.gear.default_gear = 3
    cfg.motor.speed_rpm = 2500.0
    cfg.hand.curr_lim = 400
    cfg.joint_limits.j3_elbow_pitch = [-10.0, 110.0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_yaml = Path(tmp_dir) / "test_cfg.yaml"
        cfg.save_yaml(tmp_yaml)
        assert tmp_yaml.exists()

        loaded = TeleopConfig.load(tmp_yaml)
        assert loaded.gear.default_gear == 3
        assert loaded.motor.speed_rpm == 2500.0
        assert loaded.hand.curr_lim == 400
        assert loaded.joint_limits.j3_elbow_pitch == [-10.0, 110.0]


def test_teleop_config_validation_safety():
    cfg = TeleopConfig.default()
    # 正常无异常
    assert isinstance(cfg.validate(), list)

    # 异常测试 1: 电机超速
    cfg.motor.speed_rpm = 4000.0
    with pytest.raises(ValueError, match="超出硬件极限"):
        cfg.validate()
    cfg.motor.speed_rpm = 2000.0

    # 异常测试 2: 负速度比例
    cfg.gear.gear_1_low.lin_scale = -0.5
    with pytest.raises(ValueError, match="超出安全范围"):
        cfg.validate()
    cfg.gear.gear_1_low.lin_scale = 0.03

    # 异常测试 3: 关节倍率非法
    cfg.joint_factor.j1_base_yaw = 10.0
    with pytest.raises(ValueError, match="超出安全范围"):
        cfg.validate()
    cfg.joint_factor.j1_base_yaw = 2.5

    # 异常测试 4: 关节限位下限大于上限
    cfg.joint_limits.j2_shoulder_pitch = [150.0, 10.0]
    with pytest.raises(ValueError, match="下限 lo=150.0° 必须严格小于上限 hi=10.0°"):
        cfg.validate()
    cfg.joint_limits.j2_shoulder_pitch = [-1.0, 150.0]

    # 异常测试 5: 预设姿态超出用户限位
    cfg.joint_limits.j2_shoulder_pitch = [0.0, 50.0]  # READY pose J2 is 75°
    with pytest.raises(ValueError, match="READY 姿态 J2=75.0° 超出设置的关节限位"):
        cfg.validate()
    cfg.joint_limits.j2_shoulder_pitch = [-1.0, 150.0]
