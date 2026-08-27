"""Co_Teleop/config/teleop_config.py — 视觉遥操系统全参数集中配置文件 (Modular Teleoperation Configuration).

【设计说明】
本文件是真机视觉遥操系统的“单一可信配置源 (Single Source of Truth)”。
所有关于【灵敏度档位】、【各关节独立倍率】、【机械臂关节限位】、【底层电机极限】、【预设作业姿态】、【视觉滤波】与【灵巧手参数】
均集中配置于此，且每个参数均附带：
  1. 作用对象 (硬件电机 / 算法模块 / UI层)
  2. 物理单位 (RPM, mm/s, rad/s, 度, 归一化比例等)
  3. 调参影响与工程建议范围
如需调整参数，直接编辑对应的 teleop_config.yaml 即可在遥操中立即生效！
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# 1. 灵敏度档位系统配置 (Gear & Sensitivity Configuration)
# ==============================================================================

@dataclass
class SingleGearSetting:
    """单个灵敏度档位参数配置."""
    name: str                   # 档位名称 (用于控制台日志与 HUD 提示)
    badge: str                  # 界面角标 (如 "LOW", "MID", "HIGH")
    color: Tuple[int, int, int] # OpenCV BGR 颜色格式 (如 (0, 230, 100)=浅绿, (0, 220, 255)=明黄, (0, 120, 255)=橙红)

    # ── 平移线速度控制 ──
    lin_scale: float
    """
    【笛卡尔平移速度比例 (Linear Speed Scale)】
    - 作用对象: 机械臂末端在基坐标系下的 XYZ 轴平移线速度 (vx, vy, vz).
    - 物理单位: 归一化比例 (0.01 ~ 1.0, 1.0 表示 100% 映射).
    - 调参建议:
        * 低速档: 0.02 ~ 0.04 (3%~4%), 用于精准找穴、微距贴合;
        * 中速档: 0.05 ~ 0.08 (5%~8%), 日常标准推拿揉捏，平顺不急促;
        * 高速档: 0.08 ~ 0.15 (8%~15%), 跨区域大范围换位.
    """

    # ── 姿态旋转角速度控制 ──
    max_omega: float
    """
    【姿态最大角速度上限 (Max Angular Velocity)】
    - 作用对象: 虚拟手势摇杆 (Roll 滚转 / Pitch 俯仰) 输出的最大旋转角速度.
    - 物理单位: rad/s (弧度/秒, 1.0 rad/s ≈ 57.3°/s).
    - 调参建议:
        * 低速档: 0.20 ~ 0.40 rad/s (约 11°~23°/s), 适合精细微调角度;
        * 中速档: 0.50 ~ 0.80 rad/s (约 28°~46°/s), 适合平稳调整手腕姿态;
        * 高速档: 0.90 ~ 1.50 rad/s (约 51°~86°/s), 适合快速翻转手腕.
    """

    # ── 动态扫掠加速度 ──
    gain_xyz: float
    """
    【动态挥手位移增益 (Dynamic Swipe Acceleration Gain)】
    - 作用对象: 快速挥动手掌时产生的位移放大倍率 (鼠标级动力学加速).
    - 物理单位: 无量纲倍数 (1.0 ~ 3.0).
    - 调参建议:
        * 1.0: 纯线性映射 (手移多少臂移多少);
        * 1.1 ~ 1.3: 轻微加速，快速挥手时单次可跨越更远距离，减少反复离合.
    """


@dataclass
class GearConfig:
    """3 档灵敏度变速箱系统全局配置."""
    default_gear: int = 2       # 默认启动档位 (1=低速, 2=中速, 3=高速)
    gear_1_low: SingleGearSetting = field(default_factory=lambda: SingleGearSetting(
        name="1.低速档",
        badge="LOW",
        color=(0, 230, 100),    # 浅绿色
        lin_scale=0.030,        # 3.0%
        max_omega=0.10,         # 0.10 rad/s (5.7°/s)
        gain_xyz=1.0,           # 1.0x 纯线性
    ))
    gear_2_mid: SingleGearSetting = field(default_factory=lambda: SingleGearSetting(
        name="2.中速档",
        badge="MID",
        color=(0, 220, 255),    # 明黄色
        lin_scale=0.080,        # 8.0%
        max_omega=0.20,         # 0.20 rad/s (11.5°/s)
        gain_xyz=1.0,           # 1.0x 线性平稳
    ))
    gear_3_high: SingleGearSetting = field(default_factory=lambda: SingleGearSetting(
        name="3.高速档",
        badge="HIGH",
        color=(0, 120, 255),    # 亮橙色
        lin_scale=0.085,        # 8.5%
        max_omega=0.25,         # 0.25 rad/s (14.3°/s)
        gain_xyz=1.1,           # 1.1x 轻度动态加速
    ))


def build_gear_configs(cfg: TeleopConfig) -> dict:
    """由 TeleopConfig 数据类动态构建 HUD 与速度计算字典 (1=LOW, 2=MID, 3=HIGH)."""
    return {
        1: {
            "name": cfg.gear.gear_1_low.name,
            "badge": cfg.gear.gear_1_low.badge,
            "color": cfg.gear.gear_1_low.color,
            "lin_scale": cfg.gear.gear_1_low.lin_scale,
            "max_omega": cfg.gear.gear_1_low.max_omega,
            "gain_xyz": cfg.gear.gear_1_low.gain_xyz,
        },
        2: {
            "name": cfg.gear.gear_2_mid.name,
            "badge": cfg.gear.gear_2_mid.badge,
            "color": cfg.gear.gear_2_mid.color,
            "lin_scale": cfg.gear.gear_2_mid.lin_scale,
            "max_omega": cfg.gear.gear_2_mid.max_omega,
            "gain_xyz": cfg.gear.gear_2_mid.gain_xyz,
        },
        3: {
            "name": cfg.gear.gear_3_high.name,
            "badge": cfg.gear.gear_3_high.badge,
            "color": cfg.gear.gear_3_high.color,
            "lin_scale": cfg.gear.gear_3_high.lin_scale,
            "max_omega": cfg.gear.gear_3_high.max_omega,
            "gain_xyz": cfg.gear.gear_3_high.gain_xyz,
        },
    }


# ==============================================================================
# 2. 6 关节独立速度补偿配置 (Per-Joint Speed Factor Configuration)
# ==============================================================================

@dataclass
class JointFactorConfig:
    """6 关节独立速度响应倍率 (针对不同减速比实现手感一致性)."""
    j1_base_yaw: float = 2.5
    """J1 基座回转 / X 轴横向摆臂倍率 (默认 2.5x, 补偿 51:1 减速比)"""

    j2_shoulder_pitch: float = 2.5
    """J2 大臂俯仰 / 前后上下主推力倍率 (默认 2.5x, 补偿 51:1 减速比)"""

    j3_elbow_pitch: float = 2.5
    """J3 小臂俯仰 / 空间伸缩主推力倍率 (默认 2.5x, 补偿 51:1 减速比)"""

    j4_wrist_roll_1: 1.5 = 1.5
    """J4 腕部滚转 1 轴倍率 (默认 1.5x, 补偿 51:1 减速比)"""

    j5_wrist_pitch: float = 1.0
    """J5 手腕俯仰 轴倍率 (默认 1.0x 基准, 27:1 减速比响应合适)"""

    j6_wrist_roll_2: float = 1.0
    """J6 腕部滚转 2 / 末端自转倍率 (默认 1.0x, 补偿 51:1 减速比)"""

    def as_list(self) -> List[float]:
        """返回 6 元素浮点列表 [J1, J2, J3, J4, J5, J6]."""
        return [
            float(self.j1_base_yaw),
            float(self.j2_shoulder_pitch),
            float(self.j3_elbow_pitch),
            float(self.j4_wrist_roll_1),
            float(self.j5_wrist_pitch),
            float(self.j6_wrist_roll_2),
        ]


# ==============================================================================
# 3. 机械臂 6 关节软件限位与缓冲配置 (Joint Limits & Safety Margin in Degrees)
# ==============================================================================

@dataclass
class JointLimitsConfig:
    """机械臂 6 关节物理与软件安全限位配置 (单位: 度 / deg)."""
    j1_base_yaw: list[float] = field(default_factory=lambda: [-1.0, 360.0])
    """【J1 基座水平回转限位】 [min, max] (默认 [-1.0, 360.0] 度, 360° 旋转)"""

    j2_shoulder_pitch: list[float] = field(default_factory=lambda: [-1.0, 150.0])
    """【J2 大臂主俯仰限位】 [min, max] (默认 [-1.0, 150.0] 度, 避免大臂后仰撞底座)"""

    j3_elbow_pitch: list[float] = field(default_factory=lambda: [-1.0, 120.0])
    """【J3 小臂肘部俯仰限位】 [min, max] (默认 [-1.0, 120.0] 度, 避免折臂自碰)"""

    j4_wrist_roll_1: list[float] = field(default_factory=lambda: [-90.0, 90.0])
    """【J4 小臂滚转限位】 [min, max] (默认 [-90.0, 90.0] 度)"""

    j5_wrist_pitch: list[float] = field(default_factory=lambda: [-1.0, 180.0])
    """【J5 手腕俯仰限位】 [min, max] (默认 [-1.0, 180.0] 度)"""

    j6_wrist_roll_2: list[float] = field(default_factory=lambda: [-1.0, 360.0])
    """【J6 末端自转限位】 [min, max] (默认 [-1.0, 360.0] 度, 360° 旋转)"""

    joint_limit_margin_deg: float = 2.0
    """【接近限位时的缓冲减速边界】 (度, 越接近边界越慢直至平滑停下, 默认 2.0°)"""

    def as_list(self) -> list[tuple[float, float]]:
        """获取 6 关节 (min, max) 元组列表."""
        return [
            (float(self.j1_base_yaw[0]), float(self.j1_base_yaw[1])),
            (float(self.j2_shoulder_pitch[0]), float(self.j2_shoulder_pitch[1])),
            (float(self.j3_elbow_pitch[0]), float(self.j3_elbow_pitch[1])),
            (float(self.j4_wrist_roll_1[0]), float(self.j4_wrist_roll_1[1])),
            (float(self.j5_wrist_pitch[0]), float(self.j5_wrist_pitch[1])),
            (float(self.j6_wrist_roll_2[0]), float(self.j6_wrist_roll_2[1])),
        ]


# ==============================================================================
# 4. 底层电机驱动与安全限速配置 (Motor Limits & Safety Configuration)
# ==============================================================================

@dataclass
class MotorLimitConfig:
    """底层闭环步进驱动器 (Emm42 V5.0) 与笛卡尔控制器安全极限."""

    speed_rpm: float = 2000.0
    """
    【0xFD CAN 报文电机轴最高转速上限 (Max Motor Shaft RPM)】
    - 作用对象: 驱动器底层 0xFD 相对位置脉冲下发报文中的速度字段 (Speed RPM).
    - 物理单位: RPM (转/分钟).
    - 说明: 42 步进电机最高额定转速 3000 RPM，2000 RPM 留足 33% 扭矩裕度.
    """

    position_acc: int = 0
    """
    【电机加速度启动档位 (Position Acceleration Step)】
    - 作用对象: 0xFD 报文中的加速度字段 acc.
    - 物理含义: 0 = 无加减速延迟 (直冲最高速, 50ms 周期运动最佳响应); 1~255 为梯形加减速.
    """

    max_vel_mm_s: float = 600.0
    """
    【笛卡尔末端平移最大线速度硬限幅 (Max Cartesian Linear Velocity)】
    - 作用对象: CartesianController.step() 输入线速度向量范数 ||v||.
    - 物理单位: mm/s (毫米/秒).
    """

    max_ang_rad_s: float = 10.0
    """
    【笛卡尔末端旋转最大角速度硬限幅 (Max Cartesian Angular Velocity)】
    - 作用对象: CartesianController.step() 输入角速度向量范数 ||w||.
    - 物理单位: rad/s (弧度/秒).
    """

    max_joint_vel_deg_s: float = 540.0
    """
    【关节输出轴最大角速度 (Max Joint Output Velocity)】
    - 作用对象: 每个控制周期 (50ms) 内允许的单轴最大角速度.
    - 物理单位: deg/s (度/秒, 540°/s = 1.5 圈/秒).
    """

    max_joint_acc_deg_s2: float = 2000.0
    """
    【关节输出轴最大角加速度 (Max Joint Acceleration)】
    - 作用对象: 连续两帧之间允许的最大关节速度突变率.
    - 物理单位: deg/s² (度/秒平方).
    """

    max_dq_deg: float = 30.0
    """
    【单步最大允许角度跳变 (Max Delta Angle Per Step)】
    - 作用对象: 防止逆解多解跳变或丢步导致飞车.
    - 物理单位: deg (度).
    """


# ==============================================================================
# 5. 预设作业与复位姿态配置 (Preset Poses Configuration)
# ==============================================================================

@dataclass
class PresetPoseConfig:
    """机械臂常用作业角度与复位角度 (单位: 度 / deg)."""

    ready_pose_deg: List[float] = field(default_factory=lambda: [0.0, 75.0, 55.0, 0.0, 130.0, 0.0])
    """
    【标准按摩推拿准备姿态 (READY POSE)】
    - 关节角度: [J1=0°, J2=75°, J3=55°, J4=0°, J5=130°, J6=0°].
    """

    home_pose_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    """
    【上电初始垂直复位姿态 (HOME POSE)】
    - 关节角度: [0°, 0°, 0°, 0°, 0°, 0°].
    """

    ready_speed_rpm: float = 100.0
    """
    【姿态同步运动转速 (Ready Movement Speed)】
    - 作用对象: 运动到 READY / HOME 姿态时的电机转速 (100 RPM 保证平稳安全).
    """


# ==============================================================================
# 6. 视觉跟踪与滤波配置 (Vision Tracking & Filter Configuration)
# ==============================================================================

@dataclass
class VisionFilterConfig:
    """手势视觉跟踪与滤波抗噪配置."""

    pts_min_cutoff: float = 1.5
    """【手腕 3D 坐标 1€ 滤波基准截止频率 (Min Cutoff)】 (Hz, 建议 1.0~2.0)"""

    pts_beta: float = 0.08
    """【手腕 3D 坐标 1€ 滤波速度响应系数 (Beta)】 (建议 0.05~0.12)"""

    rot_min_cutoff: float = 1.0
    """【手掌姿态李代数 1€ 滤波基准截止频率 (Min Cutoff)】 (Hz)"""

    rot_beta: float = 0.05
    """【手掌姿态李代数 1€ 滤波速度响应系数 (Beta)】"""

    deadband_angle_deg: float = 5.0
    """【虚拟摇杆姿态倾斜锁定死区角度 (Angle Deadband)】 (度, 倾斜 < 5.0° 视为中立不旋转)"""

    deadband_vel_mm_s: float = 10.0
    """【消除生理手颤的平移最小静止死区 (Velocity Deadband)】 (mm/s, 低于 10mm/s 视为静止)"""


# ==============================================================================
# 7. 灵巧手控制与安全配置 (Dexterous Hand Configuration)
# ==============================================================================

@dataclass
class HandConfig:
    """LEAP Hand 16-DOF 灵巧手硬件与视觉映射配置."""

    port: str = "/dev/ttyUSB0"
    """【灵巧手 Dynamixel 串口路径】"""

    kP: int = 300
    """【位置环比例增益 kP】 (默认 300)"""

    kI: int = 0
    """【位置环积分增益 kI】 (默认 0)"""

    kD: int = 100
    """【位置环微分增益 kD】 (默认 100)"""

    curr_lim: int = 150
    """【电机最大电流限制 (Current Limit)】 (mA, 默认 150mA, 防止揉捏过力堵转发热)"""

    source_mode: int = 2
    """【3D 关键点来源模式】 (0: HAMER, 1: WORLD, 2: PSEUDO-3D 默认)"""

    filter_min_cutoff: float = 1.0
    """【手指 16 关节 1€ 滤波基准截止频率】 (Hz)"""

    filter_beta: float = 0.02
    """【手指 16 关节 1€ 滤波速度响应系数】"""

    bend_threshold: float = 0.20
    """【手指弯曲判定阈值】 (rad)"""

    hand_type: str = "right"
    """【物理控制目标手】 ('right' / 'left' / 'first')"""


# ==============================================================================
# 8. 全局汇总配置主类 (Master Teleoperation Configuration)
# ==============================================================================

@dataclass
class TeleopConfig:
    """视觉遥操系统全参数主配置容器."""
    gear: GearConfig = field(default_factory=GearConfig)
    joint_factor: JointFactorConfig = field(default_factory=JointFactorConfig)
    joint_limits: JointLimitsConfig = field(default_factory=JointLimitsConfig)
    motor: MotorLimitConfig = field(default_factory=MotorLimitConfig)
    pose: PresetPoseConfig = field(default_factory=PresetPoseConfig)
    vision: VisionFilterConfig = field(default_factory=VisionFilterConfig)
    hand: HandConfig = field(default_factory=HandConfig)

    def to_dict(self) -> Dict[str, Any]:
        """转换为标准字典 (便于序列化为 JSON / YAML)."""
        return asdict(self)

    @classmethod
    def default(cls) -> "TeleopConfig":
        """获取默认配置实例."""
        return cls()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TeleopConfig":
        """从字典还原强类型 TeleopConfig 对象."""
        cfg = cls()
        if "gear" in data and isinstance(data["gear"], dict):
            g = data["gear"]
            if "default_gear" in g:
                cfg.gear.default_gear = int(g["default_gear"])
            for g_key in ("gear_1_low", "gear_2_mid", "gear_3_high"):
                if g_key in g and isinstance(g[g_key], dict):
                    setattr(cfg.gear, g_key, SingleGearSetting(**g[g_key]))

        if "joint_factor" in data and isinstance(data["joint_factor"], dict):
            cfg.joint_factor = JointFactorConfig(**data["joint_factor"])

        if "joint_limits" in data and isinstance(data["joint_limits"], dict):
            jl = data["joint_limits"]
            cfg.joint_limits = JointLimitsConfig(
                j1_base_yaw=list(jl.get("j1_base_yaw", [-1.0, 360.0])),
                j2_shoulder_pitch=list(jl.get("j2_shoulder_pitch", [-1.0, 150.0])),
                j3_elbow_pitch=list(jl.get("j3_elbow_pitch", [-1.0, 120.0])),
                j4_wrist_roll_1=list(jl.get("j4_wrist_roll_1", [-90.0, 90.0])),
                j5_wrist_pitch=list(jl.get("j5_wrist_pitch", [-1.0, 180.0])),
                j6_wrist_roll_2=list(jl.get("j6_wrist_roll_2", [-1.0, 360.0])),
                joint_limit_margin_deg=float(jl.get("joint_limit_margin_deg", 2.0)),
            )

        if "motor" in data and isinstance(data["motor"], dict):
            cfg.motor = MotorLimitConfig(**data["motor"])

        if "pose" in data and isinstance(data["pose"], dict):
            cfg.pose = PresetPoseConfig(**data["pose"])

        if "vision" in data and isinstance(data["vision"], dict):
            cfg.vision = VisionFilterConfig(**data["vision"])

        if "hand" in data and isinstance(data["hand"], dict):
            cfg.hand = HandConfig(**data["hand"])

        return cfg

    def validate(self) -> List[str]:
        """
        全面验证配置参数的物理合理性与安全边界 (Pre-Flight Safety Validator).
        - 若存在严重安全隐患，抛出 ValueError 中断启动；
        - 若存在调优建议/提示项，返回警告信息列表.
        """
        warnings: List[str] = []

        # 1. 档位系统安全校验
        for g_name, g in [("1.低速档", self.gear.gear_1_low),
                          ("2.中速档", self.gear.gear_2_mid),
                          ("3.高速档", self.gear.gear_3_high)]:
            if not (0.001 <= g.lin_scale <= 3.0):
                raise ValueError(f"[{g_name}] 平移比例 lin_scale={g.lin_scale} 超出安全范围 [0.001, 3.0]")
            if not (0.01 <= g.max_omega <= 10.0):
                raise ValueError(f"[{g_name}] 姿态角速度 max_omega={g.max_omega} rad/s 超出安全范围 [0.01, 10.0]")
            if not (1.0 <= g.gain_xyz <= 5.0):
                raise ValueError(f"[{g_name}] 动态增益 gain_xyz={g.gain_xyz} 超出安全范围 [1.0, 5.0]")
            if g.lin_scale > 1.0:
                warnings.append(f"[{g_name}] 平移比例 lin_scale={g.lin_scale:.2f} (>100%) 处于超高速档，请在空旷区域谨慎操作")

        # 2. 6 关节独立倍率安全校验
        factors = self.joint_factor.as_list()
        for idx, f in enumerate(factors, start=1):
            if not (0.1 <= f <= 5.0):
                raise ValueError(f"[关节 J{idx}] 速度倍率 factor={f} 超出安全范围 [0.1, 5.0]")

        # 3. 机械臂 6 关节限位安全校验
        limits_list = self.joint_limits.as_list()
        for idx, (lo, hi) in enumerate(limits_list, start=1):
            if lo >= hi:
                raise ValueError(f"[关节限位 J{idx}] 下限 lo={lo}° 必须严格小于上限 hi={hi}°")
        if not (0.0 <= self.joint_limits.joint_limit_margin_deg <= 20.0):
            raise ValueError(f"[关节限位] 减速缓冲边界 margin={self.joint_limits.joint_limit_margin_deg}° 超出有效范围 [0.0, 20.0]")

        # 4. 电机与控制器安全极限校验
        if not (50.0 <= self.motor.speed_rpm <= 3000.0):
            raise ValueError(f"[电机限速] speed_rpm={self.motor.speed_rpm} RPM 超出硬件极限 [50, 3000]")
        if self.motor.position_acc not in range(256):
            raise ValueError(f"[电机加速度] position_acc={self.motor.position_acc} 必须在 0~255 之间")
        if not (10.0 <= self.motor.max_vel_mm_s <= 1000.0):
            raise ValueError(f"[笛卡尔线速度] max_vel_mm_s={self.motor.max_vel_mm_s} mm/s 超出安全范围 [10, 1000]")
        if not (0.5 <= self.motor.max_ang_rad_s <= 15.0):
            raise ValueError(f"[笛卡尔角速度] max_ang_rad_s={self.motor.max_ang_rad_s} rad/s 超出安全范围 [0.5, 15]")
        if not (1.0 <= self.motor.max_dq_deg <= 45.0):
            raise ValueError(f"[单步角度限制] max_dq_deg={self.motor.max_dq_deg} deg/step 超出安全范围 [1, 45]")

        # 5. 预设姿态维度与限位校验
        if len(self.pose.ready_pose_deg) != 6:
            raise ValueError(f"[预设姿态] READY 姿态必须包含 6 关节角度，当前为 {len(self.pose.ready_pose_deg)} 项")
        if len(self.pose.home_pose_deg) != 6:
            raise ValueError(f"[预设姿态] HOME 姿态必须包含 6 关节角度，当前为 {len(self.pose.home_pose_deg)} 项")

        for idx, (ang, (lo, hi)) in enumerate(zip(self.pose.ready_pose_deg, limits_list), start=1):
            if not (lo <= ang <= hi):
                raise ValueError(f"[预设姿态] READY 姿态 J{idx}={ang}° 超出设置的关节限位 [{lo}°, {hi}°]")
        for idx, (ang, (lo, hi)) in enumerate(zip(self.pose.home_pose_deg, limits_list), start=1):
            if not (lo <= ang <= hi):
                raise ValueError(f"[预设姿态] HOME 姿态 J{idx}={ang}° 超出设置的关节限位 [{lo}°, {hi}°]")

        # 6. 视觉滤波参数校验
        if not (0.1 <= self.vision.pts_min_cutoff <= 10.0):
            raise ValueError(f"[视觉滤波] pts_min_cutoff={self.vision.pts_min_cutoff} Hz 超出有效范围 [0.1, 10.0]")
        if not (0.001 <= self.vision.pts_beta <= 1.0):
            raise ValueError(f"[视觉滤波] pts_beta={self.vision.pts_beta} 超出有效范围 [0.001, 1.0]")
        if not (0.5 <= self.vision.deadband_angle_deg <= 20.0):
            raise ValueError(f"[摇杆死区] deadband_angle_deg={self.vision.deadband_angle_deg}° 超出有效范围 [0.5, 20.0]")

        # 7. 灵巧手参数校验
        if not (50 <= self.hand.kP <= 1500):
            raise ValueError(f"[灵巧手] kP={self.hand.kP} 超出安全范围 [50, 1500]")
        if not (10 <= self.hand.kD <= 600):
            raise ValueError(f"[灵巧手] kD={self.hand.kD} 超出安全范围 [10, 600]")
        if not (50 <= self.hand.curr_lim <= 800):
            raise ValueError(f"[灵巧手电流] curr_lim={self.hand.curr_lim} mA 超出安全范围 [50, 800]")
        if not (0.1 <= self.hand.filter_min_cutoff <= 10.0):
            raise ValueError(f"[灵巧手滤波] filter_min_cutoff={self.hand.filter_min_cutoff} Hz 超出有效范围 [0.1, 10.0]")
        if self.hand.source_mode not in (0, 1, 2):
            raise ValueError(f"[灵巧手源] source_mode={self.hand.source_mode} 必须为 0(HAMER), 1(WORLD), 2(PSEUDO)")
        if self.hand.hand_type not in ("right", "left", "first"):
            raise ValueError(f"[灵巧手目标] hand_type={self.hand.hand_type} 必须为 'right', 'left' 或 'first'")

        return warnings

    @classmethod
    def load(cls, file_path: str | Path, validate: bool = True) -> "TeleopConfig":
        """从 .yaml, .json 或 .py 配置文件加载配置."""
        path = Path(file_path)
        cfg: Optional[TeleopConfig] = None
        if path.exists():
            suffix = path.suffix.lower()
            if suffix in (".yaml", ".yml"):
                try:
                    import yaml
                    with open(path, "r", encoding="utf-8") as f:
                        content = yaml.safe_load(f)
                        if isinstance(content, dict):
                            cfg = cls.from_dict(content)
                except Exception:
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            content = _simple_yaml_parse(f.read())
                            if isinstance(content, dict) and content:
                                cfg = cls.from_dict(content)
                    except Exception as err:
                        print(f"[配置警告] 加载 YAML 失败 ({err})，使用默认配置")
            elif suffix == ".json":
                try:
                    import json
                    with open(path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                        if isinstance(content, dict):
                            cfg = cls.from_dict(content)
                except Exception as e:
                    print(f"[配置警告] 加载 JSON 失败 ({e})，使用默认配置")

        if cfg is None:
            cfg = cls()

        if validate:
            warnings = cfg.validate()
            for w in warnings:
                print(f"[配置安全提示] \033[93m{w}\033[0m")

        return cfg

    def save_yaml(self, file_path: str | Path) -> None:
        """保存为 YAML 格式配置文件 (纯标准库支持)."""
        path = Path(file_path)
        try:
            import yaml
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                f.write(_simple_yaml_dump(self.to_dict()))


def _simple_yaml_parse(text: str) -> dict:
    """基于 Python 标准库的轻量级嵌套 YAML 解析器 (零依赖)."""
    import ast
    root: dict = {}
    stack: list = [(0, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#")[0].rstrip()
        if not line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        curr_dict = stack[-1][1]
        line = line.strip()
        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if not val:
                new_dict: dict = {}
                curr_dict[key] = new_dict
                stack.append((indent, new_dict))
            else:
                if val.lower() == "true":
                    parsed_val: Any = True
                elif val.lower() == "false":
                    parsed_val = False
                elif val.lower() in ("null", "none"):
                    parsed_val = None
                else:
                    try:
                        parsed_val = ast.literal_eval(val)
                    except Exception:
                        parsed_val = val
                curr_dict[key] = parsed_val
    return root


def _simple_yaml_dump(data: dict, indent: int = 0) -> str:
    """基于 Python 标准库的轻量级 YAML 序列化器 (零依赖)."""
    lines = []
    prefix = "  " * indent
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            lines.append(_simple_yaml_dump(v, indent + 1))
        elif isinstance(v, (list, tuple)):
            lines.append(f"{prefix}{k}: {list(v)}")
        elif isinstance(v, str):
            lines.append(f'{prefix}{k}: "{v}"')
        elif isinstance(v, bool):
            lines.append(f"{prefix}{k}: {'true' if v else 'false'}")
        else:
            lines.append(f"{prefix}{k}: {v}")
    return "\n".join(lines)


# 默认单例配置实例 (可以直接 import 使用)
DEFAULT_TELEOP_CONFIG = TeleopConfig.default()
