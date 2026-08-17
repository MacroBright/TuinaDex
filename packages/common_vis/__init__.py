"""packages.common_vis — 跨子系统共享视觉工具（占位骨架）.

跨 Arm/Hand/顶层应用共享的视觉工具（图像处理、相机标定、关键点检测等）。
当前阶段: 不复制任何 submodule 代码——通过 sys.path 让两个 submodule
内部的纯函数模块（Arm-robot_VLA/scripts/mujoco_sim.camera_server,
Leap_Hand/python/gesture_mapping.camera/hand_tracker/hamer_3d 等）可被顶层应用直接复用。

具体共用工具的提取与命名空间统一, 留待 Phase A 启动后根据实际调用频率决定.
"""
__all__: list[str] = []
__version__ = "0.1.0"
