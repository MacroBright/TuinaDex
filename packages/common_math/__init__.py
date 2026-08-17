"""packages.common_math — 跨子系统共享数学工具（占位骨架）.

跨 Arm/Hand/顶层应用共享的数学工具（旋转、IK、四元数、滤波等）。
当前阶段: 不复制任何 submodule 代码——通过 sys.path 让两个 submodule
内部的纯函数模块（gesture_mapping.wrist_tracker / leap_hand_utils
等）可被顶层应用直接复用。

具体共用工具的提取与命名空间统一, 留待 Phase A 启动后根据实际调用频率决定.
"""
__all__: list[str] = []
__version__ = "0.1.0"
