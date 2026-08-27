#!/usr/bin/env python3
"""Co_Teleop 协同视觉遥操统一启动入口 (One-Click Launcher).

用法示例:
  # 真机协同遥操 (默认使用 can0 + /dev/ttyUSB0):
  python run_teleop.py --iface can0 -y

  # 空跑测试模式 (无需连接任何硬件，仅测试相机与算法):
  python run_teleop.py --no-drive-arm --no-drive-hand
"""
import os
import sys
from pathlib import Path

# 保证当前工程根目录与各子模块在 sys.path 中
CO_TELEOP_DIR = Path(__file__).resolve().parent
REPO_ROOT = CO_TELEOP_DIR.parent if (CO_TELEOP_DIR.parent / "Arm-robot_VLA").exists() else CO_TELEOP_DIR

for p in [REPO_ROOT, REPO_ROOT / "Arm-robot_VLA", REPO_ROOT / "Leap_Hand" / "python", CO_TELEOP_DIR]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from Co_Teleop.pipeline.unified_teleop import main

if __name__ == "__main__":
    main()
