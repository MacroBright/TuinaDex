#!/usr/bin/env python3
"""TuinaDex — 机械臂-灵巧手协同视觉遥操统一启动入口.

用法示例:
  # 真机协同遥操 (默认使用 can0 + /dev/ttyUSB0):
  python run_teleop.py --iface can0 -y

  # 纯软件空跑测试 (无需连接物理臂或灵巧手):
  python run_teleop.py --no-drive-arm --no-drive-hand
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
for p in [ROOT_DIR, ROOT_DIR / "Arm-robot_VLA", ROOT_DIR / "Leap_Hand" / "python", ROOT_DIR / "Co_Teleop"]:
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from Co_Teleop.pipeline.unified_teleop import main

if __name__ == "__main__":
    main()
