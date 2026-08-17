#!/usr/bin/env bash
#
# setup_dev.sh — 一键安装 TuinaDex 开发环境 (含两个 submodule).
#
# 前置: 已 git clone --recurse-submodules
#
# 安装步骤:
#   1. submodule 初始化 (若未)
#   2. Arm-robot_VLA/lerobot_robot_massage → pip install -e (compat 模式)
#   3. Leap_Hand/python (作为 sys.path 注入, 不需 pip install)
#   4. 顶层 TuinaDex → pip install -e .
#
# 用法:
#   bash scripts/setup_dev.sh

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "==> [1/4] submodule 状态"
git submodule status
echo ""

echo "==> [2/4] Arm-robot_VLA editable install"
ARM_DIR="$(pwd)/Arm-robot_VLA"
pip install -e "$ARM_DIR/lerobot_robot_massage" --config-settings editable_mode=comm
echo ""

echo "==> [3/4] Leap_Hand 不需 pip install (sys.path 注入)"
LEAP_HAND_PKG_DIR="$(pwd)/Leap_Hand/python"
echo "        添加到 PYTHONPATH: $LEAP_HAND_PKG_DIR"
echo ""

echo "==> [4/4] 顶层 TuinaDex editable install"
pip install -e ".[vla,hand,arm,dev]"
echo ""

echo "==> 验证"
python -c "
import sys
sys.path.insert(0, '$LEAP_HAND_PKG_DIR')

from apps.massage import MassageRobotV2
print('[OK] apps.massage.MassageRobotV2')

import lerobot_robot_massage
from lerobot_robot_massage.massage_robot import MassageRobot
print('[OK] lerobot_robot_massage.MassageRobot')

from leap_hand_utils.dynamixel_client import DynamixelClient
print('[OK] leap_hand_utils.DynamixelClient')
"

echo ""
echo "==> 修 .pth (setuptools compat 模式 bug 绕过)"
PTH_FILE="$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)/__editable__.lerobot_robot_massage-0.1.0.pth"
if [[ -f "$PTH_FILE" ]]; then
    echo "    修改前: $(cat "$PTH_FILE")"
    echo "$ARM_DIR" > "$PTH_FILE"
    echo "    修改后: $(cat "$PTH_FILE")"
else
    echo "    (跳过) $PTH_FILE 不存在 (pip install 可能没跑)"
fi