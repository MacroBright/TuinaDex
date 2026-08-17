#!/usr/bin/env bash
#
# bump_submodules.sh — 一键 bump 两个 submodule 引用到远端最新 commit.
#
# 用法:
#   bash tools/bump_submodules.sh               # 两个都 bump
#   bash tools/bump_submodules.sh Arm-robot_VLA # 仅 Arm
#   bash tools/bump_submodules.sh Leap_Hand     # 仅 Hand
#
# 典型 workflow:
#   1. 在 Arm-robot_VLA 内开发 + commit + push
#   2. 回到 TuinaDex 顶层, 跑本脚本 → 顶层自动记录新 commit
#   3. git add Arm-robot_VLA && git commit -m "chore(submodule): bump arm"
#   4. git push origin main
#
# 注意: 需要在 TuinaDex 顶层仓库根目录运行.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

TARGETS=("Arm-robot_VLA" "Leap_Hand")
if [[ $# -gt 0 ]]; then
    TARGETS=("$@")
fi

for sub in "${TARGETS[@]}"; do
    if [[ ! -d "$sub" ]]; then
        echo "[skip] $sub not a directory"
        continue
    fi
    echo "==> bump $sub"
    git submodule update --remote "$sub"
    echo "    new commit: $(git -C "$sub" log --oneline -1)"
    echo "    branch:     $(git -C "$sub" symbolic-ref --short HEAD 2>/dev/null || echo detached)"
done

echo ""
echo "==> review:"
git diff --stat
echo ""
echo "==> next: git add . && git commit -m \"chore(submodule): bump\" && git push"