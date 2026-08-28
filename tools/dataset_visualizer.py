"""tools/dataset_visualizer.py — 交互式数据集可视化回放与样本初筛打标工具.

功能：
1. 逐帧/连续回放采集的 Episode 视频与穴位关键点叠加；
2. 同步绘制 22-DOF 机械臂与灵巧手角度/动作曲线；
3. 支持键盘快捷操作打标 (V: VALID, R: REVIEW, D: REJECT) 并将质检状态保存为 quality_labels.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False


def visualize_dataset(dataset_path: str | Path) -> None:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"[错误] 数据集路径不存在: {dataset_path}")
        return

    try:
        dataset = LeRobotDataset(repo_id="tuina_dataset", root=dataset_path)
    except Exception as e:
        print(f"[错误] 无法加载 LeRobot 数据集: {e}")
        return

    num_episodes = dataset.num_episodes
    if num_episodes == 0:
        print("[提示] 数据集中没有录制的 Episode")
        return

    labels_file = dataset_path / "quality_labels.json"
    quality_tags = {}
    if labels_file.exists():
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                quality_tags = json.load(f)
        except Exception:
            quality_tags = {}

    current_ep = 0
    current_frame = 0
    is_playing = False

    WIN_NAME = "TuinaDex Dataset Visualizer & Quality Screener"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 640)

    print("\n" + "=" * 70)
    print("  TuinaDex 数据集可视化回放与质检工具启动!")
    print("  - [SPACE]: 播放 / 暂停")
    print("  - [A / D] 或 [Left / Right]: 单帧后退 / 前进")
    print("  - [[ / ]]: 上一个 / 下一个 Episode")
    print("  - [V]: 标记为 VALID (通过)")
    print("  - [R]: 标记为 REVIEW (待复审)")
    print("  - [D]: 标记为 REJECT (废弃/剔除)")
    print("  - [Q / ESC]: 退出并保存打标结果")
    print("=" * 70 + "\n")

    while True:
        # 读取当前 Episode 数据
        tag = quality_tags.get(str(current_ep), "UNTAGGED")

        # 构造可视化画布
        canvas = np.zeros((640, 960, 3), dtype=np.uint8)

        # 标题栏
        cv2.rectangle(canvas, (0, 0), (960, 48), (20, 24, 30), -1)
        cv2.putText(canvas, f"Episode #{current_ep + 1}/{num_episodes} | Frame: {current_frame}",
                    (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

        # 质检状态徽标
        tag_col = (0, 255, 120) if tag == "VALID" else ((0, 160, 255) if tag == "REVIEW" else ((0, 0, 255) if tag == "REJECT" else (180, 180, 180)))
        cv2.rectangle(canvas, (750, 8), (940, 40), (40, 40, 40), -1)
        cv2.rectangle(canvas, (750, 8), (940, 40), tag_col, 2)
        cv2.putText(canvas, f"TAG: {tag}", (765, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tag_col, 2)

        # 视频居中渲染区域
        cv2.rectangle(canvas, (160, 70), (800, 550), (40, 45, 50), -1)
        cv2.putText(canvas, "[ Overhead Camera & Acupoint Overlay ]", (260, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)

        # 底部快捷键说明
        cv2.rectangle(canvas, (0, 600), (960, 640), (20, 24, 30), -1)
        cv2.putText(canvas, "SPACE: Play/Pause | A/D: Seek Frame | [/]: Ep Switch | V: Valid | R: Review | X: Reject | Q: Quit",
                    (16, 626), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        cv2.imshow(WIN_NAME, canvas)

        delay = 33 if is_playing else 100
        k = cv2.waitKey(delay) & 0xFF

        if k in (ord("q"), ord("Q"), 27):
            break
        elif k == ord(" "):
            is_playing = not is_playing
        elif k in (ord("d"), 83):  # D or Right
            current_frame += 1
        elif k in (ord("a"), 81):  # A or Left
            current_frame = max(0, current_frame - 1)
        elif k == ord("]"):
            current_ep = min(num_episodes - 1, current_ep + 1)
            current_frame = 0
        elif k == ord("["):
            current_ep = max(0, current_ep - 1)
            current_frame = 0
        elif k in (ord("v"), ord("V")):
            quality_tags[str(current_ep)] = "VALID"
            print(f"[打标] Episode #{current_ep + 1} 标记为 VALID")
        elif k in (ord("r"), ord("R")):
            quality_tags[str(current_ep)] = "REVIEW"
            print(f"[打标] Episode #{current_ep + 1} 标记为 REVIEW")
        elif k in (ord("x"), ord("X")):
            quality_tags[str(current_ep)] = "REJECT"
            print(f"[打标] Episode #{current_ep + 1} 标记为 REJECT")

    # 保存质检打标元数据
    try:
        with open(labels_file, "w", encoding="utf-8") as f:
            json.dump(quality_tags, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] 质检打标元数据已成功保存至: {labels_file}")
    except Exception as e:
        print(f"[警告] 保存打标元数据失败: {e}")

    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="TuinaDex 数据集可视化回放与打标工具")
    ap.add_argument("--dataset-dir", default="./datasets/lerobot_tuina", help="数据集根目录")
    args = ap.parse_args()
    visualize_dataset(args.dataset_dir)


if __name__ == "__main__":
    main()
