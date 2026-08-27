# -*- coding: utf-8 -*-
"""单张图片端到端测试脚本（不依赖摄像头）。

用法：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd Scripts/tools
    python test_single.py /path/to/test.jpg
    python test_single.py /path/to/test.jpg --det ../../Model/RTMDet_Tiny.om --pose ../../Model/RTMPose_s.om
    python test_single.py /path/to/test.jpg --no-det   # 跳过检测，直接关键点

输出：
    - 控制台打印检测框、37 个关键点（名称/坐标/置信度）、耗时汇总
    - ../runs/<图片名>_result.jpg  可视化结果
    - ../runs/<图片名>.json        结构化结果
"""

import os
import sys
import time
import json
import argparse
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "libs"))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import cv2
import numpy as np

from acl_utils import init_acl, release_acl
from det_infer import DetInfer
from pose_infer import PoseInfer
from visualize import draw_results
from acupoints import ACUPOINT_NAMES_CN, ACUPOINT_NAMES_EN
from preprocess import letterbox_resize, inv_letterbox, DET_INPUT_HW

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test")

DEFAULT_DET = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Model", "RTMDet_Tiny.om"))
DEFAULT_POSE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Model", "RTMPose_s.om"))
DEFAULT_OUT = "/home/HwHiAiUser/Desktop/AcuPointDet/Output"


def find_image(path=None):
    if path:
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f"图片不存在: {path}")
    for name in ("test.jpg", "test.jpeg", "test.png", "test.bmp",
                 "sample.jpg", "sample.png"):
        for d in (SCRIPT_DIR, os.path.join(SCRIPT_DIR, ".."),
                  os.path.join(SCRIPT_DIR, "..", "TestData"),
                  os.path.join(SCRIPT_DIR, "..", "..")):
            p = os.path.normpath(os.path.join(d, name))
            if os.path.isfile(p):
                return p
    return None


def print_boxes(boxes):
    print("\n========== 检测结果 ==========")
    if boxes is None or len(boxes) == 0:
        print("  未检测到背部区域（boxes 为空）")
        return
    print(f"  共检测到 {len(boxes)} 个框:")
    for i, b in enumerate(boxes):
        print(f"    [{i}] x1={b[0]:.1f} y1={b[1]:.1f} x2={b[2]:.1f} y2={b[3]:.1f}  "
              f"w={b[2]-b[0]:.1f} h={b[3]-b[1]:.1f}  score={b[4]:.4f}")


def print_keypoints(kpts, kpt_thr=0.3):
    print("\n========== 关键点（37 个穴位）==========")
    if kpts is None:
        print("  无关键点输出（检测框为空或关键点推理未执行）")
        return
    n_valid = int(np.sum((kpts[:, 2] >= kpt_thr) & (kpts[:, 0] >= 0)))
    print(f"  有效关键点: {n_valid}/37  (阈值={kpt_thr})")
    print(f"  {'idx':>3}  {'英文名':<16}  {'中文名':<8}  {'x':>8}  {'y':>8}  {'conf':>6}  状态")
    print("  " + "-" * 70)
    for i in range(kpts.shape[0]):
        x, y, c = float(kpts[i, 0]), float(kpts[i, 1]), float(kpts[i, 2])
        ok = (c >= kpt_thr) and (x >= 0)
        status = "OK" if ok else "--"
        x_str = f"{x:8.1f}" if x >= 0 else "    N/A"
        y_str = f"{y:8.1f}" if y >= 0 else "    N/A"
        print(f"  {i:>3}  {ACUPOINT_NAMES_EN[i]:<16}  {ACUPOINT_NAMES_CN[i]:<8}  "
              f"{x_str}  {y_str}  {c:6.3f}  {status}")


def print_summary(img, result, kpt_thr=0.3):
    h, w = img.shape[:2]
    print("\n========== 汇总 ==========")
    print(f"  图像尺寸    : {w} x {h}")
    print(f"  检测耗时    : {result['det_ms']:.2f} ms")
    print(f"  关键点耗时  : {result['pose_ms']:.2f} ms")
    print(f"  端到端耗时  : {result['total_ms']:.2f} ms")
    if result["keypoints"] is not None:
        confs = result["keypoints"][:, 2]
        n_valid = int(np.sum((confs >= kpt_thr) & (result["keypoints"][:, 0] >= 0)))
        print(f"  有效关键点  : {n_valid}/37")
        print(f"  关键点置信度: mean={confs.mean():.3f} max={confs.max():.3f} "
              f"min={confs.min():.3f}")
        xy = result["keypoints"][:, :2]
        valid_xy = xy[(confs >= kpt_thr) & (xy[:, 0] >= 0)]
        if len(valid_xy) > 0:
            print(f"  坐标范围    : x[{valid_xy[:,0].min():.0f}~{valid_xy[:,0].max():.0f}] "
                  f"y[{valid_xy[:,1].min():.0f}~{valid_xy[:,1].max():.0f}]")
            in_range = np.all((valid_xy[:, 0] >= 0) & (valid_xy[:, 0] < w) &
                              (valid_xy[:, 1] >= 0) & (valid_xy[:, 1] < h))
            print(f"  坐标在原图内: {'是' if in_range else '否（存在越界点！）'}")


def save_outputs(out_dir, img_orig, img_det_input, img_pose_input,
                 result, kpt_thr=0.3):
    """保存全部输出到 out_dir（已是时间命名的子文件夹）。

    文件：
      00_original.jpg      原图
      01_det_input.jpg     letterbox 后输入 RTMDet 的图 (416×416)
      02_pose_input.jpg    RTMDet 裁剪后输入 RTMPose 的图 (256×192)
      03_result.jpg        最终可视化结果（原图上绘制）
      result.json          结构化结果
    """
    os.makedirs(out_dir, exist_ok=True)
    boxes = result["boxes"]
    kpts = result["keypoints"]

    cv2.imwrite(os.path.join(out_dir, "00_original.jpg"), img_orig)
    cv2.imwrite(os.path.join(out_dir, "01_det_input.jpg"), img_det_input)
    if img_pose_input is not None:
        cv2.imwrite(os.path.join(out_dir, "02_pose_input.jpg"), img_pose_input)

    canvas = draw_results(img_orig, boxes, kpts, conf_thr=kpt_thr, draw_names=True)

    vis_path = os.path.join(out_dir, "03_result.jpg")
    cv2.imwrite(vis_path, canvas)

    payload = {
        "image_size": {"h": int(img_orig.shape[0]), "w": int(img_orig.shape[1])},
        "boxes": boxes.tolist() if boxes is not None else [],
        "keypoints": None,
        "elapsed_ms": {
            "det": round(result["det_ms"], 2),
            "pose": round(result["pose_ms"], 2),
            "total": round(result["total_ms"], 2),
        },
    }
    if kpts is not None:
        payload["keypoints"] = [
            {"index": i, "name_en": ACUPOINT_NAMES_EN[i], "name_cn": ACUPOINT_NAMES_CN[i],
             "x": round(float(kpts[i, 0]), 2),
             "y": round(float(kpts[i, 1]), 2),
             "score": round(float(kpts[i, 2]), 4)}
            for i in range(kpts.shape[0])
        ]
    json_path = os.path.join(out_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return vis_path, json_path


def build_argparser():
    p = argparse.ArgumentParser(description="AcuPointDet 单图测试（昇腾 310B OM）")
    p.add_argument("image", nargs="?", default=None,
                   help="图片路径（省略则在默认位置查找）")
    p.add_argument("--det", default=DEFAULT_DET, help="检测 om 路径")
    p.add_argument("--pose", default=DEFAULT_POSE, help="关键点 om 路径")
    p.add_argument("--output", "-o", default=DEFAULT_OUT, help="输出目录")
    p.add_argument("--device", type=int, default=0, help="昇腾 device id")
    p.add_argument("--score-thr", type=float, default=0.25)
    p.add_argument("--iou-thr", type=float, default=0.65)
    p.add_argument("--kpt-thr", type=float, default=0.3)
    p.add_argument("--no-det", action="store_true",
                   help="跳过目标检测，直接把整图仿射到 256×192 喂给 RTMPose 做关键点估计")
    p.add_argument("--no-save", action="store_true", help="不保存可视化与 JSON")
    return p


def main():
    args = build_argparser().parse_args()

    img_path = find_image(args.image)
    if img_path is None:
        print("未指定图片，且在默认位置未找到测试图片。")
        print("用法: python test_single.py /path/to/test.jpg")
        sys.exit(1)

    checks = [("关键点模型", args.pose)]
    if not args.no_det:
        checks.insert(0, ("检测模型", args.det))
    for label, p in checks:
        if not os.path.isfile(p):
            print(f"{label}不存在: {p}")
            sys.exit(1)

    print(f"图片    : {img_path}")
    print(f"检测模型: {args.det}" + ("  (--no-det 已关闭)" if args.no_det else ""))
    print(f"关键点  : {args.pose}")
    print(f"输出目录: {args.output}")

    img = cv2.imread(img_path)
    if img is None:
        print(f"读取图片失败（cv2.imread 返回 None）: {img_path}")
        sys.exit(1)
    h, w = img.shape[:2]
    print(f"原始图像尺寸: {w} x {h}")

    target_hw = DET_INPUT_HW
    if (h, w) != target_hw:
        img_padded, r, (pad_left, pad_top) = letterbox_resize(
            img, target_hw, pad_value=128)
        print(f"尺寸不匹配，已等比缩放 + 灰色填充: "
              f"{w}x{h} -> {target_hw[1]}x{target_hw[0]}  "
              f"(r={r:.4f}, pad_left={pad_left}, pad_top={pad_top})")
    else:
        img_padded = img
        r, pad_left, pad_top = 1.0, 0, 0
        print("尺寸匹配，无需填充")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output, timestamp)
    print(f"输出文件夹: {out_dir}")

    det = DetInfer(args.det, device_id=args.device) if not args.no_det else None
    pose = PoseInfer(args.pose, device_id=args.device)
    try:
        t0 = time.perf_counter()
        if det is not None:
            boxes = det.detect(
                img_padded, score_thr=args.score_thr,
                iou_thr=args.iou_thr, max_det=1)
            t1 = time.perf_counter()

            if boxes.shape[0] > 0:
                print(f"[debug] 模型输出原始框(480x640空间): "
                      f"{['({:.1f},{:.1f})-({:.1f},{:.1f}) s={:.3f}'.format(*b) for b in boxes]}")
                print(f"[debug] letterbox: r={r:.4f} pad=({pad_left},{pad_top}) "
                      f"缩放后图在原canvas内偏移=({pad_left},{pad_top})")
                boxes = inv_letterbox(boxes, r, pad_left, pad_top)
                boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
                boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
                boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
                boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        else:
            boxes = np.zeros((0, 5), dtype=np.float32)
            t1 = time.perf_counter()

        kpts = None
        best_box = None
        img_pose_input = None
        if det is not None and boxes.shape[0] > 0:
            best_box = boxes[0]
            kpts, _, _, _, img_pose_input = pose.estimate(
                img, best_box[:4], kpt_thr=args.kpt_thr, return_crop=True)
        elif det is None:
            # 跳过检测：用整图 bbox 直接做关键点估计
            full_box = [0.0, 0.0, float(w - 1), float(h - 1)]
            best_box = np.asarray(full_box + [1.0], dtype=np.float32)
            kpts, _, _, _, img_pose_input = pose.estimate(
                img, full_box, kpt_thr=args.kpt_thr, return_crop=True)
        t2 = time.perf_counter()

        result = {
            "boxes": boxes,
            "keypoints": kpts,
            "best_box": best_box,
            "det_ms": (t1 - t0) * 1000,
            "pose_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
        }

        print_boxes(boxes)
        print_keypoints(kpts, kpt_thr=args.kpt_thr)
        print_summary(img, result, kpt_thr=args.kpt_thr)

        if not args.no_save:
            vis_path, json_path = save_outputs(
                out_dir, img, img_padded, img_pose_input, result,
                kpt_thr=args.kpt_thr)
            print(f"\n输出文件夹: {out_dir}")
            print(f"  00_original.jpg   原图")
            print(f"  01_det_input.jpg  {'(跳过检测)' if det is None else 'RTMDet 输入 (letterbox)'}")
            if img_pose_input is not None:
                print(f"  02_pose_input.jpg RTMPose 输入 (仿射 256×192)")
            print(f"  03_result.jpg     可视化结果")
            print(f"  result.json       结构化结果")

        print("\n测试完成 ✓")
    finally:
        if det is not None:
            det.close()
        pose.close()


if __name__ == "__main__":
    main()