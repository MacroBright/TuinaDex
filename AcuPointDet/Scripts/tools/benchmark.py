#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""管线延迟分布测量脚本（t0~t5 分段计时）。

运行环境（香橙派 AI Pro / 昇腾 310B）：
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    cd ~/Desktop/AcuPointDet/Scripts/tools
    python benchmark.py ../../TestData/test1.jpg -n 10
    python benchmark.py ../../TestData -n 20       # 目录下全部 jpg
    python benchmark.py a.jpg b.jpg -n 5           # 多张图

分段定义（对应部署讨论的 t0~t5）：
    t0 读图           : cv2.imread（模拟取帧）
    t1 预处理(检测)   : letterbox 缩放填充 + 归一化，产出 RTMDet 输入
    t2 检测推理       : h2d 拷贝 + acl.mdl.execute(RTMDet) + d2h 拷贝（细项分别输出）
    t3 后处理(检测)   : decode + 分数过滤 + NMS + 逆 letterbox（host 侧）
    t4 关键点推理     : 裁剪 + 仿射 + 归一化 + h2d + acl.mdl.execute(RTMPose) + d2h（细项分别输出）
    t5 关键点后处理   : 逆仿射回原图

输出：每个图片一个分段表 + 全局聚合表（平均/最大/最小/占比），
      --json 可把全部数据存到 Output/bench_<时间戳>.json。
"""

import os
import sys
import glob
import time
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "libs"))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import cv2
import numpy as np

from acl_utils import init_acl, release_acl
from det_infer import DetInfer, DET_OUTPUT_SHAPE
from pose_infer import PoseInfer, POSE_OUTPUT_SHAPE, DEFAULT_KPT_THR
from postprocess import postprocess_rtmdet, postprocess_rtmpose
from preprocess import (letterbox_resize, preprocess_rtmdet_raw,
                        crop_and_affine, DET_INPUT_HW, POSE_EXPAND)

DEFAULT_DET = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Model", "RTMDet_Tiny.om"))
DEFAULT_POSE = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Model", "RTMPose_s.om"))
DEFAULT_OUT = "/home/HwHiAiUser/Desktop/AcuPointDet/Output"

SEG_KEYS = ["t0_read", "t1_pre_det", "t2_det_infer", "t3_post_det",
            "t4_pose_infer", "t5_post_pose"]
SEG_NAMES = {
    "t0_read":       "t0 读图",
    "t1_pre_det":    "t1 预处理(检测)",
    "t2_det_infer":  "t2 检测推理",
    "t3_post_det":   "t3 后处理(检测)",
    "t4_pose_infer": "t4 关键点推理",
    "t5_post_pose":  "t5 关键点后处理",
}
SUBTIME_KEYS = [
    ("det_h2d",  "det h2d"),
    ("det_exec", "det acl.mdl.execute"),
    ("det_d2h",  "det d2h"),
    ("pose_crop", "pose 裁剪+仿射+归一化"),
    ("pose_h2d", "pose h2d"),
    ("pose_exec", "pose acl.mdl.execute"),
    ("pose_d2h", "pose d2h"),
]


def discover_images(paths):
    """把命令行参数展开为图片列表（支持文件与目录）。"""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "*.jpg"))) +
                         sorted(glob.glob(os.path.join(p, "*.jpeg"))) +
                         sorted(glob.glob(os.path.join(p, "*.png"))))
        else:
            files.append(p)
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def run_once(det, pose, img_path, args):
    """对单张图片执行一次端到端推理，返回 (seg 字典, sub 字典, 状态字符串)。"""
    seg = {k: 0.0 for k in SEG_KEYS}
    sub = {k: 0.0 for k, _ in SUBTIME_KEYS}
    status = "ok"

    t = time.perf_counter()
    img = cv2.imread(img_path)
    if img is None:
        return seg, sub, "read_fail"
    seg["t0_read"] = (time.perf_counter() - t) * 1000

    h, w = img.shape[:2]
    target_hw = DET_INPUT_HW

    t = time.perf_counter()
    if (h, w) != target_hw:
        img_padded, r, (pad_left, pad_top) = letterbox_resize(
            img, target_hw, pad_value=128)
    else:
        img_padded, r, pad_left, pad_top = img, 1.0, 0, 0
    inp_det = preprocess_rtmdet_raw(img_padded, target_hw=target_hw)
    seg["t1_pre_det"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    outs = det.model.infer([inp_det])
    seg["t2_det_infer"] = (time.perf_counter() - t) * 1000
    sub["det_h2d"] = det.model.last_h2d_ms
    sub["det_exec"] = det.model.last_exec_ms
    sub["det_d2h"] = det.model.last_d2h_ms
    dets = np.asarray(outs[0]).reshape(DET_OUTPUT_SHAPE)[0]

    t = time.perf_counter()
    boxes = postprocess_rtmdet(dets, score_thr=args.score_thr,
                               iou_thr=args.iou_thr, max_det=args.max_det,
                               img_hw=target_hw)
    if boxes.shape[0] > 0 and (h, w) != target_hw:
        boxes = _inv_letterbox_boxes(boxes, r, pad_left, pad_top, (h, w))
    seg["t3_post_det"] = (time.perf_counter() - t) * 1000

    if boxes.shape[0] > 0:
        box = boxes[0, :4]

        t = time.perf_counter()
        inp_pose, M, center, scale, crop = crop_and_affine(
            img, box, out_wh=pose.out_wh, expand=POSE_EXPAND)
        sub["pose_crop"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        outs = pose.model.infer([inp_pose])
        seg["t4_pose_infer"] = (time.perf_counter() - t) * 1000
        sub["pose_h2d"] = pose.model.last_h2d_ms
        sub["pose_exec"] = pose.model.last_exec_ms
        sub["pose_d2h"] = pose.model.last_d2h_ms
        kpts = np.asarray(outs[0]).reshape(POSE_OUTPUT_SHAPE)[0]

        t = time.perf_counter()
        kpts_orig = postprocess_rtmpose(kpts, M, kpt_thr=args.kpt_thr)
        seg["t5_post_pose"] = (time.perf_counter() - t) * 1000

        n_valid = int(np.sum((kpts_orig[:, 2] >= args.kpt_thr) &
                             (kpts_orig[:, 0] >= 0)))
        if n_valid == 0:
            status = "no_valid_kpt"
    else:
        status = "no_box"

    return seg, sub, status


def _inv_letterbox_boxes(boxes, r, pad_left, pad_top, orig_hw):
    """把 letterbox 空间的框坐标映射回原图。boxes: (N,5)[x1,y1,x2,y2,score]。"""
    oh, ow = orig_hw
    out = boxes.copy()
    out[:, 0] = np.clip((out[:, 0] - pad_left) / r, 0, ow - 1)
    out[:, 2] = np.clip((out[:, 2] - pad_left) / r, 0, ow - 1)
    out[:, 1] = np.clip((out[:, 1] - pad_top) / r, 0, oh - 1)
    out[:, 3] = np.clip((out[:, 3] - pad_top) / r, 0, oh - 1)
    return out


def stat(cols):
    """cols: list[float] -> (avg, max, min)。空列表返回 (0,0,0)。"""
    if len(cols) == 0:
        return 0.0, 0.0, 0.0
    a = np.asarray(cols, dtype=np.float64)
    return float(a.mean()), float(a.max()), float(a.min())


def _aggregate(all_seg, all_sub):
    """合并 seg/sub 统计，返回 (rows, totals, sub_rows, total_exec)。"""
    rows = []
    for k in SEG_KEYS:
        avg, mx, mn = stat(all_seg[k])
        rows.append((SEG_NAMES[k], avg, mx, mn))
    total_avg = sum(r[1] for r in rows)
    total_max = sum(r[2] for r in rows)
    total_min = sum(r[3] for r in rows)

    sub_rows = []
    for k, label in SUBTIME_KEYS:
        avg, _, _ = stat(all_sub.get(k, []))
        if avg > 0:
            sub_rows.append((label, avg))

    det_exec = stat(all_sub.get("det_exec", []))[0]
    pose_exec = stat(all_sub.get("pose_exec", []))[0]
    total_exec = det_exec + pose_exec
    return rows, (total_avg, total_max, total_min), sub_rows, total_exec


def print_seg_table(rows, totals, sub_rows, total_exec, title=None):
    total_avg, total_max, total_min = totals
    if title:
        print(f"== {title} ==")
    width = max(len(r[0]) for r in rows)
    print(f"  {'segment':<{width}}  {'平均(ms)':>9}  {'最大(ms)':>9}  "
          f"{'最小(ms)':>9}  {'占比':>7}")
    print("  " + "-" * (width + 50))
    for name, avg, mx, mn in rows:
        pct = avg / total_avg * 100 if total_avg > 0 else 0.0
        print(f"  {name:<{width}}  {avg:9.2f}  {mx:9.2f}  {mn:9.2f}  {pct:6.1f}%")
    print("  " + "-" * (width + 50))
    fps = 1000.0 / total_avg if total_avg > 0 else 0.0
    print(f"  {'端到端合计':<{width}}  {total_avg:9.2f}  {total_max:9.2f}  "
          f"{total_min:9.2f}   (≈{fps:.1f} FPS)")
    if total_exec > 0 and total_avg > 0:
        print(f"  算子执行(acl.mdl.execute)合计: {total_exec:.2f} ms "
              f"({total_exec / total_avg * 100:.1f}% of 端到端)")
    if sub_rows:
        print("  细项拆分:")
        for label, avg in sub_rows:
            print(f"    {label}: {avg:.2f} ms")


def build_argparser():
    p = argparse.ArgumentParser(description="AcuPointDet 延迟分布测量（昇腾 310B OM）")
    p.add_argument("images", nargs="+", help="图片文件或目录（目录自动展开全部 jpg/png）")
    p.add_argument("--det", default=DEFAULT_DET, help="检测 om 路径")
    p.add_argument("--pose", default=DEFAULT_POSE, help="关键点 om 路径")
    p.add_argument("-n", "--iters", type=int, default=5, help="每张图片重复次数")
    p.add_argument("--warmup", type=int, default=1, help="预热次数（不计入统计）")
    p.add_argument("--device", type=int, default=0, help="昇腾 device id")
    p.add_argument("--score-thr", type=float, default=0.25)
    p.add_argument("--iou-thr", type=float, default=0.65)
    p.add_argument("--kpt-thr", type=float, default=DEFAULT_KPT_THR)
    p.add_argument("--max-det", type=int, default=1, help="最多保留检测框数")
    p.add_argument("--json", action="store_true", help="保存完整数据到 Output/bench_<时间戳>.json")
    return p


def main():
    args = build_argparser().parse_args()

    if args.iters < 1:
        print("--iters 必须 >= 1")
        sys.exit(1)
    for label, p in (("检测模型", args.det), ("关键点模型", args.pose)):
        if not os.path.isfile(p):
            print(f"{label}不存在: {p}")
            sys.exit(1)

    images = discover_images(args.images)
    if len(images) == 0:
        print("未找到任何图片文件")
        sys.exit(1)

    print(f"图片列表 : {len(images)} 张")
    for p in images:
        print(f"  - {p}")
    print(f"检测模型 : {args.det}")
    print(f"关键点   : {args.pose}")
    print(f"每张重复 : {args.iters} 次 (预热 {args.warmup} 次)")
    print()

    all_seg = {k: [] for k in SEG_KEYS}
    all_sub = {k: [] for k, _ in SUBTIME_KEYS}
    per_image = []

    det = DetInfer(args.det, device_id=args.device)
    pose = PoseInfer(args.pose, device_id=args.device)
    try:
        for img_path in images:
            print(f"===== {os.path.basename(img_path)} =====")
            img_seg = {k: [] for k in SEG_KEYS}
            img_sub = {k: [] for k, _ in SUBTIME_KEYS}
            for i in range(args.iters + args.warmup):
                seg, sub, status = run_once(det, pose, img_path, args)
                if i < args.warmup:
                    if i == 0:
                        print(f"  预热: total={sum(seg.values()):.2f} ms status={status}")
                    continue
                for k in SEG_KEYS:
                    all_seg[k].append(seg[k])
                    img_seg[k].append(seg[k])
                for k in all_sub:
                    all_sub[k].append(sub[k])
                    img_sub[k].append(sub[k])

            rows, totals, sub_rows, total_exec = _aggregate(img_seg, img_sub)
            print_seg_table(rows, totals, sub_rows, total_exec)
            print()
            per_image.append({
                "image": img_path,
                "status": status,
                "segments_ms": {k: [round(v, 3) for v in img_seg[k]] for k in SEG_KEYS},
            })

        rows, totals, sub_rows, total_exec = _aggregate(all_seg, all_sub)
        print_seg_table(rows, totals, sub_rows, total_exec, title="全局聚合（所有图片/轮次）")
    finally:
        det.close()
        pose.close()

    if args.json:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(DEFAULT_OUT, timestamp)
        os.makedirs(out_dir, exist_ok=True)
        payload = {
            "timestamp": timestamp,
            "args": {
                "det": args.det, "pose": args.pose,
                "iters": args.iters, "warmup": args.warmup,
                "score_thr": args.score_thr, "iou_thr": args.iou_thr,
                "kpt_thr": args.kpt_thr, "max_det": args.max_det,
            },
            "per_image": per_image,
            "global": {
                "segments_ms": {
                    k: {"avg": round(stat(all_seg[k])[0], 3),
                        "max": round(stat(all_seg[k])[1], 3),
                        "min": round(stat(all_seg[k])[2], 3)}
                    for k in SEG_KEYS},
                "subtime_ms": {
                    k: round(stat(all_sub.get(k, []))[0], 3)
                    for k, _ in SUBTIME_KEYS},
                "end2end_avg_ms": round(
                    sum(stat(all_seg[k])[0] for k in SEG_KEYS), 3),
            },
        }
        jpath = os.path.join(out_dir, f"bench_{timestamp}.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 已保存: {jpath}")


if __name__ == "__main__":
    main()