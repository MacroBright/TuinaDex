# -*- coding: utf-8 -*-
"""穴位识别端到端推理主入口。

流程：原图 → RTMDet 检测背部 bbox → 取最佳框 → RTMPose 估计 37 关键点
     → 逆仿射回原图 → 可视化 / 保存 JSON

支持三种输入模式：
  image : 单张图像
  batch : 文件夹批量
  video : 视频文件 / 摄像头（索引）

用法示例：
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  python main.py --det Model/RTMDet_Tiny.om --pose Model/RTMPose_s.om \
      --input test.jpg --output runs/
  python main.py --input images_dir/ --mode batch --output runs/
  python main.py --input 0 --mode video --output runs/
"""

import os
import sys
import time
import json
import argparse
import logging
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "libs"))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import cv2
import numpy as np

from acl_utils import init_acl, release_acl
from det_infer import DetInfer
from pose_infer import PoseInfer
from visualize import draw_results, put_info
from acupoints import ACUPOINT_NAMES_CN, ACUPOINT_NAMES_EN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


class AcuPointPipeline:
    """两阶段级联推理管线。"""

    def __init__(self, det_om, pose_om, device_id=0,
                 score_thr=0.25, iou_thr=0.65, kpt_thr=0.3, max_det=1):
        self.det = DetInfer(det_om, device_id=device_id)
        self.pose = PoseInfer(pose_om, device_id=device_id)
        self.score_thr = score_thr
        self.iou_thr = iou_thr
        self.kpt_thr = kpt_thr
        self.max_det = max_det

    def run(self, img_bgr):
        """对一张 BGR 图像执行端到端推理。

        Returns:
            dict: {
                "boxes": (N,5) float32 原图坐标,
                "keypoints": (37,3) float32 原图坐标 或 None,
                "best_box": (5,) 或 None,
                "det_ms": float, "pose_ms": float, "total_ms": float,
            }
        """
        t0 = time.perf_counter()
        boxes = self.det.detect_on_original(
            img_bgr, score_thr=self.score_thr,
            iou_thr=self.iou_thr, max_det=self.max_det)
        t1 = time.perf_counter()

        kpts = None
        best_box = None
        if boxes.shape[0] > 0:
            best_box = boxes[0]
            kpts, _, _, _ = self.pose.estimate(
                img_bgr, best_box[:4], kpt_thr=self.kpt_thr)
        t2 = time.perf_counter()

        return {
            "boxes": boxes,
            "keypoints": kpts,
            "best_box": best_box,
            "det_ms": (t1 - t0) * 1000,
            "pose_ms": (t2 - t1) * 1000,
            "total_ms": (t2 - t0) * 1000,
        }

    def close(self):
        self.det.close()
        self.pose.close()


def _save_json(out_dir, name, img_bgr, result):
    os.makedirs(out_dir, exist_ok=True)
    boxes = result["boxes"].tolist() if result["boxes"] is not None else []
    kpts = result["keypoints"]
    keypoints_out = None
    if kpts is not None:
        keypoints_out = []
        for idx in range(kpts.shape[0]):
            x, y, c = float(kpts[idx, 0]), float(kpts[idx, 1]), float(kpts[idx, 2])
            keypoints_out.append({
                "index": idx,
                "name_en": ACUPOINT_NAMES_EN[idx],
                "name_cn": ACUPOINT_NAMES_CN[idx],
                "x": round(x, 2),
                "y": round(y, 2),
                "score": round(c, 4),
            })
    payload = {
        "image": name,
        "image_size": {"h": int(img_bgr.shape[0]), "w": int(img_bgr.shape[1])},
        "boxes": [[round(v, 2) for v in b] for b in boxes],
        "keypoints": keypoints_out,
        "elapsed_ms": {
            "det": round(result["det_ms"], 2),
            "pose": round(result["pose_ms"], 2),
            "total": round(result["total_ms"], 2),
        },
    }
    json_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return json_path


def _annotate(img_bgr, result, draw_names=True):
    canvas = draw_results(img_bgr, result["boxes"], result["keypoints"],
                          conf_thr=0.3, draw_names=draw_names)
    info = [
        f"det: {result['det_ms']:.1f} ms",
        f"pose: {result['pose_ms']:.1f} ms",
        f"total: {result['total_ms']:.1f} ms",
    ]
    if result["keypoints"] is not None:
        n_valid = int(np.sum((result["keypoints"][:, 2] >= 0.3) &
                             (result["keypoints"][:, 0] >= 0)))
        info.append(f"kpts valid: {n_valid}/37")
    put_info(canvas, info)
    return canvas


def run_image(pipeline, img_path, out_dir, save_vis=True, save_json=True,
              show=False, draw_names=True):
    img = cv2.imread(img_path)
    if img is None:
        logger.error("读取图像失败: %s", img_path)
        return None
    result = pipeline.run(img)
    name = os.path.basename(img_path)
    logger.info("[%s] det=%.1fms pose=%.1fms total=%.1fms boxes=%d kpts=%s",
                name, result["det_ms"], result["pose_ms"], result["total_ms"],
                result["boxes"].shape[0],
                "yes" if result["keypoints"] is not None else "no")

    if out_dir:
        if save_json:
            _save_json(out_dir, name, img, result)
        if save_vis:
            canvas = _annotate(img, result, draw_names=draw_names)
            vis_path = os.path.join(out_dir, os.path.splitext(name)[0] + "_result.jpg")
            cv2.imwrite(vis_path, canvas)
            if show:
                cv2.imshow("AcuPointDet", canvas)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    return result


def run_batch(pipeline, input_dir, out_dir, **kw):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob.glob(os.path.join(input_dir, "*" + ext)))
        files.extend(glob.glob(os.path.join(input_dir, "*" + ext.upper())))
    files = sorted(set(files))
    if not files:
        logger.error("目录中未找到图像: %s", input_dir)
        return
    logger.info("批量推理: 共 %d 张图像", len(files))
    latencies = []
    for fp in files:
        r = run_image(pipeline, fp, out_dir, **kw)
        if r is not None:
            latencies.append(r["total_ms"])
    if latencies:
        logger.info("批量完成: 平均 %.1f ms, 最大 %.1f ms, 最小 %.1f ms",
                    np.mean(latencies), np.max(latencies), np.min(latencies))


def run_video(pipeline, input_src, out_dir, save_json=False, show=True,
              draw_names=True, fourcc="mp4v", fps=None, queue_size=2):
    """三线程异步流水线（读图 / 检测 / 关键点各一线程）。

    读图、RTMDet 推理、RTMPose 推理三阶段用线程 + 有界队列重叠执行，
    使帧率由最慢单级决定而非所有阶段累加。帧策略：丢旧帧、追最新
    （队列满时丢弃最旧帧，实时低延迟优先）。
    """
    import threading
    import queue
    from acl_utils import bind_thread_context

    cap = cv2.VideoCapture(input_src)
    if not cap.isOpened():
        logger.error("无法打开视频源: %s", input_src)
        return
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps is None:
        fps = src_fps
    writer = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tag = "cam" if isinstance(input_src, int) else os.path.basename(str(input_src))
        out_path = os.path.join(out_dir, f"{tag}_result.mp4")
        fourcc_obj = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(out_path, fourcc_obj, fps, (src_w, src_h))
        logger.info("视频输出: %s", out_path)

    q_frames = queue.Queue(maxsize=queue_size)   # (frame_idx, frame)
    q_boxes = queue.Queue(maxsize=queue_size)    # (frame_idx, frame, boxes, det_ms)
    q_out = queue.Queue(maxsize=queue_size)      # (frame_idx, canvas, result)
    stop = threading.Event()
    dropped = {"cap": 0, "det": 0, "pose": 0}

    def _drop_oldest(q, key):
        """队列满时丢弃最旧元素（追最新策略）。"""
        try:
            q.get_nowait()
            dropped[key] += 1
        except queue.Empty:
            pass

    def _capture():
        idx = 0
        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            if q_frames.full():
                _drop_oldest(q_frames, "cap")
            q_frames.put((idx, frame))
            idx += 1
        stop.set()

    def _det():
        bind_thread_context()
        while not stop.is_set():
            try:
                idx, frame = q_frames.get(timeout=0.05)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            boxes = pipeline.det.detect_on_original(
                frame, score_thr=pipeline.score_thr,
                iou_thr=pipeline.iou_thr, max_det=pipeline.max_det)
            det_ms = (time.perf_counter() - t0) * 1000
            if q_boxes.full():
                _drop_oldest(q_boxes, "det")
            q_boxes.put((idx, frame, boxes, det_ms))

    def _pose():
        bind_thread_context()
        while not stop.is_set():
            try:
                idx, frame, boxes, det_ms = q_boxes.get(timeout=0.05)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            kpts = None
            if boxes.shape[0] > 0:
                kpts, _, _, _ = pipeline.pose.estimate(
                    frame, boxes[0, :4], kpt_thr=pipeline.kpt_thr)
            pose_ms = (time.perf_counter() - t0) * 1000
            result = {
                "boxes": boxes,
                "keypoints": kpts,
                "best_box": boxes[0] if boxes.shape[0] > 0 else None,
                "det_ms": det_ms,
                "pose_ms": pose_ms,
                "total_ms": det_ms + pose_ms,
            }
            canvas = _annotate(frame, result, draw_names=draw_names)
            if q_out.full():
                _drop_oldest(q_out, "pose")
            q_out.put((idx, canvas, result))

    threads = [
        threading.Thread(target=_capture, name="capture", daemon=True),
        threading.Thread(target=_det, name="det-worker", daemon=True),
        threading.Thread(target=_pose, name="pose-worker", daemon=True),
    ]
    for t in threads:
        t.start()

    frame_idx = 0
    while True:
        try:
            _, canvas, result = q_out.get(timeout=0.05)
        except queue.Empty:
            if stop.is_set() and q_out.empty() \
                    and q_frames.empty() and q_boxes.empty():
                break
            continue
        if writer is not None:
            writer.write(canvas)
        if show:
            cv2.imshow("AcuPointDet", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop.set()
                break
        frame_idx += 1
        if frame_idx % 30 == 0:
            logger.info("frame %d, total=%.1fms | 丢弃: %s",
                        frame_idx, result["total_ms"], dropped)

    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    logger.info("视频推理完成, 共 %d 帧 (丢弃: %s)", frame_idx, dropped)


def build_argparser():
    p = argparse.ArgumentParser(description="AcuPointDet 端到端推理（昇腾 310B OM）")
    p.add_argument("--det", default="Model/RTMDet_Tiny.om", help="检测 om 路径")
    p.add_argument("--pose", default="Model/RTMPose_s.om", help="关键点 om 路径")
    p.add_argument("--input", "-i", required=True,
                   help="图像/文件夹/视频文件/摄像头索引(如 0)")
    p.add_argument("--output", "-o", default="runs", help="输出目录")
    p.add_argument("--mode", "-m", choices=["image", "batch", "video", "auto"],
                   default="auto", help="输入模式，auto 自动判断")
    p.add_argument("--device", type=int, default=0, help="昇腾 device id")
    p.add_argument("--score-thr", type=float, default=0.25, help="检测分数阈值")
    p.add_argument("--iou-thr", type=float, default=0.65, help="NMS IoU 阈值")
    p.add_argument("--kpt-thr", type=float, default=0.3, help="关键点置信度阈值")
    p.add_argument("--max-det", type=int, default=1, help="最多保留检测框数")
    p.add_argument("--queue-size", type=int, default=2,
                   help="异步流水线队列深度（默认 2，越大越能吃满但延迟越高）")
    p.add_argument("--no-vis", action="store_true", help="不保存可视化图")
    p.add_argument("--no-json", action="store_true", help="不保存 JSON 结果")
    p.add_argument("--no-names", action="store_true", help="可视化不画穴位名")
    p.add_argument("--show", action="store_true", help="弹窗显示（图像/视频）")
    return p


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


def resolve_mode(mode, input_src):
    if mode != "auto":
        return mode
    if input_src.isdigit():
        return "video"
    if os.path.isdir(input_src):
        return "batch"
    ext = os.path.splitext(str(input_src))[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if os.path.isfile(input_src):
        return "image"
    return "video"


def main():
    args = build_argparser().parse_args()

    det_path = args.det if os.path.isabs(args.det) else \
        os.path.join(SCRIPT_DIR, "..", "..", args.det)
    pose_path = args.pose if os.path.isabs(args.pose) else \
        os.path.join(SCRIPT_DIR, "..", "..", args.pose)
    det_path = os.path.normpath(det_path)
    pose_path = os.path.normpath(pose_path)

    if not os.path.isfile(det_path):
        logger.error("检测模型不存在: %s", det_path)
        sys.exit(1)
    if not os.path.isfile(pose_path):
        logger.error("关键点模型不存在: %s", pose_path)
        sys.exit(1)

    mode = resolve_mode(args.mode, args.input)
    logger.info("模式: %s | det=%s | pose=%s | input=%s | output=%s",
                mode, det_path, pose_path, args.input, args.output)

    pipeline = AcuPointPipeline(
        det_path, pose_path, device_id=args.device,
        score_thr=args.score_thr, iou_thr=args.iou_thr,
        kpt_thr=args.kpt_thr, max_det=args.max_det)

    try:
        if mode == "image":
            run_image(pipeline, args.input, args.output,
                      save_vis=not args.no_vis, save_json=not args.no_json,
                      show=args.show, draw_names=not args.no_names)
        elif mode == "batch":
            run_batch(pipeline, args.input, args.output,
                      save_vis=not args.no_vis, save_json=not args.no_json,
                      show=args.show, draw_names=not args.no_names)
        elif mode == "video":
            src = int(args.input) if args.input.isdigit() else args.input
            run_video(pipeline, src, args.output,
                      save_json=not args.no_json, show=args.show,
                      draw_names=not args.no_names,
                      queue_size=args.queue_size)
    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()