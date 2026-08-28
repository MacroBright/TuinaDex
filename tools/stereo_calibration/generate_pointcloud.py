"""Capture one rectified stereo pair and export a coloured point cloud."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .cameras import OpenCVCameraPair
from .config import AppConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a stereo point cloud")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--min-disparity", type=int, default=-432)
    parser.add_argument("--num-disparities", type=int, default=432)
    parser.add_argument("--max-depth-mm", type=float, default=5000.0)
    parser.add_argument("--stride", type=int, default=2)
    return parser


def _write_binary_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("xb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def generate(args: argparse.Namespace) -> dict[str, object]:
    if args.warmup_frames < 1:
        raise ValueError("warmup frames must be positive")
    if args.num_disparities <= 0 or args.num_disparities % 16:
        raise ValueError("num disparities must be a positive multiple of 16")
    if args.stride < 1:
        raise ValueError("stride must be positive")

    config = AppConfig.from_json(args.config)
    output = args.output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    camera = OpenCVCameraPair(config)
    camera.open()
    try:
        pair = None
        for _ in range(args.warmup_frames):
            pair = camera.read()
    finally:
        camera.close()
    assert pair is not None

    with np.load(args.calibration) as calibration:
        left_rectified = cv2.remap(
            pair.left,
            calibration["map_left_x"],
            calibration["map_left_y"],
            cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            pair.right,
            calibration["map_right_x"],
            calibration["map_right_y"],
            cv2.INTER_LINEAR,
        )
        disparity_to_depth = calibration["disparity_to_depth"].copy()

    block_size = 7
    matcher = cv2.StereoSGBM_create(
        minDisparity=args.min_disparity,
        numDisparities=args.num_disparities,
        blockSize=block_size,
        P1=8 * block_size * block_size,
        P2=32 * block_size * block_size,
        disp12MaxDiff=2,
        preFilterCap=31,
        uniquenessRatio=8,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
    disparity = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    points = cv2.reprojectImageTo3D(disparity, disparity_to_depth)
    depth = points[:, :, 2]
    upper_disparity = args.min_disparity + args.num_disparities
    valid = (
        (disparity > args.min_disparity + 1)
        & (disparity < upper_disparity - 1)
        & np.isfinite(points).all(axis=2)
        & (depth > 200.0)
        & (depth < args.max_depth_mm)
    )
    sampled = valid.copy()
    rows, columns = np.indices(sampled.shape)
    sampled &= (rows % args.stride == 0) & (columns % args.stride == 0)
    xyz = points[sampled].astype(np.float32)
    rgb = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2RGB)[sampled].astype(np.uint8)
    if len(xyz) < 1_000:
        raise RuntimeError(f"too few valid points: {len(xyz)}")

    cv2.imwrite(str(output / "left.png"), pair.left)
    cv2.imwrite(str(output / "right.png"), pair.right)
    cv2.imwrite(str(output / "left_rectified.png"), left_rectified)
    cv2.imwrite(str(output / "right_rectified.png"), right_rectified)
    _write_binary_ply(output / "pointcloud.ply", xyz, rgb)

    visualization = np.zeros_like(left_gray)
    scaled = np.clip(
        (disparity - args.min_disparity) / args.num_disparities * 255.0,
        0,
        255,
    ).astype(np.uint8)
    visualization[valid] = scaled[valid]
    visualization = cv2.applyColorMap(visualization, cv2.COLORMAP_TURBO)
    visualization[~valid] = 0
    cv2.imwrite(str(output / "disparity.png"), visualization)
    np.savez_compressed(
        output / "depth_and_disparity.npz",
        depth_mm=depth,
        disparity_px=disparity,
        valid_mask=valid,
    )

    summary: dict[str, object] = {
        "output_dir": str(output),
        "pointcloud": str(output / "pointcloud.ply"),
        "points": int(len(xyz)),
        "valid_pixels": int(valid.sum()),
        "depth_median_mm": float(np.median(depth[valid])),
        "depth_p05_mm": float(np.percentile(depth[valid], 5)),
        "depth_p95_mm": float(np.percentile(depth[valid], 95)),
        "calibration": str(args.calibration),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(generate(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
