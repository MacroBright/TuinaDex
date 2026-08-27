# AcuPointDet ONNX 部署指南（Ubuntu 本地主机）

> 适用范围：在 **Ubuntu 本地主机（有 NVIDIA GPU）** 上用原始 ONNX 模型运行
> RTMDet-Tiny（背部检测）+ RTMPose-s（37 穴位关键点）两阶段推理。
> 与香橙派 AI Pro（Ascend 310B / OM）部署并行，保持**同一套预处理/后处理/输出约定**。

---

## 一、准备模型文件

| 用处 | 文件 | 输入 spec | 输出 spec |
|------|------|-----------|-----------|
| 检测 | `RTMDet_Tiny_512_aug20260824.onnx` | `(B,3,480,640)` NCHW，动态 batch（第 0 维 `0`），BGR，归一化后 | `(B,1000,6)` = `[x1,y1,x2,y2,score,label]`，**no-NMS**，坐标已逆映射回 640×480 空间 |
| 关键点 | `RTMPose_s.onnx` | `(1,3,256,192)` NCHW，RGB，归一化后 | `(1,37,3)` = `[x,y,conf]`，256×192 裁剪图坐标，需逆仿射 |

> ⚠️ 模型路径：本文按 `~/Desktop/AcuPointDet/Model/` 假设，**请把两个 .onnx 实际路径核对一遍**。
> ⚠️ RTMDet ONNX 不包含归一化层，host 必须手动 `(pixel-mean)/std`。

---

## 二、创建 Python 环境（conda 示例）

```bash
# 1. 安装 Miniconda（若未装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh    # 一路 yes，完成后 source ~/.bashrc

# 2. 新建独立环境（Python 3.9，与板端一致）
conda create -n acupoint python=3.9 -y
conda activate acupoint

# 3. 基础依赖
pip install numpy opencv-python pillow onnx
```

---

## 三、安装 onnxruntime-gpu（CUDA 匹配表）

onnxruntime-gpu 与 CUDA/cuDNN 有严格版本匹配，选错会报 `LoadLibrary` / CUDA 初始化错误。

| onnxruntime-gpu | CUDA | cuDNN |
|-----------------|------|-------|
| 1.17.x / 1.16.x | 11.8 | 8.9 |
| 1.18.x          | 12.x | 8.9+ |
| 1.19.x / 1.20.x | 12.x | 9.x |

**先查你的显卡驱动支持的 CUDA 版本：**
```bash
nvidia-smi            # 看右上角 "CUDA Version: 12.x"
```

**二选一：**
```bash
# A. CUDA 12.x 环境（推荐新卡）
pip install onnxruntime-gpu>=1.18

# B. CUDA 11.8 环境
pip install onnxruntime-gpu==1.17.1
```

**验证 GPU provider 可用：**
```bash
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 期望输出里包含 'CUDAExecutionProvider'，且它在列表中排在 CPU 之前
```

> 若只显示 `['CPUExecutionProvider']`：说明 CUDA 动态库没被找到。检查
> `ldconfig -p | grep cudart`、`conda install -c nvidia cuda-toolkit`（或安装
> `cudnn`/`cudatoolkit` 对应版本）后再验。

---

## 四、目录结构（与板端工程保持一致）

```
~/Desktop/AcuPointDet/
├── Model/
│   ├── RTMDet_Tiny_512_aug20260824.onnx
│   └── RTMPose_s.onnx
├── TestData/                  # 测试图片
├── Scripts/
│   ├── libs/                  # 复用板端通用模块（无需改）
│   │   ├── preprocess.py      # letterbox / crop_and_affine / 归一化常量
│   │   ├── postprocess.py     # NMS / 关键点逆仿射
│   │   ├── acupoints.py       # 37 穴位定义
│   │   └── visualize.py       # 画框 / 关键点 / 中文名
│   └── tools/
│       ├── onnx_infer.py      #【新增】ONNX 会话封装 + det/pose 推理接口
│       └── test_single.py     #【新增】单图验证入口
```

> `Scripts/libs/` 下 `acl_utils.py / det_infer.py / pose_infer.py` 是板端 ACL 专用，
> **不要**拷进主机。主机用下面的 `onnx_infer.py` 替代。

---

## 五、核心代码

### 5.1 `Scripts/tools/onnx_infer.py`

```python
# -*- coding: utf-8 -*-
"""ONNX 版推理封装（onnxruntime-gpu）。

用法（与板端 test_single 对齐）：
    source 环境后运行 test_single.py 即可，本模块被它调用。
"""
import sys
import os
import time
import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "libs"))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

from preprocess import (
    preprocess_rtmdet, crop_and_affine, DET_INPUT_HW,
    POSE_OUTPUT_WH, POSE_EXPAND,
)
from postprocess import postprocess_rtmdet, postprocess_rtmpose

DET_OUTPUT_SHAPE = (1, 1000, 6)
POSE_OUTPUT_SHAPE = (1, 37, 3)


class OnnxDet:
    """RTMDet 检测（ONNX）。输入 480×640 BGR 归一化。"""

    def __init__(self, onnx_path, device="cuda", num_threads=None):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device == "cuda" else ["CPUExecutionProvider"]
        sess_options = ort.SessionOptions()
        if num_threads:
            sess_options.intra_op_num_threads = num_threads
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def detect(self, img_bgr, score_thr=0.25, iou_thr=0.65, max_det=1):
        """img_bgr: 已 letterbox 到 480×640 的 uint8 BGR → (N,5) 原输入空间框。"""
        inp = preprocess_rtmdet(img_bgr, target_hw=DET_INPUT_HW)
        outs = self.sess.run(None, {self.input_name: inp})
        dets = np.asarray(outs[0]).reshape(DET_OUTPUT_SHAPE)[0]
        return postprocess_rtmdet(
            dets, score_thr=score_thr, iou_thr=iou_thr,
            max_det=max_det, img_hw=DET_INPUT_HW)


class OnnxPose:
    """RTMPose 关键点（ONNX）。输入 256×192 RGB 归一化。"""

    def __init__(self, onnx_path, device="cuda"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
            if device == "cuda" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    def estimate(self, img_bgr, bbox, kpt_thr=0.3, return_crop=False):
        inp, M, center, scale, crop = crop_and_affine(
            img_bgr, bbox, out_wh=POSE_OUTPUT_WH, expand=POSE_EXPAND)
        outs = self.sess.run(None, {self.input_name: inp})
        kpts = np.asarray(outs[0]).reshape(POSE_OUTPUT_SHAPE)[0]
        kpts_orig = postprocess_rtmpose(kpts, M, kpt_thr=kpt_thr)
        if return_crop:
            return kpts_orig, M, center, scale, crop
        return kpts_orig, M, center, scale
```

> ⚠️ `preprocess_rtmdet` 在 libs 里默认 `DET_INPUT_HW=(480,640)`，已与你的 onnx 一致
> （H=480, W=640）。若你的 onnx 另有尺寸，改 `libs/preprocess.py` 顶部常量即可。

### 5.2 `Scripts/tools/test_single.py`

主干与板端 `test_single.py` 相同，只把模型加载换成 ONNX：

```python
import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "libs"))
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import cv2
import numpy as np

from onnx_infer import OnnxDet, OnnxPose
from preprocess import letterbox_resize, inv_letterbox, DET_INPUT_HW
from visualize import draw_results
from acupoints import ACUPOINT_NAMES_CN, ACUPOINT_NAMES_EN

MODEL_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Model"))
DET_ONNX = os.path.join(MODEL_DIR, "RTMDet_Tiny_512_aug20260824.onnx")
POSE_ONNX = os.path.join(MODEL_DIR, "RTMPose_s.onnx")


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../TestData/test1.jpg"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"原图 {w}x{h}")

    # letterbox 到 480×640（与板端 AIPP 版输入一致）
    img_padded, r, (pad_left, pad_top) = letterbox_resize(img, DET_INPUT_HW, 128)
    print(f"letterbox r={r:.4f}, pad=({pad_left},{pad_top})")

    det = OnnxDet(DET_ONNX)
    pose = OnnxPose(POSE_ONNX)

    t0 = time.perf_counter()
    boxes = det.detect(img_padded)          # 480×640 空间
    t1 = time.perf_counter()
    if boxes.shape[0] > 0:
        boxes = inv_letterbox(boxes, r, pad_left, pad_top)   # 回原图
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        kpts, *_ = pose.estimate(img, boxes[0, :4])          # 原图坐标
    t2 = time.perf_counter()

    print(f"det {t1-t0:.1f} ms | pose {t2-t1:.1f} ms | total {t2-t0:.1f} ms")
    if boxes.shape[0] > 0:
        print(f"box {boxes[0]} score={boxes[0,4]:.3f}")
        print(f"kpts: {kpts.shape}")
        canvas = draw_results(img, boxes, kpts)
        out = os.path.join(os.path.dirname(img_path), "onnx_result.jpg")
        cv2.imwrite(out, canvas)
        print(f"结果图: {out}")


if __name__ == "__main__":
    import time
    main()
```

---

## 六、运行与验证

### 6.1 单图验证

```bash
conda activate acupoint
cd ~/Desktop/AcuPointDet/Scripts/tools
python test_single.py ../../TestData/test1.jpg
```

### 6.2 验证预期（与板端基线对照）

| 检查项 | 期望 |
|--------|------|
| 检测置信度 | 有背部区域时 `score > 0.85`（板端 ft 版约 0.93，主机 ONNX 应在同量级） |
| 检测框 | 包住整个背部，`x∈[277,1112], y∈[591,1729]`（test1.jpg 标准值，允许 ±20px） |
| 关键点 | 全部落框内，`conf mean > 0.6`；大椎(idx0)贴近颈部中线 |
| 坐标 | 关键点/框坐标都在原图 `[0,w)×[0,h)` 内 |

### 6.3 端到端延迟预期（GPU）

里程碑参考（依显卡而定）：检测 `(1,3,480,640)` + 关键点 `(1,3,256,192)`，
总耗时通常在 **5~25ms**（RTX 30/40 系），显著快于板端 310B 的 ~37ms。

---

## 七、常见问题

| 症状 | 原因 | 处理 |
|------|------|------|
| 检测分数普遍 <0.1 | 归一化 mean/std 或通道序错 | 确认 BGR mean `[103.53,116.28,123.675]`，**BGR 输入不转 RGB** |
| 关键点乱飞/飞出图 | 仿射矩阵 `M` 未记录或逆变换错 | 后处理必须用 `cv2.invertAffineTransform(M)` 回原图 |
| 输出全 0 | 输入尺寸/布局与 onnx 不符 | 确认 NCHW `(B,3,480,640)`、归一化后 |
| 报 CUDA 初始化失败 | onnxruntime-gpu 与 CUDA/cuDNN 版本不匹配 | 按第三节对照表换版本 |
| `providers` 列表没有 CUDA | cuDNN/cudart 缺失 | `conda install -c nvidia cudnn cuda-toolkit` 后重验 |
| 检测框比板端偏右/坐标错位 | letterbox 参数 `r/pad` 或逆映射错 | 复现板端 `inv_letterbox(boxes, r, pad_left, pad_top)` 流程 |

---

## 八、与板端 OM 部署的差异速查

| 项 | 香橙派（OM/AIPP） | Ubuntu（ONNX） |
|----|-------------------|----------------|
| 后端 | `acl.mdl.execute` | `onnxruntime` CUDAExecutionProvider |
| RTMDet 输入送法 | AIPP：送 `HWC uint8` 原始字节，设备端归一化 | **送 NCHW float32 已归一化张量** |
| 归一化 | 设备端 AIPP 做 | host 手动 `(pixel-mean)/std` |
| 坐标空间 | 模型输出即 640×480 空间 | 同（decoder 已映射回），仍需 `inv_letterbox` 回原图 |

> ⚠️ 最易踩坑：板端 AIPP 版**禁转 NCHW、禁手动归一化**；主机 ONNX 版**必须
> 转 NCHW + 手动归一化**。两者方向相反，别把板端习惯带到主机。