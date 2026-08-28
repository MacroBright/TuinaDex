# 香橙派推理代码编写指南

> 适用于已转换格式的 RTMDet-Tiny + RTMPose-s 模型（ONNX → RKNN/OM 等）
> 本指南提供推理所需全部模型元信息：输入输出规格、预处理参数、后处理逻辑、关键点编号

## 1. 模型元信息总览

| 属性 | RTMDet-Tiny（检测） | RTMPose-s（关键点） |
|------|---------------------|---------------------|
| 模型作用 | 检测背部区域 bbox | 估计 37 个背部穴位关键点 |
| 输入尺寸 | **720×1280**（H×W） | **256×192**（H×W） |
| 输入格式 | BGR，NCHW，Float32 | RGB，NCHW，Float32 |
| 输入范围 | 归一化后 | 归一化后 |
| 输出形状 | `(batch, 1000, 6)` | `(batch, 37, 3)` |
| 输出列含义 | `[x1, y1, x2, y2, score, label]` | `[x, y, conf]` |
| 输出坐标 | **原图像素坐标**（720×1280） | **输入图坐标**（256×192 内） |
| NMS | **未内置**（需自行实现） | — |
| 类别数 | 1（`back`） | — |

### 1.1 RTMDet 输出详解

输出形状 `(batch, 1000, 6)`，固定 1000 行候选框（已按分数降序排列）：

```
每行: [x1, y1, x2, y2, score, label]
  x1,y1,x2,y2: 检测框坐标，单位像素，直接对应原图 (0~720/0~1280)
  score: 置信度 0~1
  label: 类别 ID，恒为 0 (back)
```

⚠️ **注意**：该模型导出时 `--no-nms`，输出 1000 个**候选框**而非最终检测结果。
推理后必须自行做：
1. **分数阈值过滤**（如 `score > 0.25`）
2. **NMS** 去重（IoU 阈值如 0.65）
3. 单类检测场景：取 score 最高的框即可

### 1.2 RTMPose 输出详解

输出形状 `(batch, 37, 3)`：

```
每行: [x, y, conf]
  x,y: 关键点坐标，单位像素，位于输入的 256×192 图像内
  conf: 置信度 0~1
```

⚠️ **注意**：RTMPose 是 top-down 模型，输入是**裁剪并仿射变换后**的 256×192 人体区域。
输出的关键点坐标在 256×192 裁剪图坐标系内，需要**逆仿射变换**映射回原图。

## 2. 预处理参数

### 2.1 归一化（两模型都必须手动做）

ONNX 模型**不含**归一化层，推理时必须手动执行（`(pixel - mean) / std`）：

| 模型 | mean (BGR 顺序) | std (BGR 顺序) | 色彩空间 |
|------|-----------------|----------------|----------|
| RTMDet | `[103.53, 116.28, 123.675]` | `[57.375, 57.12, 58.395]` | BGR |
| RTMPose | `[123.675, 116.28, 103.53]` | `[58.395, 57.12, 57.375]` | RGB |

> ⚠️ **RTMDet 用 BGR**（mean/std 按 BGR 顺序排列，模型训练时 `bgr_to_rgb=False`）
> ⚠️ **RTMPose 用 RGB**（mean/std 按 RGB 顺序排列，训练时 `bgr_to_rgb=True`）

### 2.2 RTMDet 输入

```python
def preprocess_rtmdet(img_bgr):  # img_bgr: HxWx3, uint8, BGR
    # 转 NCHW float32
    x = img_bgr.transpose(2, 0, 1)[None].astype(np.float32)
    # 归一化 (BGR 通道顺序)
    mean = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    std = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    x = (x - mean[:, None, None]) / std[:, None, None]
    return x  # (1, 3, 720, 1280)
```

> RTMDet ONNX 内部已内置 Resize（自动缩放到 640×640），**外部无需缩放**，直接输入 720×1280 原图。

### 2.3 RTMPose 输入（从检测框裁剪）

RTMPose 输入需要从检测框裁剪出背部区域并仿射到 256×192：

```python
def crop_and_affine(img_bgr, bbox, out_size=(192, 256)):
    """从 bbox 裁剪背部区域并仿射变换到 256×192 (宽192, 高256)"""
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    w, h = x2 - x1, y2 - y1
    # 外扩 1.25 倍，保留周围皮肤上下文
    scale = max(w, h) / 200 * 1.25  # 200 ≈ 基准尺寸
    # 仿射矩阵: 将 center 映射到输出中心, scale 映射到输出尺寸
    rot = 0.0
    src = np.array([
        [center[0] - 0.5 * scale * 192, center[1] - 0.5 * scale * 256],
        [center[0] + 0.5 * scale * 192, center[1] - 0.5 * scale * 256],
    ], dtype=np.float32)
    dst = np.array([
        [0, 0],
        [out_size[0] - 1, 0],
    ], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(img_bgr, M, out_size, flags=cv2.INTER_LINEAR)
    # 转 RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    # NCHW + 归一化 (RGB 顺序)
    x = crop_rgb.transpose(2, 0, 1)[None].astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)  # RGB
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)     # RGB
    x = (x - mean[:, None, None]) / std[:, None, None]
    return x, M
```

> ⚠️ 记录仿射矩阵 `M`，用于后处理把关键点映射回原图。

## 3. 后处理逻辑

### 3.1 RTMDet：过滤 + NMS

```python
def postprocess_rtmdet(dets, score_thr=0.25, iou_thr=0.65):
    """dets: (1000, 6) → 过滤后的检测框 (N, 5)"""
    # 分数过滤
    valid = dets[:, 4] > score_thr
    boxes = dets[valid, :5]  # (N, 5) [x1,y1,x2,y2,score]
    if len(boxes) == 0:
        return boxes
    # NMS (单类)
    keep = cv2.dnn.NMSBoxes(
        boxes[:, :4].tolist(), boxes[:, 4].tolist(),
        score_thr, iou_thr)
    return boxes[keep.flatten()] if isinstance(keep, np.ndarray) else boxes[keep]
```

### 3.2 RTMPose：坐标逆变换

```python
def postprocess_rtmpose(kpts, M):
    """kpts: (37, 3) [x, y, conf]，M: 正向仿射矩阵 → 原图坐标"""
    # kpts 坐标在 256×192 内
    xy = kpts[:, :2]  # (37, 2)
    # 逆仿射: 用 getAffineTransform 计算的 M 求逆
    M_inv = cv2.invertAffineTransform(M)
    xy_orig = cv2.transform(xy.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
    return np.concatenate([xy_orig, kpts[:, 2:]], axis=1)  # (37, 3)
```

## 4. 关键点编号与名称（37 个）

> 用于映射输出索引 → 穴位名。**下标顺序固定**，推理代码可直接使用。

| 索引 | 名称 | 中文 | 索引 | 名称 | 中文 |
|------|------|------|------|------|------|
| 0 | dazhui | 大椎 | 19 | tianzong_L | 左天宗 |
| 1 | jianjing_L | 左肩井 | 20 | tianzong_R | 右天宗 |
| 2 | jianjing_R | 右肩井 | 21 | geshu_L | 左膈俞 |
| 3 | naoshu_L | 左臑俞 | 22 | geshu_R | 右膈俞 |
| 4 | naoshu_R | 右臑俞 | 23 | ganshu_L | 左肝俞 |
| 5 | jianzhen_L | 左肩贞 | 24 | ganshu_R | 右肝俞 |
| 6 | jianzhen_R | 右肩贞 | 25 | danshu_L | 左胆俞 |
| 7 | dazhu_L | 左大杼 | 26 | danshu_R | 右胆俞 |
| 8 | dazhu_R | 右大杼 | 27 | pishu_L | 左脾俞 |
| 9 | fengmen_L | 左风门 | 28 | pishu_R | 右脾俞 |
| 10 | fengmen_R | 右风门 | 29 | weishu_L | 左胃俞 |
| 11 | feishu_L | 左肺俞 | 30 | weishu_R | 右胃俞 |
| 12 | feishu_R | 右肺俞 | 31 | sanjiaoshu_L | 左三焦俞 |
| 13 | jueyinshu_L | 左厥阴俞 | 32 | sanjiaoshu_R | 右三焦俞 |
| 14 | jueyinshu_R | 右厥阴俞 | 33 | shenshu_L | 左肾俞 |
| 15 | xinshu_L | 左心俞 | 34 | shenshu_R | 右肾俞 |
| 16 | xinshu_R | 右心俞 | 35 | dachangshu_L | 左大肠俞 |
| 17 | gaohuang_L | 左膏肓 | 36 | dachangshu_R | 右大肠俞 |
| 18 | gaohuang_R | 右膏肓 | | | |

**对称性**：每对穴位相邻（1=L, 2=R, 3=L, 4=R...），奇数=左，偶数=右。
dazhui（0）为脊柱中线点，无左右之分。

## 5. 完整推理流水线（参考）

```python
import cv2
import numpy as np

# ---- 加载模型（示例：RKNN Lite / ONNX Runtime 二选一）----
# RKNN:
# from rknnlite.api import RKNNLite
# det_model = RKNNLite(); det_model.load_rknn('RTMDet_Tiny.rknn'); det_model.init_runtime()
# pose_model = RKNNLite(); pose_model.load_rknn('RTMPose_s.rknn'); pose_model.init_runtime()
#
# ONNX Runtime:
# import onnxruntime as ort
# det_model = ort.InferenceSession('RTMDet_Tiny.onnx')
# pose_model = ort.InferenceSession('RTMPose_s.onnx')

def infer(frame_bgr):
    """frame_bgr: 720×1280 BGR uint8"""
    # 1. RTMDet 检测
    det_input = preprocess_rtmdet(frame_bgr)
    dets = det_model.run(None, {'input': det_input})[0][0]  # (1000, 6)
    boxes = postprocess_rtmdet(dets)  # (N, 5)

    if len(boxes) == 0:
        return None

    # 2. 取最佳检测框 → RTMPose 关键点
    best = boxes[0]  # 最高分
    pose_input, M = crop_and_affine(frame_bgr, best[:4])
    kpts = pose_model.run(None, {'input': pose_input})[0][0]  # (37, 3)

    # 3. 坐标映射回原图
    kpts_orig = postprocess_rtmpose(kpts, M)
    return boxes, kpts_orig  # (N,5) 检测框 + (37,3) 原图关键点
```

## 6. 调试与验证

### 6.1 输出正确性检查

| 检查项 | 期望 |
|--------|------|
| dets[0, :, 4].max() | 含真实背部时 > 0.5 |
| kpts[:, 2].mean() | 正常背部 > 0.6 |
| 关键点坐标范围 | 应在原图 (720×1280) 内 |
| dazhui (idx 0) 位置 | 脊柱上端，靠近颈部 |

### 6.2 常见问题

| 问题 | 原因 |
|------|------|
| 检测分数普遍很低 (<0.1) | 归一化参数或 BGR/RGB 通道顺序错误 |
| 关键点坐标错乱/飞出图 | 仿射矩阵 M 未记录或逆变换错误 |
| 输出全 0 | 模型输入尺寸与导出时不一致 |
| 格式转换后输出形状变化 | 检查转换工具是否改变了 batch 维度 |

## 7. 关键文件

| 文件 | 说明 |
|------|------|
| `configs/back_acupoint_37.py` | 37 关键点完整定义（名称/颜色/连接） |
| `scripts/export_rtmpose_onnx.py` | RTMPose ONNX 导出脚本 |
| `scripts/export_rtmdet_onnx.py` | RTMDet ONNX 导出脚本 |
| `work_dirs/rtmpose_s/RTMPose_s.onnx` | RTMPose ONNX 模型 |
| `work_dirs/rtmdet_tiny/RTMDet_Tiny.onnx` | RTMDet ONNX 模型 |
