# 当前 OM 模型部署关键注意点（板端必读）

> 适用范围：香橙派 AI Pro（Ascend310B4）上加载的 **AIPP 版 RTMDet-Tiny OM**。
> 本文是「改板端代码前必读的坑位清单」，由主机侧逐条实测验证（2026-08-24）。

---

## 一、当前推荐的 OM 文件

| 用途 | 文件 | 说明 |
|------|------|------|
| 检测（更稳） | `RTMDet_Tiny_512_ft.om` | ft 微调版，pad=0，**no-NMS**，AIPP 内嵌 |
| 检测（更快） | `RTMDet_Tiny_416_ft.om` | 同上，计算 416×416，更快 |
| 关键点 | `RTMPose_s.om` / `RTMPose_s_v3.om` | **不受本文影响，代码保持不变** |

这些 OM 的规格：
- **输入** `input`：(1, 3, 480, 640)（NCHW 概念上），但**实际送的是 HWC uint8 原始字节**，布局与归一化全由 AIPP 在设备端做。
- **输出** `detections`：(1000, 6) = `[x1, y1, x2, y2, score, label]`，**无 NMS**，板端后处理自己过滤。
- 输入缓冲大小 = `aclmdlGetInputSizeByName(..., 0)` ≈ **921600** 字节（640×480×3）。

---

## 二、输入预处理铁律（本会话根因所在）

AIPP 版 OM 期望 host 送 **HWC 交错 uint8 原始字节**。板端代码只需：

```python
# ✅ 正确：直接送 HWC 交错 uint8
inp = np.ascontiguousarray(img_bgr, dtype=np.uint8)   # img_bgr 已 letterbox 到 640×480
```

```python
# ❌ 错误（板上 0.63 假阳性根因）：转成 NCHW 平面再送
x = img_bgr.transpose(2, 0, 1)[None]
inp = np.ascontiguousarray(x, dtype=np.uint8)
```

**为什么**：AIPP 按 HWC 逐像素读输入缓冲。NCHW 平面数据会被 AIPP 当成 HWC 交错读，通道/空间全部错乱 → 模型输出一个高分假框（实测 0.63，框还偏到画面外）。HWC 与 NCHW **字节数相同**（都是 921600），`aclrtMemcpyAsync` 照常拷贝、不会报错，所以纯看字节数无法发现。

**同时禁止**：`cv2.resize`（AIPP 不做 resize，尺寸必须已到 640×480）、减均值/除方差（AIPP 已做）。

---

## 三、三个已踩过的坑（含修复）

### 坑 1：`preprocess_rtmdet_raw` 返回 NCHW → 板上 0.63 假阳性
- **文件**：板端 `Scripts/preprocess.py`
- **症状**：检测框位置奇怪（如 `[0, 1343, 74, 1702]`）、score≈0.63，关键点全在框外；但同一模型同一图主机上正确检出 score≈0.90。
- **修复**：该函数返回 **HWC uint8**，不要 `transpose`。正确实现：

```python
def preprocess_rtmdet_raw(img_bgr, target_hw=DET_INPUT_HW):
    th, tw = target_hw
    h, w = img_bgr.shape[:2]
    if (h, w) != (th, tw):
        img_bgr = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(img_bgr, dtype=np.uint8)   # HWC uint8
```

### 坑 2：ONNX 图内 letterbox 的 pad 用了 114 → 置信度崩到 0.03
- **症状**：板端完全不出框 / score≈0.03~0.27。
- **根因**：AIPP **先归一化**，ONNX 图内 `KeepRatioResizePad` **再 pad**。pad=114 时，归一化空间里填进原始 114 是巨值尖峰，把分数砸穿。
- **修复（已做，勿再改）**：导 ONNX 时 `--pad-value 0.0`。当前 fix/ft OM 全部是 pad=0。
- **对板端的影响**：**无**。板端只负责 640×480 之前的 letterbox（灰条 128），图内 512/416 的 pad 是模型内部的事。

### 坑 3：加载了带 NMS 的旧 OM → `ValueError: cannot reshape array of size 500`
- **症状**：
  ```
  ValueError: cannot reshape array of size 500 into shape (1,1000,6)
  ```
- **根因**：板端 `det_infer.py` 按 `DET_OUTPUT_SHAPE=(1,1000,6)`（6000 元素，no-NMS）写死；加载的 OM 若输出 500 元素 = 带 NMS 的 `(100,5)` 旧版。
- **修复**：换用本文「一」列的 no-NMS OM。板端 `DET_OUTPUT_SHAPE` **保持 `(1,1000,6)`**，不要改。

---

## 四、AIPP 归一化参数（板端无需处理，仅供核对）

OM 内 AIPP 配置（`aipp_rtmdet_512.cfg`）：
- 均值 min（BGR 序）`103.53, 116.28, 123.675`；方差倒数 `1/57.375, 1/57.12, 1/58.395`
- 公式：`out = (pixel - min_chn) * var_reci`（mean_chn=0）
- `rbuv_swap_switch: false`（不翻通道，BGR 按位序透传）

主机已按位序验证与训练预处理逐位一致。**板端不要**再自己做归一化，否则双重归一化结果全错。

---

## 五、坐标空间约定

- RTMDet 输出坐标已在 **640×480 相机空间**（解码器内部已映射回），对 live 取流（帧本来就是 640×480）**无需任何坐标缩放**。
- 若走 `test_single.py` 的任意尺寸图片流程（先 letterbox 到 640×480），**必须**用 `inv_letterbox(boxes, r, pad_left, pad_top)` 把框映射回原图并 clip。
- letterbox 外部 pad 用 `pad_value=128`（灰），与模型训练一致。`pad_value=0` 也可以（模型对 pad 值不敏感，只要不是 114 巨值）。

---

## 六、板端代码修改清单（对照检查）

| 文件 | 位置 | 要求 |
|------|------|------|
| `Scripts/preprocess.py` | `preprocess_rtmdet_raw` | **返回 HWC uint8，禁止 transpose**（见坑 1 代码） |
| `Scripts/det_infer.py` | `DET_OUTPUT_SHAPE` | `(1, 1000, 6)`，不变 |
| `Scripts/det_infer.py` | `detect()` | 调 `preprocess_rtmdet_raw`（AIPP 版），**不要**调 `preprocess_rtmdet`（那是旧 float32 归一化版） |
| `Scripts/test_single.py` | letterbox | `pad_value=128`；框逆映射 `inv_letterbox` + clip |
| 任何输入拷贝 | aclmemcpy | 只拷字节，不换布局、不归一化 |

> ⚠️ 板上若存在**旧的非 AIPP OM**（旧的 `RTMDet_Tiny.om`，8-14 产物），代码里 `preprocess_rtmdet`（float32 NCHW 归一化）仍可兼容它；但**两者不可混用**。当前部署链统一走 AIPP 版：`preprocess_rtmdet_raw` + fix/ft OM。

---

## 七、验证标准

### test1.jpg（1280×1706，分布外但应正确）
- 修复后预期（主机实测正确值）：
  ```
  score ≈ 0.896
  box   ≈ [277, 591, 1112, 1729]   # 原图坐标，包住整个背部
  ```
- 关键点应全部落在框内。
- 若得分回到 0.9 附近且框正常 → **部署链打通**。

### 实时取流（640×480）
- 背部占据画面时 score 应 >0.9（ft 版平均 0.930）。
- 画框到 640×480 帧上，框应包住背部、无偏移。

---

## 八、排查速查表

| 症状 | 根因 | 处理 |
|------|------|------|
| reshape 500 报错 | 加载了带 NMS 旧 OM | 换 no-NMS fix/ft OM |
| score≈0.63 假框、框位置离谱 | `preprocess_rtmdet_raw` 返回了 NCHW | 改成返回 HWC uint8 |
| score≈0.03~0.27 完全不出框 | 图内 pad=114（旧 ONNX/OM） | 换 pad=0 fix/ft OM |
| 框偏、坐标错位 | 忘了 `inv_letterbox` / clip | letterbox 逆映射回原图 |
| 通道色偏（若出现） | AIPP 通道序不符 | 兜底：重导 OM 时 `rbuv_swap_switch=true`（当前 false 是验证过的正确值） |
