# AcuPointDet —— 中医穴位识别（香橙派 AI Pro 板端部署）

在香橙派 AI Pro（昇腾 310B）上运行 **RTMDet-Tiny 背部区域检测 + RTMPose-s 37 穴位关键点** 两阶段级联推理，
输入图像或摄像头视频流，输出背部检测框、37 个穴位坐标与置信度，并生成可视化结果与结构化 JSON。

- 部署平台：香橙派 AI Pro（昇腾 310B4），AIPP 版 `.om` 离线模型
- 语言/框架：Python 3.9 + CANN pyACL（原生 acl 接口）
- 相关文档：[OM_Deploy_Notes.md](OM_Deploy_Notes.md)（板端部署关键坑位清单，必读）

---

## 一、硬件与软件环境

| 项目 | 要求 |
|------|------|
| 设备 | 香橙派 AI Pro（aarch64 / ARM64） |
| AI 芯片 | 昇腾 310B4（Atlas 310B 系列） |
| 操作系统 | Ubuntu 22.04.5 LTS（内核 5.10.0+，本工程实测环境） |
| CANN 工具链 | CANN 8.0.0（`/usr/local/Ascend/ascend-toolkit/`，含 atc、pyACL） |
| Python | 3.9.x（本工程实测 3.9.2） |
| Python 依赖 | `numpy>=1.20`（实测 1.26.4）、`opencv-python`（实测 4.10.0） |

> 依赖中不包含 `acl` 的 pip 包：acl 是 CANN 自带的 pyACL 库，
> 加载环境变量后直接 `import acl` 即可（见"环境准备"）。

---

## 二、目录结构

```
AcuPointDet/
├── Model/
│   ├── RTMDet_Tiny.om        # 检测模型（AIPP 版，输入 480×640）
│   └── RTMPose_s.om          # 关键点模型（输入 256×192）
├── Scripts/
│   ├── libs/                 # 通用库（可被 tools 复用）
│   │   ├── acl_utils.py      # ACL 初始化/OmModel 封装/线程 context 绑定
│   │   ├── preprocess.py     # letterbox、AIPP 原始字节、仿射裁剪
│   │   ├── det_infer.py      # RTMDet 推理封装
│   │   ├── pose_infer.py     # RTMPose 推理封装
│   │   ├── postprocess.py    # NMS、关键点逆仿射
│   │   ├── acupoints.py      # 37 穴位名称表
│   │   └── visualize.py      # 可视化绘制
│   └── tools/                # 入口脚本
│       ├── test_single.py    # 单张图片端到端测试
│       ├── benchmark.py      # 延迟分布测量（t0~t5 分段）
│       └── main.py           # 图像/批量/视频/摄像头推理主入口
├── TestData/                 # 测试图片 test1~test11.jpg
├── Output/                   # test_single / benchmark 默认输出（时间戳子目录）
└── Scripts/tools/runs/       # main.py 默认输出（视频）
```

---

## 三、环境准备

```bash
# 1. 加载 CANN 环境变量（每个新终端都必须执行一次）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 确认 acl 可导入
python3 -c "import acl; print(acl.__file__)"
# 期望输出: /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/acl.so

# 3. 确认依赖已装（缺则 pip install）
python3 -c "import numpy, cv2; print(numpy.__version__, cv2.__version__)"
```

若 `import acl` 失败，通常是没走第 1 步；若依赖缺失：

```bash
pip3 install numpy opencv-python
```

---

## 四、模型文件

当前仓库 `Model/` 已包含转换好的两个 OM：

| 模型 | 用途 | 输入 spec | 输出 spec |
|------|------|-----------|-----------|
| `RTMDet_Tiny.om` | 背部区域检测 | **HWC uint8 原始字节，480×640**（AIPP 在设备端转 416×416 并归一化） | `(1000, 6)` = `[x1,y1,x2,y2,score,label]`，**no-NMS**，坐标已映射回 640×480 空间 |
| `RTMPose_s.om` | 37 穴位关键点 | 仿射裁剪图 256×192 RGB 归一化 | `(37, 3)` = `[x,y,conf]`，裁剪图坐标，需逆仿射回原图 |

> ⚠️ 检测模型是 **AIPP 版**：host 只送 HWC 交错 uint8 原始字节，
> **严禁 transpose 成 NCHW、严禁手动归一化、严禁再 resize**（AIPP 已完成）。
> 这是板端最容易踩的坑，详见 `Scripts/libs/preprocess.py` 的 `preprocess_rtmdet_raw` 与 OM_Deploy_Notes.md。
>
> 若需从 ONNX 自行重新转换 OM（含 AIPP 配置），请参考主机侧的 `aipp_rtmdet_512.cfg` 与 atc 命令，
> AIPP 归一化参数（BGR 序）：mean `103.53,116.28,123.675`，var_reci `1/57.375,1/57.12,1/58.395`（不翻通道）。

模型推理实现了进程级 ACL 单例（`acl_utils.py`）：`acl.init` / `set_device` / `create_context` 只做一次，
多线程流水线中 worker 线程通过 `bind_thread_context()` 显式绑定 context（ACL 要求每个线程先 set_context 再执行）。

---

## 五、快速开始

所有命令都在 `Scripts/tools/` 目录下执行：

```bash
cd ~/Desktop/AcuPointDet/Scripts/tools
```

### 5.1 单张图片测试

```bash
# 最简单形式（模型用默认相对路径 Model/RTMDet_Tiny.om 与 Model/RTMPose_s.om）
python test_single.py ../../TestData/test1.jpg

# 显示指定模型与输出目录
python test_single.py ../../TestData/test1.jpg \
    --det ../../Model/RTMDet_Tiny.om --pose ../../Model/RTMPose_s.om \
    --output /home/HwHiAiUser/Desktop/AcuPointDet/Output

# 跳过检测，直接把整图仿射到 256×192 喂关键点模型（调试用）
python test_single.py ../../TestData/test1.jpg --no-det
```

常用参数：`--score-thr`（默认 0.25）、`--iou-thr`（默认 0.65）、`--kpt-thr`（默认 0.3）、
`--no-save`（不保存结果）、`--device`（昇腾 device id，默认 0）。

### 5.2 延迟分布测量（benchmark）

按 t0~t5 分段输出每阶段耗时（读图 / 检测预处理 / 检测推理 / 检测后处理 / 关键点推理 / 关键点后处理）：

```bash
python benchmark.py ../../TestData/test1.jpg -n 10        # 单图重复 10 次
python benchmark.py ../../TestData -n 20                  # 目录内全部图片
python benchmark.py a.jpg b.jpg -n 5 --json               # 多图 + 存 JSON
```

### 5.3 摄像头 / 视频 / 批量推理（main.py）

```bash
# 摄像头实时推理（input=0 表示摄像头索引；--show 弹窗显示，按 q 退出）
python main.py --det ../../Model/RTMDet_Tiny.om --pose ../../Model/RTMPose_s.om \
    --input 0 --mode video --show

# 视频文件推理
python main.py --input /path/to/video.mp4 --mode video

# 文件夹批量推理
python main.py --input /path/to/images_dir --mode batch

# 单张图像
python main.py --input ../../TestData/test1.jpg --mode image
```

> 摄像头视频采用 **三线程异步流水线**（读图 / 检测 / 关键点各一线程 + 有界队列，丢旧帧追最新），
> 使帧率由最慢单级决定而非全流程累加。可调 `--queue-size`（默认 2）平衡吞吐与延迟。
> 若摄像头打不开，确认用户组权限：`sudo usermod -aG video $USER` 后重新登录。

---

## 六、输出结果说明

`test_single.py` / `benchmark.py` 默认输出到 `Output/<时间戳>/` 子目录（可用 `--output` 更改）：

```
Output/20260826_171651/
├── 00_original.jpg      原图
├── 01_det_input.jpg     送入 RTMDet 的 letterbox 图（480×640）
├── 02_pose_input.jpg    RTMPose 的仿射裁剪输入（256×192）
├── 03_result.jpg        可视化结果（检测框 + 37 穴位 + 名称 + 骨架）
└── result.json          结构化结果
```

`result.json` 字段：

```json
{
  "image_size": {"h": 1706, "w": 1280},
  "boxes": [[x1, y1, x2, y2, score], ...],
  "keypoints": [{"index": 0, "name_en": "dazhui", "name_cn": "大椎",
                 "x": 599.59, "y": 586.86, "score": 0.9111}, ...],
  "elapsed_ms": {"det": 26.52, "pose": 11.12, "total": 37.64}
}
```

`main.py` 默认输出到 `Scripts/tools/runs/`：每张图 `xxx_result.jpg` + `xxx.json`；
视频模式输出 `cam_result.mp4`（摄像头时）或 `<文件名>_result.mp4`。

---

## 七、验证标准（部署是否打通）

用 `TestData/test1.jpg`（1280×1706）运行 `test_single.py`，期望：

| 检查项 | 期望值 |
|--------|--------|
| 检测 score | `> 0.85`（实测约 0.90） |
| 检测框 | 包住整个背部，`x∈[277,1112], y∈[591,1729]`（原图坐标，允许 ±20px） |
| 关键点 | 37 个穴位基本全部有效（`conf ≥ 0.3` 且 `x ≥ 0`），全部落在框内 |
| char 坐标范围 | 框/点在原图 `[0,1280)×[0,1706)` 内 |
| 端到端延迟 | 板端实测约 **37 ms** 量级（det≈26ms + pose≈11ms，量纲随模型版本波动） |

摄像头 640×480 实时流：背部占据画面时 score 应 `> 0.9`，画框包住背部无偏移。

---

## 八、常见问题排查

| 症状 | 根因 | 处理 |
|------|------|------|
| `ValueError: cannot reshape array of size 500 into shape (1,1000,6)` | 加载了带 NMS 的旧 OM | 使用 no-NMS 版 `RTMDet_Tiny.om`；`DET_OUTPUT_SHAPE=(1,1000,6)` 不要改 |
| 有框但 score≈0.63、框位置离谱（如 `[0,1343,74,1702]`） | `preprocess_rtmdet_raw` 返回了 NCHW | 改成返回 **HWC uint8**（禁 transpose） |
| 完全不出框 / score≈0.03~0.27 | 图内 pad 用了 114（旧导出模型） | 换 pad=0 的 OM；板端 letterbox 外部 pad 用 128 |
| 框与关键点坐标错位 | 忘了 letterbox 逆映射 | 用 `inv_letterbox(boxes, r, pad_left, pad_top)` 回原图并 clip |
| 多线程下偶发推理异常/段错误 | 子线程未绑定 ACL context | worker 线程启动时调用 `bind_thread_context()` |
| `dcmi module initialize failed` | npu-smi 驱动未初始化（不影响推理） | 只要 `acl` 能 import 且脚本推理正常即可忽略 |
| `import acl` 失败 | 未加载 CANN 环境 | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |

---

## 九、相关文档

| 文档 | 说明 |
|------|------|
| [`OM_Deploy_Notes.md`](OM_Deploy_Notes.md) | 板端 AIPP 版部署坑位清单、AIPP 归一化参数、坐标空间约定、验证标准 |
| [`InferenceGuide.md`](InferenceGuide.md) | ONNX 模型元信息与预处理/后处理规范 |
| `AcuPointDet-Deployment-Plan.md` | 原始需求分析、atc 转换方案、任务分解（历史文档） |