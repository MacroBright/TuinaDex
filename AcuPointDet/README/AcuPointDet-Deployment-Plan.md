# 穴位识别模型部署需求与计划文档

> 文档类型：技术设计文档（TDD）
> 目标平台：香橙派 AI Pro（昇腾 310B）
> 创建日期：2026-08-08
> 文档版本：v1.0

---

## 1. 项目背景与目标

### 1.1 项目背景

中医穴位识别是智能中医辅助诊断的核心环节。本项目已基于 RTMDet（目标检测）与 RTMPose（关键点估计）两阶段级联方案完成模型训练，现需将训练好的模型部署到香橙派 AI Pro 边缘设备上，实现端侧实时穴位识别推理。

### 1.2 项目目标

| 目标编号 | 描述 | 验收标准 |
|---------|------|---------|
| G-1 | 模型格式转换 | 两个 ONNX 模型成功转换为昇腾 .om 离线模型，atc 转换无报错 |
| G-2 | 推理脚本编写 | 完成端到端推理脚本，支持图像输入并输出穴位坐标与置信度 |
| G-3 | 级联推理打通 | RTMDet 检测结果正确喂入 RTMPose，完成两阶段串联 |
| G-4 | 推理性能达标 | 单张图像端到端推理延迟满足实时性要求（建议 ≤ 200ms） |

### 1.3 业务流程

输入图像 → RTMDet 检测穴位区域（bbox）→ 裁剪 ROI → RTMPose 估计关键点 → 输出穴位坐标与可视化结果

---

## 2. 环境信息

### 2.1 硬件环境

| 项目 | 信息 |
|------|------|
| 设备 | 香橙派 AI Pro |
| 架构 | aarch64（ARM64）|
| AI 芯片 | 昇腾 310B（Ascend 310B）|
| 操作系统 | Ubuntu 22.04.5 LTS（Jammy）|
| 内核 | Linux 5.10.0+ SMP |

### 2.2 软件环境

| 组件 | 版本/路径 | 状态 |
|------|----------|------|
| 昇腾工具链 | CANN 8.0.0（/usr/local/Ascend/ascend-toolkit/8.0.0）| 已安装 |
| atc 转换工具 | /usr/local/Ascend/ascend-toolkit/latest/bin/atc | 可用 |
| ACL Python 库 | /usr/local/Ascend/ascend-toolkit/latest/python/site-packages/acl.so | 可用 |
| Python | 3.9.2 | 可用 |
| NumPy | 2.0.2 | 可用 |
| OpenCV | 4.10.0 | 可用 |

### 2.3 目录结构

    /home/HwHiAiUser/Desktop/AcuPointDet/
    ├── Model/
    │   ├── RTMDet_Tiny.onnx      # 穴位区域检测模型（约 21.2 MB）
    │   └── RTMPose_s.onnx        # 穴位关键点估计模型（约 22.8 MB）
    ├── Scripts/                  # 推理脚本目录（待编写）
    └── AcuPointDet-Deployment-Plan.md  # 本文档

---

## 3. 模型信息

### 3.1 RTMDet_Tiny（检测模型）

| 属性 | 说明 |
|------|------|
| 框架来源 | MMDetection（RTMDet-Tiny）|
| 当前格式 | ONNX |
| 文件大小 | 21.2 MB |
| 功能 | 检测图像中的穴位区域，输出 bounding box + 类别 + 置信度 |
| 输入节点 | 待通过 `atc --input_shape` 指定（需用 Netron 或 onnx 工具确认具体形状）|
| 输出节点 | 待确认（通常为 boxes、scores、labels）|

### 3.2 RTMPose_s（关键点模型）

| 属性 | 说明 |
|------|------|
| 框架来源 | MMPose（RTMPose-s）|
| 当前格式 | ONNX |
| 文件大小 | 22.8 MB |
| 功能 | 在裁剪出的穴位区域内估计关键点坐标 |
| 输入节点 | 待确认（通常为单张 ROI 图像）|
| 输出节点 | 待确认（通常为关键点坐标 + 置信度）|

> **待办（TBD）**：需使用 Netron 或 `onnx` Python 包检查两个模型的输入/输出节点名称与形状，作为 atc 转换的依据。

---

## 4. 需求分析

### 4.1 功能需求

| 编号 | 需求 | 优先级 | 说明 |
|------|------|--------|------|
| FR-1 | ONNX → OM 模型转换 | 高 | 使用 atc 工具将两个 ONNX 模型转换为昇腾 .om 离线模型 |
| FR-2 | 图像预处理 | 高 | Resize、归一化、通道转换（BGR→RGB）、NCHW 排布 |
| FR-3 | RTMDet 推理 | 高 | 调用 ACL 加载 om 模型，执行检测推理 |
| FR-4 | 检测后处理 | 高 | NMS、置信度过滤、bbox 解码 |
| FR-5 | ROI 裁剪与仿射变换 | 高 | 根据检测 bbox 裁剪区域，送入 RTMPose |
| FR-6 | RTMPose 推理 | 高 | 调用 ACL 加载 om 模型，执行关键点推理 |
| FR-7 | 关键点后处理 | 高 | 坐标还原到原图、置信度过滤 |
| FR-8 | 结果可视化 | 中 | 在原图上绘制 bbox 与关键点，保存/显示结果 |
| FR-9 | 批量推理支持 | 低 | 支持对文件夹内多张图像批量处理 |
| FR-10 | 视频流推理支持 | 低 | 支持摄像头/视频文件实时推理 |

### 4.2 非功能需求

| 编号 | 需求 | 目标 |
|------|------|------|
| NFR-1 | 推理延迟 | 单图端到端 ≤ 200ms（含两阶段）|
| NFR-2 | 内存占用 | 峰值 ≤ 2GB（310B 内存受限）|
| NFR-3 | 模型精度 | 转换后精度与原 ONNX 推理结果一致（误差 ≤ 1e-3）|
| NFR-4 | 代码可维护性 | 模块化设计，预处理/推理/后处理解耦 |
| NFR-5 | 可移植性 | 脚本可在同型号设备间直接复用 |

### 4.3 约束条件

- 昇腾 310B 仅支持 FP16/INT8 量化推理，不支持 FP32（atc 转换时需指定 `--precision_mode`）。
- 310B 算子支持列表有限，部分 ONNX 算子可能不支持，需提前用 atc 试转排查。
- 设备无独立显卡，可视化结果需保存为文件或通过远程查看。

---

## 5. 技术方案

### 5.1 总体架构

    ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  输入图像    │ ──→ │  预处理模块   │ ──→ │ RTMDet 推理   │ ──→ │  检测后处理   │
    └─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                        │
                                                                        ▼
    ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
    │  结果输出    │ ←── │ 关键点后处理  │ ←── │ RTMPose 推理  │ ←── │ ROI 裁剪变换 │
    └─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘

### 5.2 模型转换方案

**转换工具**：atc（Ascend Tensor Compiler）

**转换步骤**：

1. **检查 ONNX 模型节点信息**

   使用 Python 脚本读取 ONNX 模型的输入/输出节点名称与形状：

        import onnx
        model = onnx.load("RTMDet_Tiny.onnx")
        for inp in model.graph.input:
            print(inp.name, inp.type)

2. **RTMDet 转换命令（示例）**

        atc --framework=5 \
            --model=Model/RTMDet_Tiny.onnx \
            --output=Model/RTMDet_Tiny \
            --soc_version=Ascend310B3 \
            --input_shape="input:NxCxHxW" \
            --precision_mode=force_fp16 \
            --log=info

   - `--framework=5`：表示 ONNX 框架
   - `--soc_version=Ascend310B3`：香橙派 AI Pro 芯片型号（需确认是 310B1 还是 310B3）
   - `--precision_mode=force_fp16`：强制 FP16 精度

3. **RTMPose 转换命令（示例）**

        atc --framework=5 \
            --model=Model/RTMPose_s.onnx \
            --output=Model/RTMPose_s \
            --soc_version=Ascend310B3 \
            --input_shape="input:NxCxHxW" \
            --precision_mode=force_fp16 \
            --log=info

4. **转换验证**

   - 检查 atc 输出日志无 ERROR
   - 确认 .om 文件已生成
   - 用 ACL 加载 .om 模型，执行单次推理验证可用性

### 5.3 推理脚本方案

**技术选型**：Python + ACL（Ascend Computing Language）

**模块划分**：

| 模块 | 文件 | 职责 |
|------|------|------|
| ACL 封装层 | `acl_utils.py` | 封装 ACL 初始化、模型加载、推理执行、资源释放 |
| 预处理模块 | `preprocess.py` | 图像读取、Resize、归一化、通道转换、NCHW |
| RTMDet 推理 | `det_infer.py` | 加载检测 om 模型，执行推理，解析 bbox |
| RTMPose 推理 | `pose_infer.py` | 加载关键点 om 模型，执行推理，解析关键点 |
| 后处理模块 | `postprocess.py` | NMS、bbox 过滤、关键点坐标还原 |
| 可视化模块 | `visualize.py` | 绘制 bbox、关键点、连线，保存结果图 |
| 主入口 | `main.py` | 串联全流程，支持单图/批量/视频输入 |

**ACL 推理核心流程**：

1. `acl.init()` → 初始化 ACL
2. `acl.rt.set_device(0)` → 设置设备
3. 加载 .om 模型 → 获取模型输入/输出描述
4. 申请输入/输出 device 内存
5. 预处理数据 → H2D 拷贝 → 执行推理 → D2H 拷贝 → 后处理
6. 释放资源

### 5.4 级联推理数据流

    原图 (H×W×3)
        │ 预处理（Resize 到 det_input_size, 归一化, NCHW）
        ▼
    RTMDet 推理 → boxes[N, 5]  (x1, y1, x2, y2, score)
        │ 按 score 过滤 + NMS
        ▼
    对每个 box:
        │ 在原图上裁剪 ROI → Resize 到 pose_input_size
        ▼
    RTMPose 推理 → keypoints[K, 3]  (x, y, score)
        │ 坐标从 ROI 空间还原到原图空间
        ▼
    最终输出: { box, keypoints }

---

## 6. 实施计划与任务分解

### 6.1 任务清单

| 阶段 | 任务编号 | 任务描述 | 产出物 | 状态 |
|------|---------|---------|--------|------|
| 一、模型准备 | T-1 | 使用 onnx 工具检查两个模型的输入/输出节点与形状 | 节点信息记录 | 待开始 |
| 一、模型准备 | T-2 | 确认香橙派 AI Pro 的 SoC 版本（310B1/310B3）| SoC 版本 | 待开始 |
| 二、模型转换 | T-3 | 编写并执行 RTMDet 的 atc 转换命令 | RTMDet_Tiny.om | 待开始 |
| 二、模型转换 | T-4 | 编写并执行 RTMPose 的 atc 转换命令 | RTMPose_s.om | 待开始 |
| 二、模型转换 | T-5 | 验证 .om 模型可被 ACL 正常加载与推理 | 验证脚本与结果 | 待开始 |
| 三、推理脚本 | T-6 | 编写 `acl_utils.py` ACL 封装层 | acl_utils.py | 待开始 |
| 三、推理脚本 | T-7 | 编写 `preprocess.py` 预处理模块 | preprocess.py | 待开始 |
| 三、推理脚本 | T-8 | 编写 `det_infer.py` 检测推理模块 | det_infer.py | 待开始 |
| 三、推理脚本 | T-9 | 编写 `pose_infer.py` 关键点推理模块 | pose_infer.py | 待开始 |
| 三、推理脚本 | T-10 | 编写 `postprocess.py` 后处理模块 | postprocess.py | 待开始 |
| 三、推理脚本 | T-11 | 编写 `visualize.py` 可视化模块 | visualize.py | 待开始 |
| 三、推理脚本 | T-12 | 编写 `main.py` 主入口，串联全流程 | main.py | 待开始 |
| 四、测试验证 | T-13 | 单图端到端测试，确认输出正确 | 测试报告 | 待开始 |
| 四、测试验证 | T-14 | 性能测试，测量推理延迟与内存占用 | 性能数据 | 待开始 |
| 四、测试验证 | T-15 | 精度对齐测试，对比 OM 与 ONNX 推理结果 | 精度对比报告 | 待开始 |

### 6.2 里程碑

| 里程碑 | 内容 | 关键交付物 |
|--------|------|-----------|
| M1 | 模型转换完成 | 两个 .om 模型 + 转换日志 |
| M2 | 推理脚本完成 | Scripts/ 下完整脚本集 |
| M3 | 测试验证完成 | 测试报告 + 性能数据 + 精度报告 |

---

## 7. 风险与应对措施

| 风险编号 | 风险描述 | 影响 | 应对措施 |
|---------|---------|------|---------|
| R-1 | ONNX 模型含 310B 不支持的算子 | 转换失败 | 用 atc 试转，根据报错定位算子；考虑用 ONNX Simplifier 简化或替换等价算子 |
| R-2 | SoC 版本填写错误 | 转换失败或推理异常 | 通过 `npu-smi info` 或文档确认具体型号（310B1/310B3）|
| R-3 | FP16 精度损失导致检测/关键点偏移 | 精度下降 | 对比 FP16 与原 ONNX 结果；必要时尝试 `allow_mix_precision` |
| R-4 | 输入形状动态导致转换失败 | 转换失败 | 固定 input_shape，或使用 dynamic_dims 支持动态 batch |
| R-5 | 两阶段级联 bbox 裁剪逻辑与训练时不一致 | 精度下降 | 严格复用 MMDetection/MMPose 的预处理与 bbox 处理逻辑 |
| R-6 | 310B 内存不足导致模型加载失败 | 推理崩溃 | 监控内存；必要时减小输入分辨率或分时加载模型 |

---

## 8. 待确认事项（TBD）

| 编号 | 事项 | 确认方式 |
|------|------|---------|
| TBD-1 | RTMDet_Tiny.onnx 的输入/输出节点名称与形状 | 用 onnx Python 包或 Netron 检查 |
| TBD-2 | RTMPose_s.onnx 的输入/输出节点名称与形状 | 用 onnx Python 包或 Netron 检查 |
| TBD-3 | 香橙派 AI Pro 的具体 SoC 版本（310B1 / 310B3）| 查阅设备文档或 `npu-smi info` |
| TBD-4 | 模型训练时的预处理参数（mean、std、resize 尺寸、是否 BGR）| 查阅训练配置文件 |
| TBD-5 | 穴穴类别列表与关键点定义 | 查阅训练数据集标注 |
| TBD-6 | RTMPose 输入是否需要 bbox 扩展（expand_factor）| 查阅 MMPose 配置 |

---

## 9. 参考资料

- 昇腾 CANN 开发者文档：https://www.hiascend.com/document
- atc 工具使用指南：CANN 8.0.0 配套文档
- ACL Python API：/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/
- MMDetection RTMDet：https://github.com/open-mmlab/mmdetection
- MMPose RTMPose：https://github.com/open-mmlab/mmpose
- 香橙派 AI Pro 官方文档：http://www.orangepi.cn

---

## 10. 变更记录

| 版本 | 日期 | 变更内容 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-08-08 | 初始版本，完成需求分析与部署计划 | - |