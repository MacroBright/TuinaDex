# TuinaDex — 机械臂 + 灵巧手统一机器人系统

> 基于 VLA（Vision-Language-Action）的机械臂灵巧手按摩系统，统一整合：
>
> - **[Arm-robot_VLA](./Arm-robot_VLA)** — 6-DOF 机械臂（ZDT 步进闭环，PC 直连 SocketCAN，ADR-005）
> - **[Leap_Hand](./Leap_Hand)** — 14-DOF 灵巧手（LEAP Hand V1，Dynamixel XC330×16，PC 直连 USB）

## 集成架构

```
TuinaDex/                          ← 本仓库（顶层协调 + 跨子系统包装层）
├── apps/                          ← 跨子系统应用层 (主战场)
│   ├── massage/                   ← 22-DOF 机械臂+灵巧手 (MassageRobotV2)
│   └── teleop/                    ← 视觉遥操（占位）
├── packages/                      ← 跨子系统共享工具 (占位)
│   ├── common_math/               ← 数学工具 (rotation / IK / 滤波)
│   └── common_vis/                ← 视觉工具 (camera / 关键点 / 共享内存帧)
├── docker/                        ← Docker 多 Container 编排
│   ├── Dockerfile.arm              ← 镜像: arm-control（lerobot + zdt + python-can）
│   ├── Dockerfile.hand             ← 镜像: hand-control（dynamixel_sdk + mediapipe）
│   └── compose.yaml               ← 多 Container 编排（arm/hand 独立进程）
├── tools/                         ← 顶层脚本
│   └── bump_submodules.sh         ← 一键 bump submodule 引用
├── scripts/                       ← 顶层运维
│   └── setup_dev.sh               ← 一键安装开发环境
├── docs/                          ← 跨项目集成文档
├── pyproject.toml                 ← 顶层构建入口 (umbrella 包)
├── Arm-robot_VLA/                 ← git submodule: 机械臂（6-DOF，ZDT CAN 直连）
└── Leap_Hand/                     ← git submodule: 灵巧手（16-DOF，USB 直连）
```

## 架构决策（合并自各子项目 ADR）

| 决策 | 来源 | 说明 |
|------|------|------|
| 机械臂 PC 直连 ZDT 驱动器（不走 STM32 网关） | Arm ADR-005 | SocketCAN `can0` @500kbps；安全层在 `zdt/controller.py` |
| 灵巧手 PC 直连 USB | Hand ADR-001 | Dynamixel Protocol 2.0，4 Mbps |
| 双镜像 + submodule | 本项目 | 两子系统独立镜像、独立部署；submodule 维持独立演进 |
| 多 Container 编排 | 本项目 | arm/hand 分容器解耦，共享数据集卷与共享内存 |
| LeRobot 数据采集 | `MassageRobotV2` 计划（跨项目） | 22 DOF 合并 Robot 子类，observation.state = 22 |

> **回滚策略**：TuinaDex 顶层仓库不强制 submodules 同步；两个 submodule 各自独立 push 到 `MacroBright/Arm-robot_VLA` 与 `MacroBright/Leap_hand`。任意模块改动均可单独回滚。

## 快速开始

### 前置条件

- Ubuntu 22.04+（NVIDIA GPU 推荐，CPU-only 可跑仿真）
- Docker Engine 24+，Docker Compose v2
- NVIDIA Driver + nvidia-container-toolkit（GPU 训练需要）
- USB 设备：机械臂 CAN 适配器、灵巧手 USB、RealSense 相机（如有）

### 初始化 submodule

```bash
git clone --recurse-submodules https://github.com/MacroBright/TuinaDex.git
cd TuinaDex
bash scripts/setup_dev.sh   # 一键安装顶层 + 两个 submodule 的 editable
```

### 启动 Docker 编排

```bash
# 仅启动机械臂 + 灵巧手控制容器（不带 VLA）
cd docker
docker compose --profile control up

# 含 VLA 推理（GPU）
docker compose --profile vla up

# 仿真模式（不接硬件，MuJoCo 仿真）
docker compose --profile sim up
```

### 硬件访问（host 设备透传到容器）

| 设备 | Host 路径 | Container 挂载 |
|------|-----------|----------------|
| ZDT CAN | `can0`（SocketCAN） | `--network=host`（SocketCAN 必需 host netns） |
| LEAP Hand | `/dev/ttyUSB0` | `--device /dev/ttyUSB0:/dev/ttyUSB0` |
| RealSense | `/dev/video*` + `usbfs` | `--device /dev/video0` + privileged UVC |

详见 [`docker/README.md`](./docker/README.md)。

## 协同工作流（真机数据采集）

参见 [`docs/integration-data-collection.md`](./docs/integration-data-collection.md)：
视觉遥操（D455 + MediaPipe/HaMeR）→ 臂末端跟随 + 手 16-DOF 跟随 → `MassageRobotV2` 写入 LeRobotDataset。

## Submodule 独立工作流

```bash
# 在 submodule 内工作（独立 commit、独立 push）
cd Arm-robot_VLA
git checkout -b feat/my-feature
git commit -m "feat: ..."
git push origin feat/my-feature

# 回到顶层一键 bump
cd ..
bash tools/bump_submodules.sh Arm-robot_VLA
git add Arm-robot_VLA
git commit -m "chore(submodule): Arm-robot_VLA → new commit"
git push
```

## 版本与分支

| 模块 | URL | 默认跟踪 | 默认 commit |
|------|-----|---------|-------------|
| Arm-robot_VLA | https://github.com/MacroBright/Arm-robot_VLA.git | default branch | 由 GitHub 决定 |
| Leap_Hand | https://github.com/MacroBright/Leap_hand.git | default branch (`main`) | `2aef547`（README 重写） |

> ⚠ **分支约束**：submodule 默认跟踪上游 default branch（不固定本地活跃分支），
> 以避免对未 push 分支的依赖。子项目本地活跃分支 `feat/arm-visual-teleop` 上的未 push
> commit **对 TuinaDex 顶层不可见**，待各自 push 后 submodule 引用才会更新。
> 推荐用法：子项目内完成开发 → push → 在 TuinaDex 内 `git submodule update --remote <name>`。

## License

各子模块沿用各自上游 license（见 `Arm-robot_VLA/LICENSE`（如适用）与 `Leap_Hand/LICENSE`）。