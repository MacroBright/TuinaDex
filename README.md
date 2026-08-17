# TuinaDex

> 把分散在两个仓库的机械臂和灵巧手装到一起——6-DOF 机械臂 + 16-DOF 灵巧手 = 22-DOF 按摩机器人，按 VLA（Vision-Language-Action）路线跑通采集、训练、推理。
>
> 顶层仓库做"包装层 + 编排"，子模块独立演进。两个子模块都从原仓库迁移过来：
>
> - **[Arm-robot_VLA](./Arm-robot_VLA)** — 6-DOF 机械臂（STM32F407 + ZDT 步进闭环，PC 直连 SocketCAN, ADR-005）
> - **[Leap_Hand](./Leap_Hand)** — 16-DOF 灵巧手（LEAP Hand V1, Dynamixel XC330 × 16，PC 直连 USB, ADR-001）

---

## 目录

```
TuinaDex/
├── apps/                          ← 跨子系统应用
│   ├── massage/                   ← 22-DOF RobotV2 (LeRobot Robot 子类)
│   └── teleop/                    ← 视觉遥操入口
├── packages/                      ← 跨子系统共享工具
│   ├── common_math/               ← 旋转 / IK / 滤波
│   └── common_vis/                ← 相机 / 关键点 / 共享内存帧
├── docker/                        ← 多 Container 编排
│   ├── Dockerfile.arm             ← 镜像: tuinadex/arm:0.1
│   ├── Dockerfile.arm.cpu         ← 镜像: tuinadex/arm:0.1-cpu (无 CUDA)
│   ├── Dockerfile.hand            ← 镜像: tuinadex/hand:0.1
│   ├── compose.yaml               ← profiles: control / vla / sim
│   ├── scripts/99-leap-hand.rules ← LEAP Hand / RealSense udev
│   └── conda/smolvla-history.yml  ← arm 容器 conda env base
├── tools/
│   └── bump_submodules.sh         ← 一键 bump submodule 引用
├── scripts/
│   └── setup_dev.sh               ← 一键安装开发环境（含两个 submodule editable）
├── docs/                          ← 跨项目集成文档（待 Phase A 启动后建立）
├── pyproject.toml                 ← 顶层 umbrella 包（`tuinadex`）
│
├── Arm-robot_VLA/                 ← git submodule: 机械臂
└── Leap_Hand/                     ← git submodule: 灵巧手
```

---

## 关键决策

| 决策 | 来源 | 落实 |
|------|------|------|
| 机械臂 PC 直连 ZDT 驱动器（不走 STM32 网关） | Arm ADR-005 | SocketCAN `can0` @ 500 kbps；安全层在 `zdt/controller.py`；Compose 容器 `network_mode: host` |
| 灵巧手 PC 直连 USB | Hand ADR-001 | Dynamixel Protocol 2.0, 4 Mbps；Compose 容器 `--device /dev/ttyUSB0:/dev/ttyUSB0` |
| 顶层用双镜像 + submodule | 本项目 | arm / hand 各自独立容器、独立部署；submodule 维持原仓库节奏独立 push |
| 22-DOF 合并 Robot | 跨项目 | `apps/massage/massage_robot_v2.py`（占位骨架）：observation.state = 22，急停同时触发两个子系统 |
| 共享数学/视觉工具不复制代码 | 本项目 | `packages/common_*` 只声明包；跨项目复用走 `sys.path` 注入 submodule 内部模块 |

> 回滚靠 submodule：`MacroBright/Arm-robot_VLA` 与 `MacroBright/Leap_hand` 继续作为引用源，任意模块改动单独回滚，不牵动顶层。

---

## 快速开始

### 前置

- Ubuntu 22.04+（CPU 也能跑仿真；GPU 训练需 NVIDIA Driver + nvidia-container-toolkit）
- Docker Engine 24+，Docker Compose v2
- 硬件：机械臂 CAN 适配器、LEAP Hand USB、RealSense D455（可选）

### 安装

```bash
git clone --recurse-submodules https://github.com/MacroBright/TuinaDex.git
cd TuinaDex
bash scripts/setup_dev.sh   # 装两个 submodule 的 editable + 顶层 tuinadex 包
```

### 启动 Docker 编排

```bash
cd docker

# 仅控制容器 (arm + hand, 无 VLA 训练)
docker compose --profile control up

# 含 VLA 训练/推理 (GPU)
docker compose --profile vla up

# 纯仿真 (MuJoCo 桥接, 不接硬件)
docker compose --profile sim up
```

### 硬件访问 (host → 容器)

| 设备 | Host 路径 | 容器挂载 |
|------|----------|----------|
| ZDT CAN | `can0` (SocketCAN) | `--network=host`（CAN socket 跟 netns 绑，必须共享 host netns） |
| LEAP Hand | `/dev/ttyUSB0` | `--device /dev/ttyUSB0:/dev/ttyUSB0` + udev 规则 |
| RealSense | `/dev/bus/usb` | 整段 USB 总线透传 + udev 规则 |

详见 [docker/README.md](./docker/README.md)。

### 当前阶段 (占位骨架)

`MassageRobotV2` 接口已定义（22-DOF LeRobot Robot + 双总线 + 急停），但具体 `connect` / `get_observation` / `send_action` 还没实现，等 Phase A 启动后填。Phase A–D 路线图见 [Leap_Hand/docs/plans/2026-08-14-lerobot-integration-data-collection.md](./Leap_Hand/docs/plans/2026-08-14-lerobot-integration-data-collection.md)。

---

## 子模块独立工作流

```bash
# 在 Arm-robot_VLA 内开发
cd Arm-robot_VLA
git checkout -b feat/my-feature
git commit -m "feat: ..."
git push origin feat/my-feature

# 回到顶层 bump 引用
cd ..
bash tools/bump_submodules.sh Arm-robot_VLA
git add Arm-robot_VLA
git commit -m "chore(submodule): Arm-robot_VLA → new commit"
git push
```

`Leap_Hand` 步骤相同。

---

## 当前跟踪

| 模块 | URL | 跟踪分支 | HEAD |
|------|-----|---------|------|
| Arm-robot_VLA | https://github.com/MacroBright/Arm-robot_VLA.git | `feat/arm-visual-teleop` | `9296386` |
| Leap_Hand | https://github.com/MacroBright/Leap_hand.git | `feat/arm-visual-teleop` | `2cbed55` |

> 子模块目录里看到的活跃分支是本地 `feat/arm-visual-teleop`，未 push 的 commit 对顶层不可见。子模块内完成开发 → push → 在顶层 `git submodule update --remote <name>`。

---

## License

- 顶层 `tuinadex` 包：Apache-2.0（见 [pyproject.toml](./pyproject.toml)）
- [Arm-robot_VLA](./Arm-robot_VLA)：沿用上游（仓库内 `README.md` 未单独声明，按上游默认）
- [Leap_Hand](./Leap_Hand)：CC BY-NC 4.0（见 [Leap_Hand/LICENSE](./Leap_Hand/LICENSE)）—— **非商业使用限制**

---

## 迁移历史

| 日期 | 事件 |
|------|------|
| 2026-08-17 | `Arm-robot_VLA` 与 `Leap_Hand` 合并入 `TuinaDex`，作为 git submodule 保留 |
| 2026-08-17 | 顶层包装层建立：`apps/` / `packages/` / `tools/` / `scripts/` / `docker/` / `pyproject.toml` |
| 2026-08-17 | 多 Container 编排骨架（arm / hand / vla-train / sim-bridge）+ 硬件访问指南 |

> 原分散仓库保留说明：`MacroBright/Arm-robot_VLA` 与 `MacroBright/Leap_hand` 仓库作为 submodule 引用源保留；后续可在 GitHub 上 archive（建议在确认 TuinaDex 工作流稳定 1–2 周后执行）。本地副本已移至 `/tmp/tuinadex_local_stash/`。
>
> 回滚原仓库的开发流：
> ```bash
> mv /tmp/tuinadex_local_stash/Arm-robot_VLA <原路径>
> mv /tmp/tuinadex_local_stash/Leap_Hand     <原路径>
> ```
