# TuinaDex Docker — 部署与硬件访问

## 快速启动

```bash
cd TuinaDex/docker

# 1. 构建镜像
docker compose --profile control build

# 2. 启动 (control profile: arm + hand)
docker compose --profile control up -d

# 3. 进入 arm 容器
docker compose --profile control exec arm bash
conda activate smolvla
cd /workspace/TuinaDex/Arm-robot_VLA
python scripts/zdt_bringup.py status

# 4. 进入 hand 容器
docker compose --profile control exec hand bash
conda activate leap_hand
cd /workspace/TuinaDex/Leap_Hand/python
python gesture_mapping/demo_realtime.py
```

## 镜像清单

| 镜像 | 大小(估) | 启动条件 |
|------|---------|----------|
| `tuinadex/arm:0.1` | ~8 GB | 含 lerobot + python-can[socketcan] + mujoco + torch (CUDA 12.8) |
| `tuinadex/hand:0.1` | ~5 GB | 含 dynamixel-sdk + mediapipe + opencv + pyrealsense2 |
| `tuinadex/vla:0.1` | =arm | VLA 训练/推理专用 container |
| `tuinadex/sim:0.1` | =arm | MuJoCo 仿真桥接 |

> ⚠ 实际大小取决于 base 镜像。`nvidia/cuda:12.8.0-runtime` 约 1.5 GB，加上 Python/Conda 依赖后可超过上述估算。

## 硬件访问详细

### 1. ZDT SocketCAN (`can0`, 500kbps)

**为什么需要 `--network=host`？**
SocketCAN 是 Linux 内核网络协议族（`AF_CAN`），其 socket 与**网络命名空间**绑定。
容器默认在独立 netns 内，看不到 host 的 `can0`。`--network=host` 让容器共享 host netns。

```yaml
# compose.yaml
arm:
  network_mode: host
  cap_add: [NET_ADMIN]   # 容器内能 ip link set can0 up / bitrate
```

**Host 端初始化（推荐在 host 上做，容器共享）：**
```bash
sudo modprobe can
sudo modprobe can_raw
sudo modprobe vcan              # 可选: 虚拟 CAN 用于仿真
sudo ip link add dev can0 type can bitrate 500000
sudo ip link set up can0
# 持久化: 见 Arm-robot_VLA/scripts/can_setup.sh
```

**容器内可用命令：**
```bash
ip link show can0
candump can0                   # can-utils, 实时帧流
cansend can0 123#DEADBEEF
```

### 2. LEAP Hand USB (`/dev/ttyUSB0`)

**device 透传（推荐，不需 `--privileged`）：**
```yaml
hand:
  devices:
    - /dev/ttyUSB0:/dev/ttyUSB0
```

**Host 端 udev 规则（避免每次 sudo chmod 666）：**
```bash
sudo cp docker/scripts/99-leap-hand.rules /etc/udev/rules.d/
sudo udevadm control --reload && sudo udevadm trigger
```

**验证：**
```bash
ls -l /dev/ttyUSB0
# 应该: crw-rw---- 1 root dialout ...
```

### 3. Intel RealSense (D455)

```yaml
hand:
  devices:
    - /dev/bus/usb:/dev/bus/usb   # 整个 USB 总线 (最简单)
    # 或更精细: 找到 D455 的具体 bus/dev:
    # - /dev/bus/usb/002/003:/dev/bus/usb/002/003
```

容器内 `pyrealsense2` 应能识别。`rs-enumerate-devices` 输出即代表成功。

### 4. 手柄 (joystick_control)

USB 手柄不需要特殊处理——`pygame.joystick` 通过 evdev 读取，host 设备足够。
**建议**把 `/dev/input` 透传：
```yaml
arm:
  devices:
    - /dev/input:/dev/input
```

## 安全约束

**绝不使用 `--privileged`**——除非确实需要 root 内核能力。TuinaDex 当前编排**不**使用。

| 容器 | privileged | cap_add | network | 说明 |
|------|-----------|---------|---------|------|
| arm | false | NET_ADMIN | host | CAN 仅需 NET_ADMIN |
| hand | false | (无) | bridge | USB 透传够用 |
| vla-train | false | (无) | bridge | 训练无硬件访问 |
| sim-bridge | false | (无) | host | 仿真 TCP 也用 host |

## 数据流

```
主机 docker host                          容器
─────────────                          ─────────────
/dev/shm/mujoco_frame_*  ←──────────────  /dev/shm/mujoco_frame_*  (跨容器共享内存)
can0 (SocketCAN)         ←──────────────  ip link show can0
/dev/ttyUSB0 (LEAP Hand) ←──────────────  /dev/ttyUSB0
NVIDIA GPU               ←──────────────  CUDA 12.8 (runtime)
HOST datasets/checkpoints ─────────────→  /workspace/datasets  (named volume)
```

## 故障排查

| 症状 | 排查 |
|------|------|
| `arm` 容器内 `ip link show can0` 无输出 | 1) host 执行 `sudo ip link show can0`；2) 容器 `network_mode: host` 是否生效；3) `cap_add: NET_ADMIN` 是否给 |
| `hand` 容器内 `dynamixel_sdk` 找不到端口 | 1) `ls /dev/ttyUSB*`；2) udev 规则；3) `devices` 配置 |
| GPU 不可用 | 1) `nvidia-container-toolkit` 安装；2) `nvidia-smi` 在容器内可用 |
| 容器内 X11 显示 | `docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro` |
| 共享内存跨容器 | `/dev/shm` 默认挂载；如有问题显式 `-v /dev/shm:/dev/shm` |

## 进一步

- 仿真模式: `--profile sim up`（仅 sim-bridge，无硬件）
- VLA 训练: `--profile vla up`（加 vla-train 容器，GPU 训练）
- 自定义启动命令: `command:` 字段覆盖默认

参见 `TuinaDex/README.md` 与 `Arm-robot_VLA/docs/2026-08-14-zdt-driver-setup.md`。