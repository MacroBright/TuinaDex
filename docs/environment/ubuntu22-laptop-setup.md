# Ubuntu 22.04 笔记本配置指南

这台 Ubuntu 笔记本用于连接机械臂、LEAP Hand、USB 串口和 RealSense。
训练、批量数据处理和其他 GPU 重任务继续放在服务器上；Mac 只负责 SSH、Git
和代码查看。

## 1. 安装基础工具

先在 GitHub 账号中添加笔记本的 SSH 公钥，然后安装 Git 和 Miniconda：

```bash
sudo apt update
sudo apt install -y git curl build-essential
ssh -T git@github.com
```

Miniconda 安装完成后重新打开终端，并确认：

```bash
git --version
conda --version
python3 --version
```

## 2. 下载项目和子模块

```bash
mkdir -p ~/projects
cd ~/projects
git clone --recurse-submodules git@github.com:Brasaking1/TuinaDex.git
cd TuinaDex
git switch codex/laptop-bootstrap
git submodule update --init --recursive
git submodule status
```

`git submodule status` 应同时列出 `Arm-robot_VLA` 和 `Leap_Hand`。不要分别复制
三个目录；根仓库已经固定了两个子模块的正确提交。

## 3. 创建环境

机械臂/VLA 环境使用 Python 3.10：

```bash
conda env create -f environments/arm_vla.yml
conda activate arm_vla
python -m pip install --no-deps --config-settings editable_mode=compat \
  -e ./Arm-robot_VLA/lerobot_robot_massage
python -m pip install --no-deps -e .
python -m pip check
```

不要运行当前的 `scripts/setup_dev.sh`：上游脚本的 editable 参数存在拼写问题，
而且会把机械臂和灵巧手依赖混装到同一个环境。

灵巧手/Jupyter 环境使用 Python 3.14：

```bash
conda env create -f environments/huawei_contest.yml
conda activate huawei_contest
python -m ipykernel install --user \
  --name huawei_contest \
  --display-name "Python (huawei_contest)"
```

## 4. 无硬件验证

机械臂环境：

```bash
conda activate arm_vla
export PYTHONPATH="$PWD/Arm-robot_VLA${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch, lerobot, can, mujoco; print(torch.__version__, lerobot.__version__)"
python -m compileall -q Arm-robot_VLA/lerobot_robot_massage Arm-robot_VLA/scripts
```

灵巧手环境：

```bash
conda activate huawei_contest
export PYTHONPATH="$PWD/Leap_Hand/python${PYTHONPATH:+:$PYTHONPATH}"
python -c "import main, leap_hand_utils, gesture_mapping; print('LEAP imports OK')"
python -m pytest -q -rs Leap_Hand/python/tests
```

没有 RealSense 或 HaMeR 时，相关测试显示 `skipped` 是正常现象；其他测试不应失败。

## 5. 硬件验证顺序

硬件第一次接入必须按以下顺序逐项完成，不要直接运行联合控制：

1. 只接 LEAP Hand 通信 USB，不给电机发送动作，确认串口设备名。
2. 接入 5 V 电源，确认 16 个 Dynamixel ID 都能读取。
3. 验证断扭矩、急停和全开姿势，再做单关节低速动作。
4. 接入 RealSense，确认彩色图、深度图和时间戳。
5. 单独验证机械臂状态读取、急停和小幅低速动作。
6. 完成以上检查后，再测试机械臂 6 轴与灵巧手 16 轴的联合接口。

串口权限应使用项目提供的 udev 规则，而不是每次执行 `sudo chmod 666`：

```bash
sudo cp docker/scripts/99-leap-hand.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules
sudo udevadm trigger
```

重新插拔设备后，用以下命令确认设备权限：

```bash
ls -l /dev/ttyUSB* /dev/serial/by-id/* 2>/dev/null
```

## 6. 日常同步

```bash
cd ~/projects/TuinaDex
git pull --ff-only
git submodule update --init --recursive
```

环境文件发生变化时优先新建环境验证，不要直接改动已经能工作的环境。数据集、
训练 checkpoint 和模型权重保存在服务器或专用模型仓库，不提交到普通 Git 历史。
