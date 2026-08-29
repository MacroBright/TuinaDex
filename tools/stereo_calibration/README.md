# 双目相机标定工具

## 1. 这个工具做什么、不做什么

这个工具在 Ubuntu 笔记本上同时采集两台 USB 相机的棋盘格图像，通过 Mac 浏览器引导采集，并计算：

- 两台相机各自的内参和畸变参数；
- 两台相机之间的旋转、平移和估计基线；
- 双目校正矩阵、重映射表和 `Q` 矩阵；
- RMS、校正后纵向误差、疑似离群组和校正预览图。

它还能在标定验收后生成单帧点云，或在 Ubuntu 本地窗口显示实时彩色画面、深度图和可交互点云。它不训练 LeRobot 模型，也不控制机械臂或灵巧手。

## 2. 实物检查清单

开始正式采集前逐项确认：

- 两台相机已用刚性支架固定，不会在采集中改变距离或俯仰角；
- 棋盘格为 **9 × 6 内角点**，方格边长已实测为 **35 mm**；
- 棋盘格纸面平整，没有翘边、折痕或明显反光；
- 用直尺测量两镜头光心的水平距离，精确到 1 mm，把数值写入私有配置的 `baseline_reference_mm`；
- 两台相机保持在已确认的 USB 插口，不交换线缆；
- 正式采集期间不拆、不碰、不重新调整相机。

## 3. Ubuntu 环境检查

```bash
source /home/mzq/miniconda3/etc/profile.d/conda.sh
conda activate tuinadex_hw
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"
python -m pytest --version
```

当前已验证的环境是 Ubuntu 22.04、Python 3.10、OpenCV 5 和 NumPy 2.2。

## 4. 相机预检

确认两个稳定的 by-path 都存在：

```bash
ls -l \
  /dev/v4l/by-path/pci-0000:04:00.4-usb-0:1:1.0-video-index0 \
  /dev/v4l/by-path/pci-0000:04:00.4-usb-0:2:1.0-video-index0
```

只打开、读取一组并关闭相机，不创建会话：

```bash
cd ~/projects/TuinaDex-stereo-calibration
python -m tools.stereo_calibration.main \
  --config ~/.config/tuinadex/stereo-upright-dry-run.json \
  --check-cameras
```

相机设备仍以 `1280x960`、MJPG、30 FPS 读取；数字转正后应显示逻辑画面
`960x1280`，左相机旋转 `270°`、右相机旋转 `90°`。如果设备回读或逻辑尺寸不匹配，
工具会拒绝进入正式采集。

## 5. 启动命令与预期输出

首次建议复制一份个人配置，不改仓库里的示例：

```bash
mkdir -p ~/.config/tuinadex
cp tools/stereo_calibration/example_config.json \
  ~/.config/tuinadex/stereo-upright-dry-run.json
nano ~/.config/tuinadex/stereo-upright-dry-run.json
```

启动三组试采会话：

```bash
source /home/mzq/miniconda3/etc/profile.d/conda.sh
conda activate tuinadex_hw
cd ~/projects/TuinaDex-stereo-calibration
python -m tools.stereo_calibration.main \
  --config ~/.config/tuinadex/stereo-upright-dry-run.json \
  --session stereo-upright-dry-run
```

启动成功后终端会显示：

- Ubuntu 本机 URL；
- Mac 需要执行的 SSH 隧道命令；
- 当前会话目录；
- 左右逻辑相机与 by-path 的对应关系；
- 左相机 `270°`、右相机 `90°` 的数字转正；
- 每侧逻辑画面尺寸为 `960x1280`。

如果不写 `--session`，程序会创建带微秒的时间戳会话名。

## 6. Mac SSH 隧道

先使用不随 Wi-Fi 内网 IP 变化的 mDNS 名称：

```bash
ssh -N -L 8765:127.0.0.1:8765 mzq@mzq.local
```

在 Mac 浏览器打开 [http://localhost:8765](http://localhost:8765)。

如果 `mzq.local` 在当前网络无法解析，先在 Ubuntu 运行 `hostname -I`，再把当前 IP 作为备用，例如：

```bash
ssh -N -L 8765:127.0.0.1:8765 mzq@113.54.211.57
```

IP 可能每次换网络后改变，不要把这个示例当成固定地址。服务只监听 `127.0.0.1`，不会直接暴露给同一 Wi-Fi 的其他人。

## 7. 三组试采

先不做正式标定，只验证整条采集链路：

1. 画面中心，标定板近似正对相机；
2. 画面左上，标定板略向一侧倾斜；
3. 画面右下，向反方向倾斜。

每次都要等待左右两卡显示 **54/54 且可保存**，然后停止移动标定板，再点击「保存这一组」。
如果与上一组角点变化不足 15 px，按钮会保持禁用；请明显移动、倾斜或改变距离。
第三组保存后用 `Ctrl+C` 退出，重新执行同一条命令；计数仍应是 3，下一组编号应为 4。

## 8. 正式 30 组姿态清单

新建一个正式会话，按下列分布采集：

- 12 组：画面中心、四边和四角的不同位置；
- 6 组：左倾和右倾；
- 6 组：上俯和下仰；
- 3 组：近距离，大约 550–650 mm；
- 3 组：远距离，大约 900–1000 mm。

不要只平移一块始终正对相机的标定板。位置、距离和倾角都要有变化，但标定板必须始终完整出现在两个画面中。

## 9. 质量阈值与报告解读

点击「开始标定」后，结果目录会包含 `stereo_calibration.npz`、`stereo_calibration.yaml`、`report.json`、`report.md` 和三张带水平辅助线的校正预览图。

- 左、右、双目 RMS 都应不大于 1.5 px，低于 1.0 px 更理想；
- 校正后纵向误差中位数必须小于 1.0 px；
- 校正后纵向误差 P95 必须小于 2.0 px；
- 估计基线与直尺实测值的偏差不得超过 15%；
- 平移向量必须以 X 分量为主，报告中的“水平基线”必须为“是”；
- 至少保留 18 组有效图，建议最终约 25 组；
- `suspected_outliers` 只是疑似离群组，程序不会自动删除或静默排除；
- `skipped_pairs` 记录因文件或元数据异常而未参与计算的组号和原因。

最后必须打开至少三张校正预览图：同一棋盘格角点在左右画面中应落在同一条水平辅助线上。

## 10. 恢复、重连与安全退出

- **恢复会话：** 再次使用相同 `--session` 名称和完全一致的配置。任何相机路径、棋盘格、阈值、V4L2 参数或基线变化都会拒绝恢复；
- **旧数据：** `stereo-dry-run-20260827` 使用旧方向，只保留审计，不要用新配置恢复或覆盖；
- **相机掉线：** 查看页面错误，确认 USB 和 by-path，再点击「重新连接相机」。不会删除已保存数据；
- **移除误拍：** 「移除上一组」会把最新有效组移到 `rejected/`，不是永久删除；
- **安全退出：** 在 Ubuntu 运行服务的终端按 `Ctrl+C`。程序会先关闭 HTTP 监听，再通知相机工作线程释放两个设备；
- 如果一次 V4L2 读取卡在内核中，程序会在 2 秒后明确报告未能结束，不会从另一线程强行 `release()` 而冒 OpenCV 崩溃风险。

## 11. 什么时候必须重新标定

只要任意一台相机相对另一台的位置或姿态变了，之前的外参和校正映射就不再可信。更换支架孔位、松动螺丝、调仰角、改变基线、相机被碰撞、镜头重新对焦后，都应创建新会话并完整重新标定。

## 12. USB 非硬同步局限

这两台普通 USB 相机没有硬件触发线，程序只能先对两边 `grab()`，然后尽快分别取回画面。对于静止棋盘格和静止裸露背部，这种方式足以完成标定和后续静态建模。

后续如果要重建快速移动的灵巧手或人体，左右帧的微小时差会变成明显误差。那时需要单独评估硬件同步相机或触发线，不能把当前静态标定流程直接当成动态三维方案。

## 13. Ubuntu 本地实时点云

实时界面使用 PyQt5 + PyQtGraph。只在 `tuinadex_hw` 环境首次安装：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tuinadex_hw
python -m pip install "PyQt5==5.15.11" "pyqtgraph==0.13.7" "PyOpenGL==3.1.10"
```

在 **Ubuntu 桌面本机终端** 运行，不要在普通 SSH 终端运行：

```bash
cd ~/projects/TuinaDex-stereo-calibration
python -m tools.stereo_calibration.realtime_viewer \
  --config ~/.config/tuinadex/stereo-upright-dry-run.json \
  --calibration ~/projects/TuinaDex/data/stereo_calibration/stereo-recalibration-v2-20260828/results/20260828-215540665234/stereo_calibration.npz
```

单窗口左侧显示彩色画面和深度伪彩色图，右侧显示可旋转、平移、缩放的彩色点云。顶部显示实际 FPS、有效像素数、点数和中位深度。

平衡模式在 50% 分辨率计算视差，再恢复到原标定尺度，以降低延迟。「暂停」只冻结界面，「重置视角」会在下一帧重新对准有效点云。退出窗口时程序会在工作线程释放两台相机。
