建议你**不要只告诉 AI“修正 Adapter 和状态机”**，而是把这次任务限定为一次 **Safety-Critical Control Refactor**：先分析现状，再制定修改方案，再写测试，最后修改代码；尤其禁止它顺手改视觉遥操算法。

结合你当前仓库状态，我建议直接把下面这份 Prompt 发给新的 Agent。

# TuinaDex：真实 Arm Adapter 状态与启动状态机修正任务

你现在接管 `MacroBright/TuinaDex` 项目。请先阅读仓库当前 `main` 以及 `Arm-robot_VLA` 子模块当前指针对应代码，不要依据历史上下文猜测实现。

本次任务只处理两个问题：

1. **修正真实 `RealArmAdapter` 的机械臂状态获取**
2. **修正真实机械臂的启动 / ARM / TELEOP 状态机**

本次任务不是视觉遥操功能扩展，也不要修改 Hand、Vision Tracking、VLA、数据集等无关部分。

---

## 一、项目当前硬性架构约束

当前机械臂真实控制链：

```text
Python
  ↓
Co_Teleop
  ↓
RealArmAdapter
  ↓
CartesianController
  ↓
ZdtController
  ↓
ZdtDriver
  ↓
SocketCAN / USB-CAN
  ↓
6 × ZDT Motor
```

**不使用 STM32 作为当前机械臂主控。**

`firmware/`、旧 STM32 UART 协议等仅作为历史参考实现，不得重新引入运行链路。

笛卡尔运动唯一入口：

```text
Vision / Teleop / VLA
        ↓
CartesianCommand
        ↓
RealArmAdapter
        ↓
CartesianController
```

禁止绕过 `CartesianController` 直接发送 joint/CAN 命令。

---

# 二、第一部分：先完成代码审查，不要立即修改

先读取并分析至少以下文件：

```text
Co_Teleop/adapters/arm_adapter.py
Co_Teleop/pipeline/unified_teleop.py
Co_Teleop/pipeline/single_arm_teleop.py
Co_Teleop/tests/test_adapter.py
Co_Teleop/tests/test_real_arm_teleop.py

Arm-robot_VLA/lerobot_robot_massage/zdt/controller.py
Arm-robot_VLA/lerobot_robot_massage/zdt/cartesian.py
Arm-robot_VLA/lerobot_robot_massage/zdt/safety.py
Arm-robot_VLA/lerobot_robot_massage/zdt/types.py
Arm-robot_VLA/lerobot_robot_massage/zdt/zdt_driver.py
Arm-robot_VLA/lerobot_robot_massage/zdt/config.py
```

重点确认：

```text
1. RobotStateMachine 当前真实状态转换逻辑
2. SafetyMachine 当前职责
3. ZdtController.connect()
4. ZdtController.arm()
5. ZdtController.enter_teleop()
6. ZdtController.disarm()
7. ZdtController.e_stop()
8. ZdtController.get_real_state() / 对应真实状态接口
9. CartesianController.get_current_pose()
10. RealArmAdapter 当前调用链
11. unified_teleop.py 当前启动顺序
12. 当前测试覆盖情况
```

分析结束后，先输出：

```text
A. 当前状态机实际行为
B. 当前 RealArmAdapter 实际行为
C. 与设计目标存在的差异
D. 修改范围
E. 测试影响
```

在完成分析前不要修改代码。

---

# 三、问题 1：修正 RealArmAdapter 的真实状态

## 当前问题

目前 `RealArmAdapter.get_joint_state()` 仍然存在使用：

```text
_tracked_angles
```

以及：

```text
dq = 0
current = 0
flags = 0
```

等占位/软件跟踪数据的情况。

这不能作为真实机械臂 Observation。

---

## 目标

`RealArmAdapter.get_joint_state()` 必须反映：

```text
真实关节位置 q
真实/可靠估计关节速度 dq
真实电机 current
真实 motor flags/status
```

优先复用底层已经存在的真实状态接口，不要在 Adapter 中重复实现 CAN 协议。

优先路径：

```text
ZdtController
    ↓
get_real_state()
    ↓
RealArmAdapter
    ↓
JointState
```

不要：

```text
RealArmAdapter
    ↓
自己调用 0x36
    ↓
自己解析 CAN
```

CAN 协议只允许存在于底层 ZdtDriver。

---

## q 的要求

机械臂关节角必须来自真实位置反馈：

```text
0x36
 ↓
calibration
 ↓
anchor joint angle
```

不能默认使用：

```text
_tracked_angles
```

作为真实 observation。

允许 tracked state 作为：

```text
fallback
```

但必须：

1. 明确标记 fallback；
2. 不能静默伪装成真实反馈；
3. 不应成为正常路径。

---

## dq 的要求

优先使用底层真实速度反馈。

如果当前 ZDT 协议无法稳定获得真实速度，可以使用：

```text
dq = finite_difference(q_real, timestamp)
```

并采用合理滤波。

但必须满足：

```text
dq ≠ 永久固定 0
```

同时注意：

* 使用 `time.monotonic()`
* 处理首帧
* 处理异常 dt
* 防止异常位置跳变导致巨大 dq

---

## current 的要求

如果底层已经有实时电流读取：

```text
controller / driver
```

直接复用。

不能在 Adapter 中写：

```python
current_ma=(0.0,) * 6
```

正常运行时必须反映真实六轴状态。

---

## flags/status 的要求

必须完整保留六个关节的信息：

```text
J1
J2
J3
J4
J5
J6
```

不要只保存：

```text
flags[0]
```

推荐：

```python
flags: tuple[int, ...]
```

长度固定为 6。

如果底层 status 不适合直接映射到 `JointState`，请在 `types.py` 中明确数据结构，不要丢信息。

---

# 四、第二部分：修正启动状态机

目标状态：

```text
DISCONNECTED
      ↓
CONNECTED
      ↓
ENUMERATED
      ↓
SAFE_IDLE
      ↓
ARMED
      ↓
TELEOP
```

异常状态：

```text
FAULT
  ↓
STOPPED
```

恢复：

```text
STOPPED / FAULT
       ↓
人工确认
       ↓
re_arm
       ↓
ARMED
```

---

# 五、最重要的安全不变量

必须保证：

```text
connect() != arm()

arm() != teleop

connect() 不得自动 Enable Torque

SAFE_IDLE 不得执行机械臂运动

未通过六轴 enumerate/verify 不得 ARM

未经过用户显式确认不得进入 ARMED

未进入 ARMED/TELEOP 不得执行 Cartesian motion
```

尤其检查当前：

```text
RealArmAdapter.connect()
unified_teleop.main()
```

是否存在：

```python
connect()
arm()
enter_teleop()
```

连续自动执行的问题。

---

# 六、正确启动流程

真实机械臂启动应该类似：

```text
程序启动
   ↓
CAN connect
   ↓
scan / enumerate
   ↓
验证 6 个电机
   ↓
读取真实关节状态
   ↓
同步 anchor / state
   ↓
SAFE_IDLE
   ↓
等待用户确认
   ↓
arm()
   ↓
ARMED
   ↓
显式进入 teleop
   ↓
TELEOP
```

其中：

```text
connect()
```

不能：

```text
set_torque(True)
```

也不能：

```text
ready()
```

也不能：

```text
enter_teleop()
```

---

# 七、六轴枚举必须是硬门禁

只有：

```text
J1~J6 全部在线
CAN ID 正确
slot mapping 完整
无重复 ID
无 unexpected motor
状态读取正常
```

才能：

```text
SAFE_IDLE
```

如果出现任何问题：

```text
missing motor
duplicate ID
unexpected ID
read failure
mapping failure
```

必须：

```text
FAULT / safe failure
```

并禁止：

```text
arm()
```

不能出现：

```text
5/6 电机在线
→ 允许 ARM
```

---

# 八、ARM 操作

`arm()` 的职责是：

```text
确认重力关节
+
确认六轴状态正常
+
Enable torque
+
RobotStateMachine → ARMED
```

尤其 J2/J3 是重力关键关节。

CLI 中：

```text
-y / --gravity-confirm
```

必须真正参与逻辑。

不要出现：

```python
arm(gravity_confirmed=True)
```

把命令行参数绕过的情况。

正确：

```python
arm(
    gravity_confirmed=args.gravity_confirm
)
```

---

# 九、TELEOP 操作

只有：

```text
phase == ARMED
```

才能：

```text
enter_teleop()
```

然后：

```text
phase == TELEOP
```

CartesianController 才允许：

```text
step()
step_pose()
```

如果状态不是：

```text
ARMED / TELEOP
```

必须拒绝运动命令：

```text
moved=False
reason=not_armed(...)
```

---

# 十、READY / HOME / RESET 的安全边界

重点检查：

```text
ready()
home()
reset()
```

因为这些操作可能导致实际机械臂运动。

必须明确：

```text
reset() ≠ 纯软件 reset
```

如果 `ready()` / `home()` 会移动真实机械臂：

```text
必须要求 ARMED
```

并且：

```text
connect()
```

不能隐式调用。

不要为了简化代码将：

```text
connect → arm → ready → teleop
```

重新合并。

---

# 十一、E-stop / Re-arm

要求：

```text
e_stop()
```

必须：

```text
→ STOPPED
→ 锁存
→ 禁止继续运动
```

重新恢复必须：

```text
STOPPED
 ↓
manual re_arm()
 ↓
arm()
 ↓
ARMED
```

禁止：

```text
视觉恢复
↓
自动重新 enable
```

同时检查当前：

```text
SPACE
W
R
```

等键盘路径是否会绕过状态机。

---

# 十二、Adapter 职责边界

`RealArmAdapter` 只负责：

```text
Robot API abstraction
```

应该：

```text
connect
arm
enter_teleop
exit_teleop
get_joint_state
get_real_joint_angles
get_ee_pose
move_cartesian_velocity
ready
home
e_stop
```

不应该负责：

```text
CAN protocol
CAN frame encoding
IK
FK implementation
motor pulse conversion
Safety policy implementation
```

这些属于底层模块。

---

# 十三、测试要求

完成修改后必须新增/更新测试。

至少覆盖：

## 状态机

```text
connect → SAFE_IDLE

SAFE_IDLE → cannot move

without gravity confirmation → cannot arm

wrong motor enumeration → cannot arm

six motors valid → can arm

ARMED → can enter TELEOP

STOPPED → cannot move

STOPPED → explicit re_arm → ARMED
```

## Adapter

测试：

```text
get_joint_state().q
```

必须来自真实 state source。

测试：

```text
dq != placeholder
current == six-axis current
flags == six-axis flags
```

并验证：

```text
missing real feedback
```

不会静默伪装成正常真实状态。

## 安全行为

测试：

```text
connect()
```

不能触发：

```text
torque enable
motion
teleop
```

---

# 十四、测试方式

全部首先使用：

```text
FakeTransport
FakeController
FakeAdapter
```

不能为了测试直接连接真实 CAN。

运行：

```bash
pytest -q
```

要求：

```text
现有测试全部通过
+
新增测试全部通过
```

如果现有测试与新的安全状态机语义冲突：

> 优先修正测试以匹配新的明确设计，而不是降低安全要求使旧测试通过。

---

# 十五、代码修改范围

优先允许修改：

```text
Co_Teleop/adapters/arm_adapter.py

Co_Teleop/pipeline/unified_teleop.py

Co_Teleop/pipeline/single_arm_teleop.py

Arm-robot_VLA/lerobot_robot_massage/zdt/controller.py
Arm-robot_VLA/lerobot_robot_massage/zdt/safety.py
Arm-robot_VLA/lerobot_robot_massage/zdt/types.py

Co_Teleop/tests/
Arm-robot_VLA/tests/
```

如发现必须修改其他文件才能保持架构一致，可以修改，但必须说明原因。

禁止大范围无关重构。

---

# 十六、本次任务明确不做

本次不要：

```text
1. 修改视觉 HandTracker
2. 修改 RealSense
3. 修改灵巧手算法
4. 修改手指映射
5. 修改 VLA
6. 修改数据集格式
7. 修改 CAN 协议
8. 重新设计 IK
9. 重新设计 CartesianController
10. 重新引入 STM32
```

除非测试证明存在直接阻塞本任务的依赖，否则不要碰这些部分。

---

# 十七、完成标准

任务完成后必须能够明确回答：

### 状态

```text
当前真实机械臂启动后处于什么状态？
```

答案必须是：

```text
SAFE_IDLE
```

而不是：

```text
ARMED / TELEOP
```

### 真状态

能够得到：

```text
q[6]
dq[6]
current[6]
flags[6]
```

并明确哪些是：

```text
真实反馈
```

哪些是：

```text
估算值
```

### 运动权限

明确：

```text
SAFE_IDLE → no motion

ARMED → motion permitted

TELEOP → Cartesian teleoperation permitted

STOPPED → no motion
```

### 测试

必须：

```text
pytest -q
```

全部通过。

---

# 十八、执行纪律

采用：

```text
分析
 ↓
Plan
 ↓
Tests
 ↓
Implementation
 ↓
Regression
 ↓
Review
```

不要一次性大范围修改。

每完成一个逻辑阶段，说明：

```text
修改了什么
为什么修改
测试验证了什么
还有什么风险
```

最终输出：

```text
1. 修改文件列表
2. 状态机最终状态转换图
3. RealArmAdapter 最终数据来源
4. 安全行为变化
5. 测试结果
6. 剩余风险
7. 建议下一步
```

**尤其不要在没有明确测试和状态机验证的情况下执行真实机械臂运动。**




另外，从我刚刚重新读取的最新仓库看，有一个值得你特别注意的事实：当前 `unified_teleop.py` 的真实臂启动代码确实仍然是 `connect() → arm(gravity_confirmed=True) → enter_teleop()`，而 CLI 的 `--gravity-confirm` 并没有真正用于 `arm()` 参数；这正是这次任务最应该让 Agent 修正的地方。

同时，`RealArmAdapter.get_joint_state()` 当前仍然返回零速度、零电流和零 flags 的占位状态，这也是你现在在进入 VLA 数据采集之前必须修掉的问题。


