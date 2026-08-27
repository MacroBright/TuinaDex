"""arm_client 单测：用 pyserial loop:// 回环捕获写入的 remote_event 命令。

注: pyserial 的 loop:// 是"自回环"（每次 serial_for_url 各自持有独立队列）。
因此测试与 ArmClient 共享同一个 loop:// 连接（经 ser= 注入），
才能在同一回环上捕获/喂入数据。
"""
import threading
import time

import serial  # noqa: F401  (确保 pyserial 可用)


def test_remote_event_format():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)
    c.remote_event(vx=0.7, vy=-0.3, vz=0.5, j5=0.4, j6=-0.9, j4=0.6)
    time.sleep(0.05)
    line = s.readline().decode().strip()
    # 期望 p0=-0.700 p1=-0.300 p2=-0.900 p3=-0.400 p4=0.500 p5=-0.500 p6=0.600
    parts = line.split()
    assert parts[0] == "remote_event"
    vals = [float(v) for v in parts[1:8]]
    assert len(vals) == 7
    assert abs(vals[0] - (-0.7)) < 1e-3   # p0=-vx
    assert abs(vals[1] - (-0.3)) < 1e-3   # p1=vy
    assert abs(vals[2] - (-0.9)) < 1e-3   # p2=j6
    assert abs(vals[3] - (-0.4)) < 1e-3   # p3=-j5
    assert abs(vals[4] - 0.5) < 1e-3      # p4=vz
    assert abs(vals[5] - (-0.5)) < 1e-3   # p5=-vz
    assert abs(vals[6] - 0.6) < 1e-3      # p6=j4
    c.close()
    s.close()


def test_remote_event_j4_defaults_zero():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)
    c.remote_event(vx=0.0, vy=0.0, vz=0.0, j5=0.0)   # j6/j4 走默认 0
    time.sleep(0.05)
    line = s.readline().decode().strip()
    vals = [float(v) for v in line.split()[1:8]]
    assert len(vals) == 7
    assert vals[6] == 0.0   # p6=j4=0.0
    c.close()
    s.close()


def test_soft_reset():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)
    c.soft_reset()
    time.sleep(0.05)
    line = s.readline().decode().strip()
    assert line == "soft_reset"
    c.close()
    s.close()


def test_get_state_parse():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)

    def feed():
        time.sleep(0.05)
        s.write(b"STATE:90.00,45.00,67.00,-157.00,0.00,5.00,"
                b"0,0,0,0,0,0,0,0,0,0,0,0\n")

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    angles, _, _ = c.get_state()
    assert len(angles) == 6
    assert abs(angles[4] - 0.0) < 1e-6
    assert abs(angles[0] - 90.0) < 1e-6
    c.close()
    s.close()


def test_end_event_format():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)
    c.end_event(0.7, -0.3, 0.5, 0.4, -0.9, 0.6)
    time.sleep(0.05)
    line = s.readline().decode().strip()
    parts = line.split()
    assert parts[0] == "end_event"
    vals = [float(v) for v in parts[1:7]]
    assert len(vals) == 6
    assert abs(vals[0] - 0.7) < 1e-3
    assert abs(vals[1] - (-0.3)) < 1e-3
    assert abs(vals[2] - 0.5) < 1e-3
    assert abs(vals[3] - 0.4) < 1e-3
    assert abs(vals[4] - (-0.9)) < 1e-3
    assert abs(vals[5] - 0.6) < 1e-3
    c.close()
    s.close()


def test_get_ee_pose_parse():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)

    def feed():
        time.sleep(0.05)
        s.write(b"EEPOSE:0.50,0.10,0.30,1.00,0.00,0.00,0.00\n")

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    pos, quat = c.get_ee_pose()
    assert pos == [0.5, 0.1, 0.3]
    assert quat == [1.0, 0.0, 0.0, 0.0]
    c.close()
    s.close()


def test_get_ee_parse():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)

    def feed():
        time.sleep(0.05)
        s.write(b"EE:0.50,0.10,0.30,0.00,0.00,0.00\n")

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    ee = c.get_ee()
    assert ee == [0.5, 0.1, 0.3]
    c.close()
    s.close()


def test_get_wrist_parse():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)

    def feed():
        time.sleep(0.05)
        s.write(b"WRIST:0.30,0.05,0.40\n")

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    wrist = c.get_wrist()
    assert wrist == [0.3, 0.05, 0.4]
    c.close()
    s.close()


def test_set_joints_and_rel_rotate():
    from Co_Teleop.adapters.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://", ser=s)
    c.set_joints([90.0, 45.0, 90.0, 90.0, 0.0, 0.0])
    time.sleep(0.05)
    line = s.readline().decode().strip()
    assert line == "set_joints 90.000 45.000 90.000 90.000 0.000 0.000"
    c.rel_rotate(3, 5.0)
    time.sleep(0.05)
    line = s.readline().decode().strip()
    assert line == "rel_rotate 3 5.0"
    c.close()
    s.close()
