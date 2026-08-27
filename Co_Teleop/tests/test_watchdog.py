"""Co_Teleop.tests.test_watchdog — VisionWatchdog 分级策略测试."""
import pytest
from Co_Teleop.safety import VisionWatchdog, WatchdogAction


def _wd(**kw):
    d = dict(conf_threshold=0.5, loss_stop_s=0.4, estop_s=1.0, decay_rate=0.5,
             wrist_jump_mm=150.0, depth_invalid_hold_s=0.2)
    d.update(kw)
    return VisionWatchdog(**d)


def _upd(w, *, hand_present=True, conf=0.9, depth_valid=True, wrist=(0.0, 0.0, 100.0),
         now=0.1):
    return w.update(hand_present=hand_present, hand_confidence=conf,
                    depth_valid=depth_valid, wrist_mm=wrist, now=now)


def test_ok_when_hand_confident():
    w = _wd()
    action, scale = _upd(w)
    assert action == WatchdogAction.OK and scale == 1.0


def test_low_confidence_escalates_to_decay():
    w = _wd()
    _upd(w, conf=0.3, now=0.1)               # 首帧建立 loss_start
    action, scale = _upd(w, conf=0.3, now=0.2)   # loss_s=0.1 → scale=0.95
    assert action == WatchdogAction.DECAY
    assert 0.0 < scale < 1.0


def test_depth_invalid_escalates():
    w = _wd()
    action, _ = _upd(w, depth_valid=False)
    assert action == WatchdogAction.DECAY


def test_depth_invalid_beyond_hold_stops():
    w = _wd()
    a0, _ = _upd(w, depth_valid=False, now=0.1)
    assert a0 == WatchdogAction.DECAY
    a1, s1 = _upd(w, depth_valid=False, now=0.5)
    assert a1 == WatchdogAction.STOP and s1 == 0.0


def test_hand_lost_prolonged_stops():
    w = _wd()
    _, _ = _upd(w, hand_present=False, conf=0.0, wrist=None, now=0.1)
    action, scale = _upd(w, hand_present=False, conf=0.0, wrist=None, now=0.5)
    assert action == WatchdogAction.STOP and scale == 0.0


def test_hand_lost_long_estops():
    w = _wd()
    for now in (0.1, 0.5, 1.2):
        _, _ = _upd(w, hand_present=False, conf=0.0, wrist=None, now=now)
    action, _ = _upd(w, hand_present=False, conf=0.0, wrist=None, now=1.3)
    assert action == WatchdogAction.ESTOP


def test_decay_is_gradual_not_hold():
    w = _wd()
    _upd(w, hand_present=False, conf=0.0, wrist=None, now=0.05)
    s0 = _upd(w, hand_present=False, conf=0.0, wrist=None, now=0.15)[1]
    s1 = _upd(w, hand_present=False, conf=0.0, wrist=None, now=0.25)[1]
    assert s1 < s0 < 1.0


def test_wrist_jump_stops():
    w = _wd()
    _upd(w, now=0.1)
    action2, _ = _upd(w, wrist=(300.0, 0.0, 100.0), now=0.2)
    assert action2 == WatchdogAction.STOP


def test_recovery_after_loss_returns_ok():
    w = _wd()
    _upd(w, hand_present=False, conf=0.0, wrist=None, now=0.1)
    action, scale = _upd(w, now=0.3)
    assert action == WatchdogAction.OK and scale == 1.0
