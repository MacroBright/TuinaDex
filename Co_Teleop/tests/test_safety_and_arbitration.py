"""Unit tests for ControlArbiter, MotionSafetySupervisor, JointTargetController and LatestStateCache."""

import time
import numpy as np
import pytest

from Co_Teleop.adapters.state_poller import (
    FreshnessLimits,
    LatestStateCache,
)
from Co_Teleop.controllers.joint_target_controller import JointTargetController
from Co_Teleop.recording.teleop_frame import (
    ArmObservation,
    HandObservation,
)
from Co_Teleop.safety.control_arbiter import (
    ControlArbiter,
    ControlSource,
)
from Co_Teleop.safety.motion_supervisor import (
    MotionSafetySupervisor,
    SafetyDecision,
    SupervisorLimits,
)


def test_control_arbiter_preemption_and_expiration():
    arbiter = ControlArbiter(default_lease_duration_s=0.05)

    # Initially IDLE
    assert arbiter.get_active_source() == ControlSource.IDLE

    # Policy requests lease
    t0 = 100.0
    assert arbiter.request_lease(ControlSource.VLA_POLICY, now=t0) is True
    assert arbiter.get_active_source(now=t0) == ControlSource.VLA_POLICY

    # Human teleop preempts Policy immediately
    assert arbiter.request_lease(ControlSource.HUMAN_TELEOP, now=t0 + 0.01) is True
    assert arbiter.get_active_source(now=t0 + 0.01) == ControlSource.HUMAN_TELEOP

    # Policy tries to request while human is active -> denied
    assert arbiter.request_lease(ControlSource.VLA_POLICY, now=t0 + 0.02) is False
    assert arbiter.get_active_source(now=t0 + 0.02) == ControlSource.HUMAN_TELEOP

    # After lease expires (e.g. +60ms)
    t_expired = t0 + 0.08
    assert arbiter.get_active_source(now=t_expired) == ControlSource.IDLE

    # Now policy can acquire lease again
    assert arbiter.request_lease(ControlSource.VLA_POLICY, now=t_expired) is True
    assert arbiter.get_active_source(now=t_expired) == ControlSource.VLA_POLICY


def test_motion_safety_supervisor_clamping():
    supervisor = MotionSafetySupervisor()

    # 1. Test arm limits
    # J2 range is [-1°, 150°] -> [-0.017, 2.618] rad. If commanded 3.0 rad, should be clamped to 2.618
    target_q = np.array([0.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    clamped_q, was_clamped, is_rejected, reason = supervisor.validate_and_clamp_arm(target_q)
    assert was_clamped is True
    assert is_rejected is False
    assert clamped_q[1] <= 2.618

    # 2. Test max_dq jump limit (use dt=0.1s so max_dq is tighter than velocity limit)
    curr_q = np.zeros(6, dtype=np.float32)
    jump_q = np.array([0.4, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 0.4 rad
    clamped_jump, was_clamped, is_rejected, reason = supervisor.validate_and_clamp_arm(jump_q, current_q=curr_q, dt=0.1)
    assert is_rejected is False

    # 3. Test 22D supervise (Moderate jump -> CLAMP)
    act_22d = np.zeros(22, dtype=np.float32)
    act_22d[1] = 0.8  # Moderate J2 jump (< reject_dq 1.047)
    res_clamp = supervisor.supervise_22d(act_22d, current_state_22d=np.zeros(22, dtype=np.float32))
    assert res_clamp.is_safe is True
    assert res_clamp.decision == SafetyDecision.CLAMP
    assert res_clamp.arm_clamped is True
    assert res_clamp.clamped_arm_q[1] <= supervisor.limits.max_dq_rad + 1e-4

    # 4. Test 22D supervise (Extreme jump -> REJECT & Safe Hold)
    act_extreme = np.zeros(22, dtype=np.float32)
    act_extreme[1] = 4.0  # Extreme J2 jump (> reject_dq 1.047)
    res_reject = supervisor.supervise_22d(act_extreme, current_state_22d=np.zeros(22, dtype=np.float32))
    assert res_reject.is_safe is False
    assert res_reject.decision == SafetyDecision.REJECT
    assert np.isclose(res_reject.clamped_arm_q[1], 0.0)


def test_joint_target_controller_lease_and_smoothing():
    arbiter = ControlArbiter(default_lease_duration_s=0.05)
    supervisor = MotionSafetySupervisor()
    controller = JointTargetController(arbiter, supervisor, alpha_arm=0.5)

    curr_state = np.zeros(22, dtype=np.float32)
    action_22d = np.zeros(22, dtype=np.float32)
    action_22d[0] = 0.4  # J1 target 0.4 rad

    # First action -> granted lease, smoothed from current_state_22d and clamped
    exec_22d, res, granted = controller.process_action(action_22d, current_state_22d=curr_state, now=100.0)
    assert granted is True
    assert exec_22d is not None
    assert np.isclose(exec_22d[0], 0.2)  # 0.5 * 0.4 + 0.5 * 0.0

    # Human preempts arbiter
    arbiter.request_lease(ControlSource.HUMAN_TELEOP, now=100.01)

    # Policy tries again -> lease denied
    exec_22d_denied, res_denied, granted_denied = controller.process_action(action_22d, current_state_22d=exec_22d, now=100.02)
    assert granted_denied is False
    assert exec_22d_denied is None


def test_latest_state_cache_staleness():
    cache = LatestStateCache(limits=FreshnessLimits(max_arm_age_ms=50.0, max_hand_age_ms=50.0))

    t0 = 100.0
    cache.update_arm(ArmObservation(q=np.zeros(6), dq=np.zeros(6), current=np.zeros(6), timestamp=t0))
    cache.update_hand(HandObservation(q=np.zeros(16), currents=np.zeros(16), timestamp=t0))

    # Fresh snapshot (now = t0 + 10ms)
    snap_fresh = cache.snapshot(now=t0 + 0.010)
    assert snap_fresh.is_fresh is True
    assert snap_fresh.is_degraded is False
    assert snap_fresh.arm_age_ms == pytest.approx(10.0, abs=0.5)

    # Stale snapshot (now = t0 + 100ms)
    snap_stale = cache.snapshot(now=t0 + 0.100)
    assert snap_stale.is_fresh is False
    assert snap_stale.is_degraded is True
    assert len(snap_stale.degraded_reasons) >= 2
