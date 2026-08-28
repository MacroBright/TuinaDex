"""Unit tests for TuinaSafetyLayer policy safety envelope interceptor."""

import numpy as np
import pytest

from Co_Teleop.safety.control_arbiter import ControlArbiter, ControlSource
from Co_Teleop.safety.motion_supervisor import MotionSafetySupervisor, SupervisorLimits
from packages.safety.tuina_safety_layer import TuinaSafetyLayer


def test_safety_layer_normal_processing():
    arbiter = ControlArbiter()
    supervisor = MotionSafetySupervisor()
    safety_layer = TuinaSafetyLayer(arbiter=arbiter, supervisor=supervisor, alpha_smooth=0.5)

    curr_state = np.zeros(22, dtype=np.float32)
    safety_layer.reset(curr_state)

    # 1. Normal policy action chunk (10, 22)
    chunk = np.full((10, 22), 0.05, dtype=np.float32)
    safe_target, granted, reason = safety_layer.process_policy_chunk(
        policy_action_chunk=chunk,
        current_state_22d=curr_state,
        dt=0.033,
        now=100.0,
    )

    assert granted is True
    assert safe_target is not None
    assert safe_target.shape == (22,)
    assert np.allclose(safe_target, 0.025)  # 0.5 * 0.05 + 0.5 * 0.0


def test_safety_layer_preemption_by_human():
    arbiter = ControlArbiter()
    supervisor = MotionSafetySupervisor()
    safety_layer = TuinaSafetyLayer(arbiter=arbiter, supervisor=supervisor)

    curr_state = np.zeros(22, dtype=np.float32)
    safety_layer.reset(curr_state)

    # Human teleop acquires lease
    arbiter.request_lease(ControlSource.HUMAN_TELEOP, now=100.0)

    # Policy tries to execute -> must be preempted
    chunk = np.full((5, 22), 0.1, dtype=np.float32)
    safe_target, granted, reason = safety_layer.process_policy_chunk(
        policy_action_chunk=chunk,
        current_state_22d=curr_state,
        dt=0.033,
        now=100.01,
    )

    assert granted is False
    assert safe_target is None
    assert "HUMAN_TELEOP" in reason


def test_safety_layer_clamping_moderate_jump():
    arbiter = ControlArbiter()
    supervisor = MotionSafetySupervisor(limits=SupervisorLimits(max_dq_rad=0.05, reject_dq_rad=1.0))
    safety_layer = TuinaSafetyLayer(arbiter=arbiter, supervisor=supervisor, alpha_smooth=1.0)

    curr_state = np.zeros(22, dtype=np.float32)
    safety_layer.reset(curr_state)

    # Moderate jump (0.4 rad < reject_dq 1.0)
    chunk = np.zeros((1, 22), dtype=np.float32)
    chunk[0, 0] = 0.4

    safe_target, granted, reason = safety_layer.process_policy_chunk(
        policy_action_chunk=chunk,
        current_state_22d=curr_state,
        dt=0.033,
        now=100.0,
    )

    assert granted is True
    assert safe_target is not None
    # Must be clamped by max_dq (0.05)
    assert safe_target[0] <= 0.06


def test_safety_layer_rejecting_extreme_jump():
    arbiter = ControlArbiter()
    supervisor = MotionSafetySupervisor(limits=SupervisorLimits(max_dq_rad=0.05, reject_dq_rad=1.0))
    safety_layer = TuinaSafetyLayer(arbiter=arbiter, supervisor=supervisor, alpha_smooth=1.0)

    curr_state = np.zeros(22, dtype=np.float32)
    safety_layer.reset(curr_state)

    # Extreme jump (3.5 rad > reject_dq 1.0) -> must be rejected
    chunk = np.zeros((1, 22), dtype=np.float32)
    chunk[0, 0] = 3.5

    safe_target, granted, reason = safety_layer.process_policy_chunk(
        policy_action_chunk=chunk,
        current_state_22d=curr_state,
        dt=0.033,
        now=100.0,
    )

    assert granted is False
    assert safe_target is None
    assert "REJECT" in reason
