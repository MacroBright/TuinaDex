"""Co_Teleop.tests.test_handeye_calib — handeye_calib 纯函数单测."""
import numpy as np
import pytest

from Co_Teleop.calibration.handeye_calib import (
    apply_rotation,
    load_calib,
    procrustes_rotation,
    rot_from_euler,
    save_calib,
    solve_handeye,
)


def test_euler_zero_is_identity():
    R = rot_from_euler(0, 0, 0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


def test_euler_x_90_rotates_y_to_z():
    R = rot_from_euler(90, 0, 0)
    np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]),
                               [0, 0, 1], atol=1e-9)


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(0)
    R_true = rot_from_euler(37, -12, 88)
    pts = rng.normal(size=(8, 3))
    dst = (R_true @ pts.T).T
    R = procrustes_rotation(pts, dst)
    np.testing.assert_allclose(R, R_true, atol=1e-9)


def test_apply_rotation():
    R = rot_from_euler(90, 0, 0)
    out = apply_rotation(R, np.array([[0.0, 1.0, 0.0]]))
    np.testing.assert_allclose(out[0], [0, 0, 1], atol=1e-9)


def test_save_load_roundtrip(tmp_path):
    R = rot_from_euler(10, 20, 30)
    p = tmp_path / "calib.json"
    save_calib(p, R)
    np.testing.assert_allclose(load_calib(p), R, atol=1e-9)


def test_solve_handeye_axis_mapping():
    # 相机 +X→基座+Z, +Y→基座+X, +Z→基座+Y
    cam = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    codes = [5, 1, 3]
    R = solve_handeye(cam, codes)
    np.testing.assert_allclose(R @ np.array([1.0, 0, 0]), [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(R @ np.array([0.0, 1, 0]), [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(R @ np.array([0.0, 0, 1]), [0, 1, 0], atol=1e-9)
