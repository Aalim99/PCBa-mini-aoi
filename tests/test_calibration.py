"""Automated tests for core/calibration.py using synthetic fiducial
images (no camera/physical board required). Run directly:
    python tests/test_calibration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2

from core.calibration import auto_calibrate, manual_calibrate, mm_to_px, px_to_mm
from core.testutils import (
    make_synthetic_board_frame, make_ground_truth_homography, autosize_canvas, place_homography,
)


def check_roundtrip(H_true, fiducials_mm, recovered_H):
    pts = np.asarray(fiducials_mm, dtype=np.float64).reshape(-1, 1, 2)
    truth_px = cv2.perspectiveTransform(pts, H_true).reshape(-1, 2)
    recovered_px = cv2.perspectiveTransform(pts, recovered_H).reshape(-1, 2)
    return np.linalg.norm(truth_px - recovered_px, axis=1)


def test_simple_board():
    """3 asymmetric fiducials, moderate clutter -- the common single-board case."""
    fiducials_mm = [(5.0, 5.0), (95.0, 8.0), (10.0, 70.0)]
    H_true = place_homography(fiducials_mm)
    frame, _ = make_synthetic_board_frame(fiducials_mm, H_true, image_size=autosize_canvas(fiducials_mm, H_true),
                                           distractor_circles=15, noise_std=5.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = auto_calibrate(gray, fiducials_mm)
    assert result.success, f"expected success, got: {result.message}"
    err = check_roundtrip(H_true, fiducials_mm, result.homography)
    assert err.max() < 3.0, f"reprojection error too high: {err}"
    print("OK test_simple_board:", result.message, "max err(px)=", round(float(err.max()), 3))


def test_four_fiducials_with_clutter():
    # Deliberately asymmetric quad (not a rectangle): a real board layout
    # avoids 180-degree rotational symmetry for exactly the reason this
    # test exists -- a symmetric layout is genuinely ambiguous to match
    # from points alone, both for this algorithm and for a real AOI rig.
    fiducials_mm = [(0.0, 0.0), (120.0, 3.0), (8.0, 90.0), (115.0, 70.0)]
    H_true = place_homography(fiducials_mm, scale=6.0, angle_deg=-7.0)
    frame, _ = make_synthetic_board_frame(fiducials_mm, H_true, image_size=autosize_canvas(fiducials_mm, H_true),
                                           distractor_circles=40, noise_std=6.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = auto_calibrate(gray, fiducials_mm)
    assert result.success, f"expected success, got: {result.message}"
    err = check_roundtrip(H_true, fiducials_mm, result.homography)
    assert err.max() < 3.0, f"reprojection error too high: {err}"
    print("OK test_four_fiducials_with_clutter:", result.message, "max err(px)=", round(float(err.max()), 3))


def test_no_fiducials_present():
    """Blank board, no pads -- must fail cleanly, not crash."""
    fiducials_mm = [(5.0, 5.0), (95.0, 8.0), (10.0, 70.0)]
    frame = np.full((400, 500, 3), 80, dtype=np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = auto_calibrate(gray, fiducials_mm)
    assert not result.success
    print("OK test_no_fiducials_present:", result.message)


def test_panel_ambiguity_detected():
    """2 fiducials per unit repeated identically at a regular pitch across
    a 3x3 panel -- the known hard case. What must never happen is a
    silent, confidently WRONG fit; either a clean failure or a correctly
    ambiguity-flagged/actually-correct result is acceptable."""
    unit_fiducials = [(2.0, 2.0), (2.0, 18.0)]
    fiducials_mm = []
    for ux in range(3):
        for uy in range(3):
            for fx, fy in unit_fiducials:
                fiducials_mm.append((fx + ux * 40.0, fy + uy * 30.0))
    H_true = place_homography(fiducials_mm, scale=5.0, angle_deg=0.0)
    frame, _ = make_synthetic_board_frame(fiducials_mm, H_true, image_size=autosize_canvas(fiducials_mm, H_true),
                                           distractor_circles=10, noise_std=4.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = auto_calibrate(gray, fiducials_mm)
    if result.success:
        err = check_roundtrip(H_true, fiducials_mm, result.homography)
        assert err.max() < 3.0, (
            f"panel pattern produced a confident but WRONG fit (max err {err.max():.1f}px) "
            f"without flagging ambiguity -- this is the dangerous failure mode"
        )
        print("OK test_panel_ambiguity_detected: succeeded AND correct, max err(px)=", round(float(err.max()), 3))
    else:
        print("OK test_panel_ambiguity_detected: failed safe ->", result.message, "ambiguous=", result.ambiguous)


def test_manual_calibration():
    fiducials_mm = [(5.0, 5.0), (95.0, 8.0), (10.0, 70.0), (90.0, 65.0)]
    H_true = make_ground_truth_homography(scale=7.0, angle_deg=2.0, tx=20.0, ty=15.0)
    pts = np.asarray(fiducials_mm, dtype=np.float64).reshape(-1, 1, 2)
    clicked_px = cv2.perspectiveTransform(pts, H_true).reshape(-1, 2)
    rng = np.random.default_rng(1)
    clicked_px = clicked_px + rng.normal(0, 1.5, clicked_px.shape)  # simulate imprecise clicks
    result = manual_calibrate(fiducials_mm, [tuple(p) for p in clicked_px])
    assert result.success, result.message
    err = check_roundtrip(H_true, fiducials_mm, result.homography)
    assert err.max() < 5.0, f"manual fit error too high: {err}"
    print("OK test_manual_calibration:", result.message, "max err(px)=", round(float(err.max()), 3))


def test_manual_calibration_two_points():
    """Only 2 fiducials on the board -- must degrade to similarity, not crash."""
    fiducials_mm = [(5.0, 5.0), (95.0, 65.0)]
    H_true = make_ground_truth_homography(scale=6.0, angle_deg=10.0, tx=10.0, ty=10.0)
    pts = np.asarray(fiducials_mm, dtype=np.float64).reshape(-1, 1, 2)
    clicked_px = cv2.perspectiveTransform(pts, H_true).reshape(-1, 2)
    result = manual_calibrate(fiducials_mm, [tuple(p) for p in clicked_px])
    assert result.success, result.message
    err = check_roundtrip(H_true, fiducials_mm, result.homography)
    assert err.max() < 1.0, f"2-point fit error too high: {err}"
    print("OK test_manual_calibration_two_points:", result.message, "max err(px)=", round(float(err.max()), 3))


def test_mm_px_roundtrip():
    fiducials_mm = [(0.0, 0.0), (100.0, 0.0), (0.0, 80.0), (100.0, 80.0)]
    H_true = make_ground_truth_homography()
    clicked_px = cv2.perspectiveTransform(
        np.asarray(fiducials_mm).reshape(-1, 1, 2), H_true
    ).reshape(-1, 2)
    result = manual_calibrate(fiducials_mm, [tuple(p) for p in clicked_px])
    assert result.success
    x_mm, y_mm = 42.0, 33.0
    x_px, y_px = mm_to_px(result.homography, x_mm, y_mm)
    x_mm2, y_mm2 = px_to_mm(result.homography, x_px, y_px)
    assert abs(x_mm - x_mm2) < 1e-3 and abs(y_mm - y_mm2) < 1e-3
    print("OK test_mm_px_roundtrip:", (x_mm, y_mm), "->", (round(x_px, 2), round(y_px, 2)),
          "->", (round(x_mm2, 3), round(y_mm2, 3)))


if __name__ == "__main__":
    test_simple_board()
    test_four_fiducials_with_clutter()
    test_no_fiducials_present()
    test_panel_ambiguity_detected()
    test_manual_calibration()
    test_manual_calibration_two_points()
    test_mm_px_roundtrip()
    print("\nALL CALIBRATION TESTS PASSED")
