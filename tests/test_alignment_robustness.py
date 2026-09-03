"""Tests for the alignment work that a real 37MP board photo forced:
big-frame blob detection, boards whose fiducials are degenerate for a
full homography, and sub-pixel refinement of taught-template peaks.

Run directly:
    python tests/test_alignment_robustness.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.calibration import (
    DETECT_MAX_DIMENSION, detect_fiducial_candidates, fit_homography,
    is_usable_transform, mm_to_px, mm_to_px_batch,
)
from core.camera import StillImageSource
from core.fiducials import _subpixel_offset, match_template_peaks


def draw_pads(shape, centres, radius):
    frame = np.full(shape, 90, dtype=np.uint8)
    for cx, cy in centres:
        cv2.circle(frame, (int(round(cx)), int(round(cy))), int(radius), 240, -1)
    return frame


# ---------------------------------------------------------------------
# Large frames
# ---------------------------------------------------------------------

def test_big_frame_is_searched_downscaled_but_reported_full_size():
    """A frame past the cap must still yield candidates in *native*
    pixel coordinates -- the scaling is an internal speed trick and must
    not leak into what callers get back."""
    w, h = 4000, 3000
    assert max(w, h) > DETECT_MAX_DIMENSION
    truth = [(700.0, 600.0), (3300.0, 600.0), (700.0, 2400.0)]
    frame = draw_pads((h, w), truth, radius=26)

    start = time.time()
    found = detect_fiducial_candidates(frame, min_radius_px=8, max_radius_px=80)
    elapsed = time.time() - start

    for tx, ty in truth:
        nearest = min(found, key=lambda c: np.hypot(c[0] - tx, c[1] - ty))
        assert np.hypot(nearest[0] - tx, nearest[1] - ty) < 8.0, \
            f"pad at ({tx}, {ty}) not found in native coordinates; nearest {nearest}"
    print(f"OK test_big_frame_is_searched_downscaled_but_reported_full_size: "
          f"{len(found)} candidate(s) in {elapsed:.2f}s")


def test_small_frame_is_untouched():
    truth = [(120.0, 100.0), (500.0, 100.0), (120.0, 400.0)]
    frame = draw_pads((500, 640), truth, radius=12)
    found = detect_fiducial_candidates(frame, min_radius_px=5, max_radius_px=40)
    for tx, ty in truth:
        nearest = min(found, key=lambda c: np.hypot(c[0] - tx, c[1] - ty))
        assert np.hypot(nearest[0] - tx, nearest[1] - ty) < 3.0
    print(f"OK test_small_frame_is_untouched: {len(found)} candidate(s)")


# ---------------------------------------------------------------------
# Degenerate fiducial layouts
# ---------------------------------------------------------------------

def known_transform(scale=30.0, angle_deg=2.0, tx=150.0, ty=200.0):
    a = np.deg2rad(angle_deg)
    return np.array([
        [scale * np.cos(a), -scale * np.sin(a), tx],
        [scale * np.sin(a),  scale * np.cos(a), ty],
        [0.0, 0.0, 1.0],
    ])


def check_recovers(points_mm, label, tolerance=0.5):
    H_true = known_transform()
    mm = np.asarray(points_mm, dtype=np.float64)
    px = [tuple(p) for p in mm_to_px_batch(H_true, mm)]
    H, rms = fit_homography([tuple(p) for p in mm], px)
    assert H is not None, f"{label}: no transform fitted"
    assert is_usable_transform(H), f"{label}: fitted transform is not usable"
    assert rms < tolerance, f"{label}: rms {rms:.3f}px"
    # and it must agree with the truth away from the fitted points too
    for probe in ((0.0, 0.0), (200.0, 90.0), (-30.0, 250.0)):
        got = np.array(mm_to_px(H, *probe))
        want = np.array(mm_to_px(H_true, *probe))
        assert np.linalg.norm(got - want) < 2.0, f"{label}: drifts at {probe}"
    return rms


def test_two_row_panel_fiducials_still_fit():
    """The real failure: this user's board has ten fiducials on just two
    Y values. Every 4-point sample is then two pairs of collinear points,
    which cv2.findHomography cannot solve -- the fit has to fall back to
    a model the points do support instead of reporting failure."""
    xs = [8.0, 60.0, 112.0, 164.0, 216.0]
    points = [(x, 8.69) for x in xs] + [(x, 139.14) for x in xs]
    rms = check_recovers(points, "two-row panel")
    print(f"OK test_two_row_panel_fiducials_still_fit: 10 points, 2 rows, rms {rms:.3f}px")


def test_fiducials_along_one_edge_still_fit():
    points = [(10.0, 5.0), (60.0, 5.0), (110.0, 5.0), (160.0, 5.0)]
    rms = check_recovers(points, "single row")
    print(f"OK test_fiducials_along_one_edge_still_fit: rms {rms:.3f}px")


def test_three_and_two_point_layouts_still_fit():
    print(f"OK test_three_and_two_point_layouts_still_fit: "
          f"3pt rms {check_recovers([(5.0, 5.0), (100.0, 8.0), (12.0, 90.0)], '3 points'):.3f}px, "
          f"2pt rms {check_recovers([(5.0, 5.0), (100.0, 90.0)], '2 points'):.3f}px")


def test_well_spread_fiducials_still_use_the_full_homography():
    points = [(5.0, 5.0), (150.0, 6.0), (152.0, 100.0), (4.0, 98.0)]
    rms = check_recovers(points, "well spread")
    print(f"OK test_well_spread_fiducials_still_use_the_full_homography: rms {rms:.3f}px")


def test_a_single_point_is_refused():
    H, rms = fit_homography([(0.0, 0.0)], [(10.0, 10.0)])
    assert H is None and rms == float("inf"), "one point cannot define a transform"
    print("OK test_a_single_point_is_refused")


def test_coincident_points_are_refused():
    """All correspondences on one spot collapses the linear part; the
    caller must be told, not handed a matrix that maps everything to a
    point."""
    H, _rms = fit_homography([(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
                             [(10.0, 10.0), (10.0, 10.0), (10.0, 10.0)])
    assert H is None or not is_usable_transform(H)
    print("OK test_coincident_points_are_refused")


def test_is_usable_transform_rejects_junk():
    good = known_transform()
    assert is_usable_transform(good)
    assert not is_usable_transform(None)
    assert not is_usable_transform(np.zeros((3, 3)))
    assert not is_usable_transform(np.full((3, 3), np.nan))
    assert not is_usable_transform(np.eye(2))
    collapsed = good.copy()
    collapsed[:2, :2] = [[1.0, 2.0], [2.0, 4.0]]   # singular linear part
    assert not is_usable_transform(collapsed)
    print("OK test_is_usable_transform_rejects_junk")


# ---------------------------------------------------------------------
# Sub-pixel fiducial peaks
# ---------------------------------------------------------------------

def test_subpixel_offset_finds_a_parabola_vertex():
    response = np.zeros((5, 5), dtype=np.float32)
    # a peak whose true vertex sits a quarter-pixel right of centre
    for y in range(5):
        for x in range(5):
            response[y, x] = -((x - 2.25) ** 2) - ((y - 2.0) ** 2)
    dx, dy = _subpixel_offset(response, 2, 2)
    assert abs(dx - 0.25) < 0.02, dx
    assert abs(dy) < 0.02, dy
    print(f"OK test_subpixel_offset_finds_a_parabola_vertex: dx={dx:.3f} dy={dy:.3f}")


def test_subpixel_offset_is_clamped_and_edge_safe():
    flat = np.ones((5, 5), dtype=np.float32)
    assert _subpixel_offset(flat, 2, 2) == (0.0, 0.0), "a flat surface offers no refinement"
    assert _subpixel_offset(flat, 0, 0) == (0.0, 0.0), "border cells have no neighbours"
    assert _subpixel_offset(flat, 4, 4) == (0.0, 0.0)

    spiky = np.array([[0, 0, 0], [0.0, 1.0, 0.9], [0, 0, 0]], dtype=np.float32)
    padded = np.zeros((5, 5), dtype=np.float32)
    padded[1:4, 1:4] = spiky
    dx, dy = _subpixel_offset(padded, 2, 2)
    assert -0.5 <= dx <= 0.5 and -0.5 <= dy <= 0.5, (dx, dy)
    print("OK test_subpixel_offset_is_clamped_and_edge_safe")


def test_subpixel_beats_integer_peaks_on_fractional_shifts():
    """The measurement that justifies the refinement: shift a fiducial by
    fractions of a pixel and see whether the reported peak follows it.
    Integer peaks cannot, by construction."""
    template = draw_pads((40, 40), [(20, 20)], radius=11)
    rng = np.random.default_rng(7)

    refined_errors, integer_errors = [], []
    for shift_x, shift_y in [(0.25, 0.0), (0.5, 0.25), (-0.35, 0.4), (0.1, -0.45), (0.6, 0.6)]:
        scene = np.full((160, 160), 90, dtype=np.uint8)
        scene[60:100, 60:100] = template
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        scene = cv2.warpAffine(scene, M, (160, 160), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        scene = np.clip(scene.astype(np.int16) + rng.normal(0, 1.5, scene.shape), 0, 255).astype(np.uint8)

        truth = (80.0 + shift_x, 80.0 + shift_y)
        peaks = match_template_peaks(scene, template, top_k=1, scales=(1.0,), min_score=0.4)
        assert peaks, f"no peak found for shift {(shift_x, shift_y)}"
        refined_errors.append(np.hypot(peaks[0][0] - truth[0], peaks[0][1] - truth[1]))

        response = cv2.matchTemplate(scene, template, cv2.TM_CCOEFF_NORMED)
        _mn, _mx, _ml, max_l = cv2.minMaxLoc(response)
        integer = (max_l[0] + 20.0, max_l[1] + 20.0)
        integer_errors.append(np.hypot(integer[0] - truth[0], integer[1] - truth[1]))

    refined = float(np.mean(refined_errors))
    integer = float(np.mean(integer_errors))
    assert refined < integer, f"refinement made it worse: {refined:.3f} vs {integer:.3f}"
    print(f"OK test_subpixel_beats_integer_peaks_on_fractional_shifts: "
          f"mean error {integer:.3f}px -> {refined:.3f}px "
          f"({100 * (1 - refined / integer):.0f}% better)")


# ---------------------------------------------------------------------
# Still images are not a video stream
# ---------------------------------------------------------------------

def test_still_image_source_declares_itself_static(tmp_root=None):
    """The front-page freeze came from re-converting a still 37MP photo
    thirty times a second. A still source has to say so, so the live tab
    can read it once instead of polling it."""
    root = Path(tmp_root or __import__("tempfile").mkdtemp())
    path = root / "board.png"
    cv2.imwrite(str(path), draw_pads((200, 300), [(100, 100)], radius=20))

    source = StillImageSource(path=str(path))
    assert source.is_static is True, "a photo is not a stream"
    assert source.open()
    frame = source.read()
    assert frame is not None and frame.shape[:2] == (200, 300)
    assert source.read() is not None, "repeat reads must keep working"
    source.release()
    print("OK test_still_image_source_declares_itself_static")


if __name__ == "__main__":
    test_big_frame_is_searched_downscaled_but_reported_full_size()
    test_small_frame_is_untouched()
    test_two_row_panel_fiducials_still_fit()
    test_fiducials_along_one_edge_still_fit()
    test_three_and_two_point_layouts_still_fit()
    test_well_spread_fiducials_still_use_the_full_homography()
    test_a_single_point_is_refused()
    test_coincident_points_are_refused()
    test_is_usable_transform_rejects_junk()
    test_subpixel_offset_finds_a_parabola_vertex()
    test_subpixel_offset_is_clamped_and_edge_safe()
    test_subpixel_beats_integer_peaks_on_fractional_shifts()
    test_still_image_source_declares_itself_static()
    print("\nALL ALIGNMENT ROBUSTNESS TESTS PASSED")
