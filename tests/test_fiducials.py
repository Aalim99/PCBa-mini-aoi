"""Automated tests for core/fiducials.py (AOI-style taught fiducials).

The claim that matters: naming F1/F2/F3 and teaching their appearance
makes alignment unambiguous even on a repetitive panel, where the blob
detector could only report "ambiguous" and hand over to manual.

Run directly:
    python tests/test_fiducials.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.calibration import auto_calibrate
from core.fiducials import (
    FiducialRef, align_with_templates, delete_templates, get_fiducial_refs,
    load_templates, local_px_per_mm, match_template_peaks, save_templates,
    set_fiducial_refs, suggest_fiducial_refs, teach_templates,
)
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas,
)


def build_scene(fiducials_mm, scale=20.0, angle_deg=3.0, noise=3.0, distractors=0, seed=0):
    H = place_homography(fiducials_mm, scale=scale, angle_deg=angle_deg)
    frame, truth = make_synthetic_board_frame(
        fiducials_mm, H, image_size=autosize_canvas(fiducials_mm, H),
        noise_std=noise, distractor_circles=distractors, rng=np.random.default_rng(seed),
    )
    return frame, H, truth


def reprojection_error(H_true, H_got, points_mm):
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(pts, H_true).reshape(-1, 2)
    b = cv2.perspectiveTransform(pts, H_got).reshape(-1, 2)
    return float(np.max(np.linalg.norm(a - b, axis=1)))


def test_teach_and_align_same_board():
    fids = [(5.0, 5.0), (75.0, 9.0), (10.0, 55.0)]
    refs = [FiducialRef(f"F{i + 1}", x, y) for i, (x, y) in enumerate(fids)]
    frame, H_true, _ = build_scene(fids)

    templates = teach_templates(frame, H_true, refs)
    assert set(templates) == {"F1", "F2", "F3"}, templates.keys()

    result = align_with_templates(frame, refs, templates)
    assert result.success, result.message
    err = reprojection_error(H_true, result.homography, fids)
    assert err < 3.0, f"reprojection error {err:.2f}px"
    print(f"OK test_teach_and_align_same_board: {result.message} (err {err:.2f}px)")


def test_align_after_board_moves():
    """Teach on one placement, align on a shifted/rotated one -- the
    board is hand-placed, so it never sits exactly where it was taught."""
    fids = [(5.0, 5.0), (75.0, 9.0), (10.0, 55.0)]
    refs = [FiducialRef(f"F{i + 1}", x, y) for i, (x, y) in enumerate(fids)]
    taught_frame, H_taught, _ = build_scene(fids, angle_deg=2.0)
    templates = teach_templates(taught_frame, H_taught, refs)

    moved_frame, H_moved, _ = build_scene(fids, angle_deg=11.0, noise=4.0, seed=5)
    result = align_with_templates(moved_frame, refs, templates)
    assert result.success, result.message
    err = reprojection_error(H_moved, result.homography, fids)
    assert err < 4.0, f"reprojection error {err:.2f}px after the board moved"
    print(f"OK test_align_after_board_moves: rotated 2deg->11deg, err {err:.2f}px")


def test_panel_repetition_is_unambiguous():
    """The case blob detection could only fail safe on: a 3x3 panel whose
    fiducial pattern repeats at a regular pitch. Named marks with known
    mm spacing must align correctly, not merely refuse."""
    unit = [(2.0, 2.0), (2.0, 18.0)]
    panel = []
    for ux in range(3):
        for uy in range(3):
            panel += [(x + ux * 40.0, y + uy * 30.0) for x, y in unit]

    frame, H_true, _ = build_scene(panel, scale=12.0, angle_deg=0.0)

    # The blob path has to solve correspondence against 18 identical
    # marks; whether it manages depends on the panel's geometry, and it
    # is built to refuse rather than guess. Either outcome is fine here
    # -- recorded for contrast, not asserted.
    blob = auto_calibrate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), panel)
    blob_note = "aligned" if blob.success else f"refused ({blob.message[:34]}...)"

    # Taught fiducials have no correspondence problem: three specific
    # marks spanning the panel, verified against their mm geometry.
    refs = [FiducialRef("F1", 2.0, 2.0), FiducialRef("F2", 82.0, 2.0), FiducialRef("F3", 2.0, 78.0)]
    templates = teach_templates(frame, H_true, refs)
    result = align_with_templates(frame, refs, templates)
    assert result.success, f"taught fiducials should resolve the panel: {result.message}"
    err = reprojection_error(H_true, result.homography, panel)
    assert err < 4.0, f"panel alignment off by {err:.2f}px"
    print(f"OK test_panel_repetition_is_unambiguous: blob path {blob_note}; "
          f"templates aligned to {err:.2f}px across all 18 marks")


def test_geometry_check_rejects_differently_shaped_board():
    """Fiducial marks all look alike, so geometry is the only thing that
    can tell one board from another: a triangle of a different SHAPE
    must be refused. (A uniformly scaled one must not be -- that is just
    a different camera distance, covered by the next test.)"""
    fids = [(5.0, 5.0), (75.0, 9.0), (10.0, 55.0)]           # ratios 1 : 1.40 : 1.60
    refs = [FiducialRef(f"F{i + 1}", x, y) for i, (x, y) in enumerate(fids)]
    frame, H_true, _ = build_scene(fids)
    templates = teach_templates(frame, H_true, refs)

    other = [(5.0, 5.0), (75.0, 9.0), (70.0, 50.0)]          # ratios 1 : 1.70 : 1.86
    other_frame, _, _ = build_scene(other, scale=20.0, angle_deg=3.0, seed=2)
    result = align_with_templates(other_frame, refs, templates)
    assert not result.success, (
        "a board whose fiducial triangle is a different shape must be refused, "
        f"got: {result.message}"
    )
    print("OK test_geometry_check_rejects_differently_shaped_board:", result.message[:58])


def test_same_board_at_a_different_camera_distance_still_aligns():
    """The geometry check keys on shape, not size, so the same board
    imaged larger or smaller must still align."""
    fids = [(5.0, 5.0), (75.0, 9.0), (10.0, 55.0)]
    refs = [FiducialRef(f"F{i + 1}", x, y) for i, (x, y) in enumerate(fids)]
    taught, H_taught, _ = build_scene(fids, scale=20.0)
    templates = teach_templates(taught, H_taught, refs)

    for scale in (18.0, 22.0):
        frame, H_true, _ = build_scene(fids, scale=scale, angle_deg=4.0, seed=3)
        result = align_with_templates(frame, refs, templates)
        assert result.success, f"camera at {scale}px/mm: {result.message}"
        err = reprojection_error(H_true, result.homography, fids)
        assert err < 6.0, f"camera at {scale}px/mm: error {err:.2f}px"
    print("OK test_same_board_at_a_different_camera_distance_still_aligns: 18 and 22 px/mm")


def test_survives_clutter():
    fids = [(5.0, 5.0), (75.0, 9.0), (10.0, 55.0)]
    refs = [FiducialRef(f"F{i + 1}", x, y) for i, (x, y) in enumerate(fids)]
    clean, H_true, _ = build_scene(fids)
    templates = teach_templates(clean, H_true, refs)

    cluttered, H_c, _ = build_scene(fids, angle_deg=6.0, distractors=35, noise=6.0, seed=9)
    result = align_with_templates(cluttered, refs, templates)
    assert result.success, result.message
    err = reprojection_error(H_c, result.homography, fids)
    assert err < 5.0, f"error {err:.2f}px with 35 distractor circles"
    print(f"OK test_survives_clutter: aligned through 35 distractors, err {err:.2f}px")


def test_two_fiducials_supported():
    fids = [(5.0, 5.0), (75.0, 50.0)]
    refs = [FiducialRef("F1", *fids[0]), FiducialRef("F2", *fids[1])]
    frame, H_true, _ = build_scene(fids)
    templates = teach_templates(frame, H_true, refs)
    result = align_with_templates(frame, refs, templates)
    assert result.success, result.message
    print("OK test_two_fiducials_supported:", result.message[:60])


def test_missing_template_reports_clearly():
    refs = [FiducialRef("F1", 5.0, 5.0), FiducialRef("F2", 75.0, 9.0), FiducialRef("F3", 10.0, 55.0)]
    result = align_with_templates(np.zeros((200, 200, 3), np.uint8), refs, {})
    assert not result.success
    assert "need at least 2" in result.message, result.message
    print("OK test_missing_template_reports_clearly:", result.message[:60])


def test_fiducial_not_in_frame_names_it():
    fids = [(5.0, 5.0), (75.0, 9.0), (10.0, 55.0)]
    refs = [FiducialRef(f"F{i + 1}", x, y) for i, (x, y) in enumerate(fids)]
    frame, H_true, _ = build_scene(fids)
    templates = teach_templates(frame, H_true, refs)
    blank = np.full_like(frame, 90)
    result = align_with_templates(blank, refs, templates)
    assert not result.success
    assert "F1" in result.message or "do not form" in result.message, result.message
    print("OK test_fiducial_not_in_frame_names_it:", result.message[:70])


def test_suggest_refs_prefers_spread():
    program = {"fiducials": [{"x": 1.0, "y": 1.0}, {"x": 1.2, "y": 1.1}, {"x": 80.0, "y": 2.0},
                              {"x": 3.0, "y": 60.0}, {"x": 79.0, "y": 59.0}]}
    refs = suggest_fiducial_refs(program)
    assert len(refs) == 3
    assert [r.id for r in refs] == ["F1", "F2", "F3"]
    pts = np.array([[r.x_mm, r.y_mm] for r in refs])
    a, b = pts[1] - pts[0], pts[2] - pts[0]
    area = abs(a[0] * b[1] - a[1] * b[0]) / 2
    assert area > 500, f"suggested trio is too collinear (area {area:.0f})"
    print("OK test_suggest_refs_prefers_spread:", [(r.id, r.x_mm, r.y_mm) for r in refs])


def test_suggest_refs_with_few_fiducials():
    program = {"fiducials": [{"x": 1.0, "y": 1.0}, {"x": 50.0, "y": 2.0}]}
    refs = suggest_fiducial_refs(program)
    assert len(refs) == 2 and [r.id for r in refs] == ["F1", "F2"]
    assert suggest_fiducial_refs({"fiducials": []}) == []
    print("OK test_suggest_refs_with_few_fiducials")


def test_program_roundtrip_and_template_files():
    program = {"name": "P", "fiducials": []}
    assert get_fiducial_refs(program) == []
    refs = [FiducialRef("F1", 1.0, 2.0), FiducialRef("F2", 30.0, 4.0), FiducialRef("F3", 2.0, 25.0)]
    set_fiducial_refs(program, refs)
    back = get_fiducial_refs(program)
    assert [(r.id, r.x_mm, r.y_mm) for r in back] == [(r.id, r.x_mm, r.y_mm) for r in refs]

    tmp = Path(tempfile.mkdtemp())
    fids = [(5.0, 5.0), (75.0, 9.0), (10.0, 55.0)]
    trefs = [FiducialRef(f"F{i + 1}", x, y) for i, (x, y) in enumerate(fids)]
    frame, H_true, _ = build_scene(fids)
    templates = teach_templates(frame, H_true, trefs)
    save_templates(templates, "P", str(tmp))
    loaded = load_templates("P", str(tmp))
    assert set(loaded) == {"F1", "F2", "F3"}
    assert all(loaded[k].shape == templates[k].shape for k in loaded)
    delete_templates("P", str(tmp))
    assert load_templates("P", str(tmp)) == {}
    print("OK test_program_roundtrip_and_template_files")


def test_local_px_per_mm():
    H = place_homography([(0.0, 0.0), (10.0, 10.0)], scale=17.0, angle_deg=20.0)
    assert abs(local_px_per_mm(H, 5.0, 5.0) - 17.0) < 0.1
    print("OK test_local_px_per_mm: recovered 17.0 px/mm")


def test_match_peaks_finds_known_mark():
    fids = [(5.0, 5.0), (75.0, 9.0), (10.0, 55.0)]
    refs = [FiducialRef(f"F{i + 1}", x, y) for i, (x, y) in enumerate(fids)]
    frame, H_true, truth = build_scene(fids)
    templates = teach_templates(frame, H_true, refs)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    peaks = match_template_peaks(gray, templates["F2"])
    assert peaks, "no peaks for a template cut from this very frame"
    nearest = min(peaks, key=lambda p: np.hypot(p[0] - truth[1][0], p[1] - truth[1][1]))
    assert np.hypot(nearest[0] - truth[1][0], nearest[1] - truth[1][1]) < 3.0
    print(f"OK test_match_peaks_finds_known_mark: {len(peaks)} peak(s), best score {peaks[0][2]:.2f}")


if __name__ == "__main__":
    test_teach_and_align_same_board()
    test_align_after_board_moves()
    test_panel_repetition_is_unambiguous()
    test_geometry_check_rejects_differently_shaped_board()
    test_same_board_at_a_different_camera_distance_still_aligns()
    test_survives_clutter()
    test_two_fiducials_supported()
    test_missing_template_reports_clearly()
    test_fiducial_not_in_frame_names_it()
    test_suggest_refs_prefers_spread()
    test_suggest_refs_with_few_fiducials()
    test_program_roundtrip_and_template_files()
    test_local_px_per_mm()
    test_match_peaks_finds_known_mark()
    print("\nALL FIDUCIAL TESTS PASSED")
