"""Automated tests for core/reference_image.py.

The load-bearing claim is that a component appears in the ROI editor at
its true millimetre size, so that is what these measure: draw parts of
known mm dimensions onto a synthetic board, extract the patch, and check
the part occupies exactly width_mm * px_per_mm scene pixels.

Run directly:
    python tests/test_reference_image.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.reference_image import (
    component_patch, component_instances, load_reference, save_reference,
    delete_reference, scene_to_board_matrix,
)
from core.calibration import auto_calibrate
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
)

PX_PER_MM = 40.0
BOARD_COLOR = np.array([60, 130, 60])


def measure_component_extent(patch):
    """Bounding box (w, h) in patch pixels of everything that isn't the
    plain board background."""
    diff = np.linalg.norm(patch.astype(np.float64) - BOARD_COLOR, axis=2)
    mask = diff > 45
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0
    return cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1


# Extents are measured to a few pixels rather than exactly: the patch
# resamples the board at 40 px/mm, so each component edge carries an
# interpolation ramp of roughly (40 / board px-per-mm) pixels. The
# geometry itself is exact -- measured extent converges on the true
# millimetre size as board resolution rises.
EXTENT_TOL_PX = 5


def build_board(rotation=0.0, part_w=3.0, part_h=2.0, scale=30.0, angle_deg=5.0):
    part_sizes = {"PN-X": {"width_mm": part_w, "height_mm": part_h}}
    comps = [{"designator": "R1", "x": 30.0, "y": 25.0, "rotation": rotation,
              "library": "L", "part": "PN-X"}]
    fiducials = [(5.0, 5.0), (55.0, 8.0), (9.0, 45.0)]
    anchor = fiducials + [(c["x"], c["y"]) for c in comps]
    H = place_homography(anchor, scale=scale, angle_deg=angle_deg)
    frame, _ = make_synthetic_board_frame(fiducials, H, image_size=autosize_canvas(anchor, H),
                                           noise_std=0.0)
    draw_components(frame, comps, H, part_sizes)
    return frame, H, comps[0], part_sizes


def test_patch_scale_matches_millimetres():
    """A 3.0 x 2.0 mm part must span 3.0 and 2.0 mm of scene space."""
    frame, H, comp, _ = build_board(rotation=0.0)
    patch = component_patch(frame, H, comp["x"], comp["y"], comp["rotation"],
                            half_extent=160.0, px_per_mm=PX_PER_MM)
    assert patch is not None and patch.shape[:2] == (320, 320), patch.shape
    w, h = measure_component_extent(patch)
    expect_w, expect_h = 3.0 * PX_PER_MM, 2.0 * PX_PER_MM
    assert abs(w - expect_w) <= EXTENT_TOL_PX, f"width {w}px, expected ~{expect_w}px"
    assert abs(h - expect_h) <= EXTENT_TOL_PX, f"height {h}px, expected ~{expect_h}px"
    print(f"OK test_patch_scale_matches_millimetres: {w}x{h}px for 3.0x2.0mm "
          f"(expected {expect_w:.0f}x{expect_h:.0f})")


def test_patch_derotates_component():
    """A part placed at 90 degrees must still appear at its native
    width x height, so one ROI size fits every instance of the part."""
    frame, H, comp, _ = build_board(rotation=90.0)
    patch = component_patch(frame, H, comp["x"], comp["y"], comp["rotation"],
                            half_extent=160.0, px_per_mm=PX_PER_MM)
    w, h = measure_component_extent(patch)
    expect_w, expect_h = 3.0 * PX_PER_MM, 2.0 * PX_PER_MM
    assert abs(w - expect_w) <= EXTENT_TOL_PX, f"rotated part width {w}px, expected ~{expect_w}px"
    assert abs(h - expect_h) <= EXTENT_TOL_PX, f"rotated part height {h}px, expected ~{expect_h}px"
    print(f"OK test_patch_derotates_component: 90-degree part still reads {w}x{h}px")


def test_patch_scale_independent_of_camera_distance():
    """Same part, different camera scale -- the patch must still show
    the same millimetre size, since that is what the homography is for."""
    sizes = []
    for scale in (20.0, 45.0):
        frame, H, comp, _ = build_board(scale=scale)
        patch = component_patch(frame, H, comp["x"], comp["y"], comp["rotation"],
                                half_extent=160.0, px_per_mm=PX_PER_MM)
        sizes.append(measure_component_extent(patch))
    assert abs(sizes[0][0] - sizes[1][0]) <= EXTENT_TOL_PX, sizes
    assert abs(sizes[0][1] - sizes[1][1]) <= EXTENT_TOL_PX, sizes
    print("OK test_patch_scale_independent_of_camera_distance:", sizes)


def test_patch_centred_on_component():
    frame, H, comp, _ = build_board()
    patch = component_patch(frame, H, comp["x"], comp["y"], comp["rotation"],
                            half_extent=160.0, px_per_mm=PX_PER_MM)
    diff = np.linalg.norm(patch.astype(np.float64) - BOARD_COLOR, axis=2)
    mask = diff > 45
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    assert abs(cx - 160) <= 3 and abs(cy - 160) <= 3, f"component centre at ({cx:.1f},{cy:.1f})"
    print(f"OK test_patch_centred_on_component: centre ({cx:.1f},{cy:.1f}) of 320x320")


def test_patch_follows_fiducial_alignment():
    """The homography actually used comes from fiducial detection, so an
    auto-aligned reference must give the same patch as the true one."""
    frame, H_true, comp, _ = build_board()
    fiducials = [(5.0, 5.0), (55.0, 8.0), (9.0, 45.0)]
    result = auto_calibrate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), fiducials)
    assert result.success, result.message
    patch = component_patch(frame, result.homography, comp["x"], comp["y"], comp["rotation"],
                            half_extent=160.0, px_per_mm=PX_PER_MM)
    w, h = measure_component_extent(patch)
    assert abs(w - 3.0 * PX_PER_MM) <= EXTENT_TOL_PX, (w, h)
    assert abs(h - 2.0 * PX_PER_MM) <= EXTENT_TOL_PX, (w, h)
    print(f"OK test_patch_follows_fiducial_alignment: auto-aligned patch reads {w}x{h}px")


def test_degenerate_inputs_are_safe():
    frame, H, comp, _ = build_board()
    assert component_patch(None, H, 1, 1) is None
    assert component_patch(frame, None, 1, 1) is None
    assert component_patch(frame, np.zeros((3, 3)), 1, 1) is None, "singular matrix must not raise"
    assert component_patch(frame, H, 1, 1, half_extent=1) is None, "absurdly small patch"
    print("OK test_degenerate_inputs_are_safe")


def test_save_load_roundtrip_copies_image():
    tmp = Path(tempfile.mkdtemp())
    frame, H, _, _ = build_board()
    source = tmp / "elsewhere" / "photo.png"
    source.parent.mkdir(parents=True)
    cv2.imwrite(str(source), frame)

    programs = tmp / "programs"
    programs.mkdir()
    save_reference("BOARD_A", str(programs), str(source), H)

    # the original photo can go missing and the program still works
    source.unlink()
    loaded = load_reference("BOARD_A", str(programs))
    assert loaded is not None, "reference lost when the original image moved"
    assert np.allclose(loaded["homography"], H)
    assert cv2.imread(loaded["image_path"]) is not None
    print("OK test_save_load_roundtrip_copies_image:", Path(loaded["image_path"]).name)

    delete_reference("BOARD_A", str(programs))
    assert load_reference("BOARD_A", str(programs)) is None
    print("OK: delete_reference removes both sidecar and image")


def test_load_missing_or_corrupt_reference():
    tmp = Path(tempfile.mkdtemp())
    assert load_reference("NOPE", str(tmp)) is None
    (tmp / f"BAD{'.reference.json'}").write_text("{not json")
    assert load_reference("BAD", str(tmp)) is None, "corrupt sidecar must not raise"
    print("OK test_load_missing_or_corrupt_reference")


def test_component_instances_covers_panel_units():
    comps = [{"designator": "R1", "x": 10.0, "y": 10.0, "rotation": 0.0, "part": "PN-X"},
             {"designator": "C1", "x": 20.0, "y": 10.0, "rotation": 0.0, "part": "PN-Y"}]
    program = {"name": "P", "is_panel": True, "components": comps, "fiducials": [],
               "panel_offsets": [{"label": "U2", "dx": 60.0, "dy": 0.0}],
               "panel_mode": "replicate"}
    instances = component_instances(program, "PN-X")
    assert len(instances) == 2, instances
    assert {i["unit"] for i in instances} == {"U1", "U2"}
    print("OK test_component_instances_covers_panel_units:",
          [(i["unit"], i["x"]) for i in instances])


def test_scene_to_board_matrix_maps_centre():
    m = scene_to_board_matrix(30.0, 25.0, 0.0, half_extent=160.0, px_per_mm=40.0)
    centre = m @ np.array([160.0, 160.0, 1.0])
    assert np.allclose(centre[:2], [30.0, 25.0]), centre
    # one mm to the right in board space is px_per_mm scene pixels right
    right = m @ np.array([200.0, 160.0, 1.0])
    assert np.allclose(right[:2], [31.0, 25.0]), right
    print("OK test_scene_to_board_matrix_maps_centre")


if __name__ == "__main__":
    test_patch_scale_matches_millimetres()
    test_patch_derotates_component()
    test_patch_scale_independent_of_camera_distance()
    test_patch_centred_on_component()
    test_patch_follows_fiducial_alignment()
    test_degenerate_inputs_are_safe()
    test_save_load_roundtrip_copies_image()
    test_load_missing_or_corrupt_reference()
    test_component_instances_covers_panel_units()
    test_scene_to_board_matrix_maps_centre()
    print("\nALL REFERENCE IMAGE TESTS PASSED")
