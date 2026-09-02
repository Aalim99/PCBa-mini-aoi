"""Automated tests for core/inspection.py using synthetic board images.
Run directly:
    python tests/test_inspection.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.inspection import (
    inspect, detect_panel_mode, expand_components, panel_unit_origins,
    expanded_fiducials_mm, project_roi, PresenceThresholds,
)
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
)

PART_SIZES = {
    "PN-1001": {"width_mm": 2.0, "height_mm": 1.2},
    "PN-2002": {"width_mm": 3.0, "height_mm": 2.0},
}


def make_components(prefix="R", origin=(0.0, 0.0), count=6):
    ox, oy = origin
    comps = []
    for i in range(count):
        comps.append({
            "designator": f"{prefix}{i + 1}",
            "x": ox + 10.0 + (i % 3) * 20.0,
            "y": oy + 10.0 + (i // 3) * 20.0,
            "rotation": 0.0 if i % 2 else 90.0,
            "library": "LIB",
            "part": "PN-1001" if i % 2 else "PN-2002",
        })
    return comps


def build_scene(components, fiducials_mm=((5.0, 5.0), (75.0, 8.0), (10.0, 55.0)),
                missing=(), part_sizes=None, scale=9.0):
    part_sizes = part_sizes or PART_SIZES
    fiducials_mm = [tuple(f) for f in fiducials_mm]
    anchor = fiducials_mm + [(c["x"], c["y"]) for c in components]
    H = place_homography(anchor, scale=scale, angle_deg=2.0)
    frame, _ = make_synthetic_board_frame(
        fiducials_mm, H, image_size=autosize_canvas(anchor, H), noise_std=3.0,
    )
    draw_components(frame, components, H, part_sizes, missing_designators=missing)
    return frame, H


def test_all_present_passes():
    comps = make_components()
    program = {"name": "T1", "is_panel": False, "components": comps,
               "fiducials": [], "panel_offsets": []}
    frame, H = build_scene(comps)
    result = inspect(frame, program, PART_SIZES, H)
    assert result.verdict == "PASS", f"expected PASS, got {result.verdict}: {result.message} " \
                                      f"missing={[c.designator for c in result.missing]}"
    assert result.checked_count == len(comps)
    print("OK test_all_present_passes:", result.message)


def test_missing_components_detected():
    comps = make_components()
    missing = {"R2", "R5"}
    program = {"name": "T2", "is_panel": False, "components": comps,
               "fiducials": [], "panel_offsets": []}
    frame, H = build_scene(comps, missing=missing)
    result = inspect(frame, program, PART_SIZES, H)
    assert result.verdict == "FAIL", f"expected FAIL, got {result.verdict}: {result.message}"
    found = {c.designator for c in result.missing}
    assert found == missing, f"expected missing {missing}, detected {found}"
    print("OK test_missing_components_detected:", result.message, "->", sorted(found))


def test_unsized_part_never_passes():
    """A part with no ROI size can't be checked -- the board must not
    come back a clean PASS as if it had been."""
    comps = make_components()
    comps.append({"designator": "U9", "x": 50.0, "y": 45.0, "rotation": 0.0,
                  "library": "IC", "part": "PN-NOSIZE"})
    program = {"name": "T3", "is_panel": False, "components": comps,
               "fiducials": [], "panel_offsets": []}
    frame, H = build_scene(comps)
    result = inspect(frame, program, PART_SIZES, H)
    assert result.verdict == "INCOMPLETE", f"expected INCOMPLETE, got {result.verdict}"
    assert [c.designator for c in result.unchecked] == ["U9"]
    print("OK test_unsized_part_never_passes:", result.message)


def test_off_frame_component_never_passes():
    comps = make_components()
    comps.append({"designator": "R99", "x": 5000.0, "y": 5000.0, "rotation": 0.0,
                  "library": "LIB", "part": "PN-1001"})
    program = {"name": "T4", "is_panel": False, "components": comps,
               "fiducials": [], "panel_offsets": []}
    frame, H = build_scene(comps[:-1])
    result = inspect(frame, program, PART_SIZES, H)
    assert result.verdict == "INCOMPLETE", f"expected INCOMPLETE, got {result.verdict}"
    assert [c.status for c in result.unchecked] == ["off_frame"]
    print("OK test_off_frame_component_never_passes:", result.message)


def test_panel_mode_detection_replicate():
    """One unit's worth of components + offsets -> replicate."""
    comps = make_components(count=4)
    program = {
        "name": "P1", "is_panel": True, "components": comps, "fiducials": [],
        "panel_offsets": [{"label": "U2", "dx": 100.0, "dy": 0.0},
                           {"label": "U3", "dx": 0.0, "dy": 80.0}],
    }
    assert detect_panel_mode(program) == "replicate", detect_panel_mode(program)
    # 2 offsets describe 3 units (base + 2), so the base origin is implicit
    origins = panel_unit_origins(program)
    assert len(origins) == 3, origins
    expanded = expand_components(program)
    assert len(expanded) == len(comps) * 3, len(expanded)
    assert {c["unit"] for c in expanded} == {"U1", "U2", "U3"}
    print("OK test_panel_mode_detection_replicate:", len(expanded), "instances across", len(origins), "units")


def test_panel_mode_detection_expanded():
    """Components already spread across every offset -> expanded."""
    offsets = [{"label": "U2", "dx": 100.0, "dy": 0.0},
               {"label": "U3", "dx": 0.0, "dy": 80.0},
               {"label": "U4", "dx": 100.0, "dy": 80.0}]
    comps = make_components(prefix="A", origin=(0.0, 0.0), count=3)
    for o in offsets:
        comps += make_components(prefix=o["label"] + "R", origin=(o["dx"], o["dy"]), count=3)
    program = {"name": "P2", "is_panel": True, "components": comps,
               "fiducials": [], "panel_offsets": offsets}
    assert detect_panel_mode(program) == "expanded", detect_panel_mode(program)
    expanded = expand_components(program)
    assert len(expanded) == len(comps), "expanded mode must not duplicate rows"
    assert len({c["unit"] for c in expanded}) == 4
    print("OK test_panel_mode_detection_expanded:", len(expanded), "instances, units =",
          sorted({c["unit"] for c in expanded}))


def test_explicit_panel_mode_overrides_detection():
    comps = make_components(count=4)
    program = {"name": "P3", "is_panel": True, "components": comps, "fiducials": [],
               "panel_offsets": [{"label": "U2", "dx": 100.0, "dy": 0.0}],
               "panel_mode": "expanded"}
    assert detect_panel_mode(program) == "expanded"
    assert len(expand_components(program)) == len(comps)
    print("OK test_explicit_panel_mode_overrides_detection")


def test_panel_inspection_reports_per_unit():
    """A 2-unit panel with one missing part in the second unit only."""
    comps = make_components(count=4)
    offsets = [{"label": "U2", "dx": 90.0, "dy": 0.0}]
    program = {"name": "P4", "is_panel": True, "components": comps,
               "fiducials": [], "panel_offsets": offsets, "panel_mode": "replicate"}

    all_instances = expand_components(program)
    fiducials_mm = [(5.0, 5.0), (75.0, 8.0), (10.0, 55.0)]
    anchor = fiducials_mm + [(c["x"], c["y"]) for c in all_instances]
    H = place_homography(anchor, scale=8.0, angle_deg=1.0)
    frame, _ = make_synthetic_board_frame(fiducials_mm, H,
                                           image_size=autosize_canvas(anchor, H), noise_std=3.0)
    # draw every instance, leaving U2's R3 bare
    for inst in all_instances:
        bare = ["R3"] if inst["unit"] == "U2" and inst["designator"] == "R3" else []
        draw_components(frame, [inst], H, PART_SIZES, missing_designators=bare)

    result = inspect(frame, program, PART_SIZES, H)
    assert result.verdict == "FAIL", f"{result.verdict}: {result.message}"
    by_unit = {u.label: u for u in result.units}
    assert by_unit["U1"].passed, "U1 should pass"
    assert not by_unit["U2"].passed, "U2 should fail"
    assert [c.designator for c in by_unit["U2"].missing] == ["R3"]
    print("OK test_panel_inspection_reports_per_unit: U1 pass, U2 fail ->",
          [c.designator for c in by_unit["U2"].missing])


def test_expanded_fiducials_follow_panel_mode():
    fids = [{"x": 2.0, "y": 2.0}, {"x": 2.0, "y": 18.0}]
    program = {"name": "P5", "is_panel": True, "components": make_components(count=2),
               "fiducials": fids, "panel_offsets": [{"label": "U2", "dx": 50.0, "dy": 0.0}],
               "panel_mode": "replicate"}
    assert len(expanded_fiducials_mm(program)) == 4, "replicate must repeat fiducials per unit"
    program["panel_mode"] = "expanded"
    assert len(expanded_fiducials_mm(program)) == 2, "expanded must leave fiducials alone"
    print("OK test_expanded_fiducials_follow_panel_mode")


def test_rotation_changes_roi_footprint():
    """A 90-degree part must project a swapped footprint, not the same box."""
    H = place_homography([(0.0, 0.0), (10.0, 10.0)], scale=10.0, angle_deg=0.0)
    flat = project_roi(H, 5.0, 5.0, 4.0, 1.0, rotation_deg=0.0)
    turned = project_roi(H, 5.0, 5.0, 4.0, 1.0, rotation_deg=90.0)
    assert abs(flat[2] - turned[3]) < 0.5 and abs(flat[3] - turned[2]) < 0.5, (flat, turned)
    print("OK test_rotation_changes_roi_footprint:", [round(v, 1) for v in flat],
          "->", [round(v, 1) for v in turned])


if __name__ == "__main__":
    test_all_present_passes()
    test_missing_components_detected()
    test_unsized_part_never_passes()
    test_off_frame_component_never_passes()
    test_panel_mode_detection_replicate()
    test_panel_mode_detection_expanded()
    test_explicit_panel_mode_overrides_detection()
    test_panel_inspection_reports_per_unit()
    test_expanded_fiducials_follow_panel_mode()
    test_rotation_changes_roi_footprint()
    print("\nALL INSPECTION TESTS PASSED")
