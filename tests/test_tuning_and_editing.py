"""Tests for false-call tuning (core/thresholds.py, reevaluate) and
program editing (core/program_edit.py).

Run directly:
    python tests/test_tuning_and_editing.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from core.inspection import (
    ComponentResult, InspectionResult, PresenceThresholds, UnitResult,
    decide_presence, inspect, reevaluate,
)
from core.thresholds import (
    clamp_sensitivity, effective_thresholds, load_part_thresholds,
    save_part_thresholds, thresholds_for_false_call,
)
from core.program_edit import (
    delete_designators, delete_part, describe_removal, part_summary, save_program_json,
)
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
)

PART_SIZES = {"PN-A": {"width_mm": 2.0, "height_mm": 1.2},
              "PN-B": {"width_mm": 3.0, "height_mm": 2.0}}


def make_program():
    comps = [{"designator": f"R{i + 1}", "x": 12.0 + (i % 4) * 16.0, "y": 12.0 + (i // 4) * 16.0,
              "rotation": 0.0, "library": "L", "part": "PN-A" if i % 2 else "PN-B"}
             for i in range(8)]
    return {"name": "T", "is_panel": False, "components": comps,
            "fiducials": [{"x": 4.0, "y": 4.0}, {"x": 70.0, "y": 6.0}, {"x": 6.0, "y": 36.0}],
            "panel_offsets": [], "unknown_parts": ["PN-A", "PN-B"]}


def build_frame(program, missing=(), contrast=0.22):
    """Low-contrast parts by default: a boldly visible synthetic part
    clears the thresholds so easily that no sensitivity in range can
    make it fail, which would make the tuning tests vacuous. A subtle
    part is also closer to the real problem the slider exists for."""
    fid = [(f["x"], f["y"]) for f in program["fiducials"]]
    anchor = fid + [(c["x"], c["y"]) for c in program["components"]]
    H = place_homography(anchor, scale=25.0, angle_deg=2.0)
    frame, _ = make_synthetic_board_frame(fid, H, image_size=autosize_canvas(anchor, H), noise_std=2.0)
    draw_components(frame, program["components"], H, PART_SIZES,
                    missing_designators=missing, contrast=contrast)
    return frame, H


# ---------------------------------------------------------------------
# threshold maths
# ---------------------------------------------------------------------

def test_effective_thresholds_layering():
    base = (8.0, 25.0)
    assert effective_thresholds("PN-A", *base) == (8.0, 25.0)
    # per-part override replaces the base
    per_part = {"PN-A": {"std_min": 3.0, "range_min": 10.0}}
    assert effective_thresholds("PN-A", *base, per_part) == (3.0, 10.0)
    assert effective_thresholds("PN-B", *base, per_part) == (8.0, 25.0)
    # sensitivity scales whatever is in force, tuned parts included
    assert effective_thresholds("PN-A", *base, per_part, 2.0) == (6.0, 20.0)
    assert effective_thresholds("PN-B", *base, per_part, 0.5) == (4.0, 12.5)
    print("OK test_effective_thresholds_layering")


def test_false_call_threshold_accepts_the_measurement():
    """The numbers suggested for a false call must actually make that
    component read present, with margin to spare."""
    std, rng = 5.4, 17.2
    t = thresholds_for_false_call(std, rng)
    assert decide_presence(std, rng, t["std_min"], t["range_min"])
    assert t["std_min"] < std and t["range_min"] < rng, t
    # margin: a slightly worse board still passes
    assert decide_presence(std * 0.9, rng * 0.9, t["std_min"], t["range_min"])
    print("OK test_false_call_threshold_accepts_the_measurement:", t)


def test_false_call_override_survives_the_sensitivity_it_was_set_at():
    """Regression: sensitivity is applied on top of per-part overrides,
    so an override recorded at 3x must be stored such that 3x still
    accepts it -- otherwise accepting a false call appears to do
    nothing whenever the slider is above 1x."""
    std, rng = 16.0, 36.0
    for sensitivity in (0.5, 1.0, 2.0, 3.0, 5.0):
        t = thresholds_for_false_call(std, rng, sensitivity=sensitivity)
        std_min, range_min = effective_thresholds("PN-A", 8.0, 25.0, {"PN-A": t}, sensitivity)
        assert decide_presence(std, rng, std_min, range_min), \
            f"override set at {sensitivity}x does not hold at {sensitivity}x: {t}"
    print("OK test_false_call_override_survives_the_sensitivity_it_was_set_at: 0.5x-5x")


def test_sensitivity_clamped():
    assert clamp_sensitivity(0.01) == 0.1
    assert clamp_sensitivity(99) == 5.0
    assert clamp_sensitivity(1.4) == 1.4
    print("OK test_sensitivity_clamped")


def test_part_threshold_persistence():
    tmp = Path(tempfile.mkdtemp()) / "programs" / "part_thresholds.json"
    assert load_part_thresholds(str(tmp)) == {}
    save_part_thresholds(str(tmp), {"PN-A": {"std_min": 3.0, "range_min": 9.0}})
    assert load_part_thresholds(str(tmp))["PN-A"]["std_min"] == 3.0
    tmp.write_text("{broken")
    assert load_part_thresholds(str(tmp)) == {}, "corrupt file must not raise"
    print("OK test_part_threshold_persistence")


# ---------------------------------------------------------------------
# tuning against a real inspection
# ---------------------------------------------------------------------

def test_sensitivity_slider_changes_verdict_without_recapture():
    """The operator's core loop: same capture, move sensitivity, watch
    the verdict change. Nothing is re-measured."""
    program = make_program()
    frame, H = build_frame(program)
    result = inspect(frame, program, PART_SIZES, H)
    assert result.verdict == "PASS", result.message
    measured = [(c.std, c.intensity_range) for u in result.units for c in u.components]

    # crank sensitivity until good parts start being called missing
    strict = reevaluate(result, PresenceThresholds(sensitivity=3.0))
    assert strict.verdict == "FAIL", "over-strict sensitivity should produce false calls"
    after = [(c.std, c.intensity_range) for u in strict.units for c in u.components]
    assert measured == after, "re-evaluation must not re-measure anything"

    # and back down again
    relaxed = reevaluate(strict, PresenceThresholds(sensitivity=1.0))
    assert relaxed.verdict == "PASS", relaxed.message
    print("OK test_sensitivity_slider_changes_verdict_without_recapture: "
          f"PASS -> FAIL(x3.0) -> PASS, {len(measured)} measurements untouched")


def test_marking_a_false_call_fixes_that_part_only():
    """Point at a wrongly-failed component, accept it, and only that
    part number's threshold moves."""
    program = make_program()
    frame, H = build_frame(program)
    result = reevaluate(inspect(frame, program, PART_SIZES, H),
                        PresenceThresholds(sensitivity=3.0))
    assert result.verdict == "FAIL"

    false_call = result.missing[0]
    # Marked while the slider sits at 3x, so the stored override has to
    # account for that or the slider scales it straight back.
    part_thresholds = {false_call.part: thresholds_for_false_call(
        false_call.std, false_call.intensity_range, sensitivity=3.0)}

    fixed = reevaluate(result, PresenceThresholds(sensitivity=3.0), part_thresholds)
    still_missing = {c.designator for c in fixed.missing}
    assert false_call.designator not in still_missing, "the marked component is still failing"
    assert all(c.part != false_call.part for c in fixed.missing), \
        "every component of the tuned part should now pass"
    print(f"OK test_marking_a_false_call_fixes_that_part_only: accepted "
          f"{false_call.designator} ({false_call.part}) -> {part_thresholds}")


def test_margin_reports_how_close_a_call_was():
    program = make_program()
    frame, H = build_frame(program)
    result = inspect(frame, program, PART_SIZES, H)
    margins = [c.margin for u in result.units for c in u.components]
    assert all(m >= 1.0 for m in margins), "all present -> every margin at or above 1"
    strict = reevaluate(result, PresenceThresholds(sensitivity=3.0))
    assert any(c.margin < 1.0 for c in strict.missing), "a missing call should read below 1"
    print(f"OK test_margin_reports_how_close_a_call_was: min margin "
          f"{min(margins):.2f} at x1.0, {min(c.margin for c in strict.missing):.2f} for a fail at x3.0")


def test_real_missing_part_survives_relaxation():
    """Tuning away false calls must not quietly hide a genuinely absent
    part -- the bare pad should still fail at normal sensitivity."""
    program = make_program()
    frame, H = build_frame(program, missing=["R3"])
    result = inspect(frame, program, PART_SIZES, H)
    assert result.verdict == "FAIL"
    assert {c.designator for c in result.missing} == {"R3"}, [c.designator for c in result.missing]
    print("OK test_real_missing_part_survives_relaxation: R3 still detected as missing")


def test_per_part_threshold_applied_during_inspect():
    program = make_program()
    frame, H = build_frame(program)
    strict = PresenceThresholds(sensitivity=3.0)
    baseline = inspect(frame, program, PART_SIZES, H, thresholds=strict)
    assert baseline.verdict == "FAIL"
    failing_part = baseline.missing[0].part
    per_part = {failing_part: {"std_min": 0.5, "range_min": 1.0}}
    tuned = inspect(frame, program, PART_SIZES, H, thresholds=strict, part_thresholds=per_part)
    assert all(c.part != failing_part for c in tuned.missing), \
        "per-part override was not applied during inspect()"
    print("OK test_per_part_threshold_applied_during_inspect:", failing_part)


def test_reevaluate_leaves_unchecked_alone():
    comp = ComponentResult(designator="U1", part=None, unit="U1", x_mm=1, y_mm=1, status="unsized")
    result = InspectionResult(verdict="INCOMPLETE", units=[UnitResult("U1", [comp])])
    out = reevaluate(result, PresenceThresholds())
    assert out.verdict == "INCOMPLETE"
    assert out.units[0].components[0].status == "unsized"
    assert not out.units[0].components[0].present
    print("OK test_reevaluate_leaves_unchecked_alone")


# ---------------------------------------------------------------------
# program editing
# ---------------------------------------------------------------------

def test_delete_designator():
    program = make_program()
    removed = delete_designators(program, ["R3", "R5"])
    assert len(removed) == 2, removed
    remaining = {c["designator"] for c in program["components"]}
    assert "R3" not in remaining and "R5" not in remaining
    assert len(program["components"]) == 6
    print("OK test_delete_designator:", describe_removal(removed))


def test_delete_part_number_removes_all_its_components():
    program = make_program()
    before = part_summary(program)
    removed = delete_part(program, "PN-A")
    assert len(removed) == before["PN-A"], (len(removed), before)
    assert all(c["part"] != "PN-A" for c in program["components"])
    assert "PN-A" not in program["unknown_parts"], program["unknown_parts"]
    print("OK test_delete_part_number_removes_all_its_components:",
          before, "->", part_summary(program))


def test_delete_unspecified_bucket():
    program = make_program()
    program["components"].append({"designator": "TP1", "x": 5.0, "y": 5.0,
                                   "rotation": 0.0, "library": None, "part": None})
    assert part_summary(program)["UNSPECIFIED"] == 1
    removed = delete_part(program, "UNSPECIFIED")
    assert len(removed) == 1 and removed[0]["designator"] == "TP1"
    assert "UNSPECIFIED" not in part_summary(program)
    print("OK test_delete_unspecified_bucket")


def test_deleting_removes_it_from_inspection():
    """The point of deleting: that component stops being judged."""
    program = make_program()
    frame, H = build_frame(program, missing=["R3"])
    assert inspect(frame, program, PART_SIZES, H).verdict == "FAIL"

    delete_designators(program, ["R3"])
    after = inspect(frame, program, PART_SIZES, H)
    assert after.verdict == "PASS", f"{after.verdict}: {after.message}"
    assert all(c.designator != "R3" for u in after.units for c in u.components)
    print("OK test_deleting_removes_it_from_inspection: FAIL -> PASS after removing R3")


def test_delete_is_a_no_op_for_unknown_names():
    program = make_program()
    assert delete_designators(program, ["NOPE"]) == []
    assert delete_designators(program, []) == []
    assert delete_part(program, "NO-SUCH-PART") == []
    assert len(program["components"]) == 8
    print("OK test_delete_is_a_no_op_for_unknown_names")


def test_edited_program_saves_and_reloads():
    tmp = Path(tempfile.mkdtemp()) / "P.json"
    program = make_program()
    delete_part(program, "PN-B")
    save_program_json(program, str(tmp))
    reloaded = json.loads(tmp.read_text())
    assert len(reloaded["components"]) == 4
    assert reloaded["unknown_parts"] == ["PN-A"]
    print("OK test_edited_program_saves_and_reloads")


if __name__ == "__main__":
    test_effective_thresholds_layering()
    test_false_call_threshold_accepts_the_measurement()
    test_false_call_override_survives_the_sensitivity_it_was_set_at()
    test_sensitivity_clamped()
    test_part_threshold_persistence()
    test_sensitivity_slider_changes_verdict_without_recapture()
    test_marking_a_false_call_fixes_that_part_only()
    test_margin_reports_how_close_a_call_was()
    test_real_missing_part_survives_relaxation()
    test_per_part_threshold_applied_during_inspect()
    test_reevaluate_leaves_unchecked_alone()
    test_delete_designator()
    test_delete_part_number_removes_all_its_components()
    test_delete_unspecified_bucket()
    test_deleting_removes_it_from_inspection()
    test_delete_is_a_no_op_for_unknown_names()
    test_edited_program_saves_and_reloads()
    print("\nALL TUNING + EDITING TESTS PASSED")
