"""Headless smoke tests for the features added on top of the base app:
defining and teaching F1/F2/F3, deleting designators and part numbers,
and tuning false calls from the Live tab.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from core.camera import StillImageSource
from core.fiducials import get_fiducial_refs, load_templates
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
)
from core.thresholds import load_part_thresholds
from ui.live_tab import LiveTab, _sensitivity_from_slider, _slider_from_sensitivity
from ui.program_tab import ProgramTab
from ui.theme import apply_theme

# Auto-answer dialogs: confirmations say Yes, notices are recorded.
_dialogs = []
QMessageBox.information = staticmethod(lambda *a, **k: _dialogs.append(("info", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: _dialogs.append(("warn", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: _dialogs.append(("crit", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: _dialogs.append(("ask", a)) or QMessageBox.Yes)

tmp = Path(tempfile.mkdtemp())
programs_dir = tmp / "programs"
programs_dir.mkdir()

PART_SIZES = {"PN-A": {"width_mm": 2.0, "height_mm": 1.2},
              "PN-B": {"width_mm": 3.0, "height_mm": 2.0}}
comps = [{"designator": f"R{i + 1}", "x": 14.0 + (i % 4) * 17.0, "y": 14.0 + (i // 4) * 16.0,
          "rotation": 0.0, "library": "L", "part": "PN-A" if i % 2 else "PN-B"}
         for i in range(8)]
fiducials = [{"x": 5.0, "y": 5.0}, {"x": 72.0, "y": 8.0}, {"x": 9.0, "y": 42.0}]
program = {"name": "UI_BOARD", "is_panel": False, "components": comps,
           "fiducials": fiducials, "panel_offsets": [], "unknown_parts": ["PN-A", "PN-B"]}
program_path = programs_dir / "UI_BOARD.json"
program_path.write_text(json.dumps(program))
(programs_dir / "part_sizes.json").write_text(json.dumps(PART_SIZES))

fid_mm = [(f["x"], f["y"]) for f in fiducials]
anchor = fid_mm + [(c["x"], c["y"]) for c in comps]
H = place_homography(anchor, scale=22.0, angle_deg=3.0)
frame, _ = make_synthetic_board_frame(fid_mm, H, image_size=autosize_canvas(anchor, H), noise_std=2.0)
draw_components(frame, comps, H, PART_SIZES, contrast=0.22)
photo = tmp / "ref.png"
cv2.imwrite(str(photo), frame)

app = apply_theme(QApplication(sys.argv))


# =====================================================================
# Program Manager: fiducials and deletion
# =====================================================================
win = QMainWindow()
tab = ProgramTab(programs_dir=str(programs_dir), part_sizes_path=str(programs_dir / "part_sizes.json"))
win.setCentralWidget(tab)
win.resize(1500, 900)
win.show()
app.processEvents()

tab.load_program(str(program_path))
app.processEvents()
print("OK: program loaded ->", tab.program["name"], len(tab.program["components"]), "components")

# --- fiducials: none defined at first ---
assert get_fiducial_refs(tab.program) == []
assert not tab.fiducial_panel.teach_btn.isEnabled(), "cannot teach with no reference and no refs"

# --- auto-suggest picks a spread trio ---
tab.fiducial_panel.auto_suggest()
app.processEvents()
refs = get_fiducial_refs(tab.program)
assert [r.id for r in refs] == ["F1", "F2", "F3"], refs
print("OK: auto-suggested fiducials ->", [(r.id, r.x_mm, r.y_mm) for r in refs])

# suggestions are persisted to the program on disk
saved = json.loads(program_path.read_text())
assert len(saved.get("fiducial_refs", [])) == 3, saved.get("fiducial_refs")
print("OK: fiducial definitions persisted to the program JSON")

# --- teach needs a reference; supply one the way the UI does ---
from core.calibration import auto_calibrate

align = auto_calibrate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), fid_mm)
assert align.success, align.message
tab._apply_reference(frame, align.homography)
app.processEvents()
assert tab.fiducial_panel.teach_btn.isEnabled(), "teach should be available once a reference exists"

tab.fiducial_panel.teach_from_reference()
app.processEvents()
templates = load_templates("UI_BOARD", str(programs_dir))
assert set(templates) == {"F1", "F2", "F3"}, templates.keys()
assert "Ready" in tab.fiducial_panel.status_label.text(), tab.fiducial_panel.status_label.text()
print("OK: taught 3 fiducial templates ->", sorted(templates))

# --- moving a fiducial drops the template taught at the old spot ---
tab.fiducial_panel._set_ref("F2", 70.0, 10.0)
app.processEvents()
assert "F2" not in load_templates("UI_BOARD", str(programs_dir)), \
    "a moved fiducial must not keep the template taught at its old position"
print("OK: moving F2 invalidated its stale template")
tab.fiducial_panel.teach_from_reference()
app.processEvents()

# --- delete a designator ---
tab.component_search.setText("R7")
app.processEvents()
assert tab.component_list.count() == 1, tab.component_list.count()
tab.component_list.selectAll()
app.processEvents()
assert tab.delete_component_btn.isEnabled()
tab.delete_selected_components()
app.processEvents()
assert all(c["designator"] != "R7" for c in tab.program["components"])
assert len(json.loads(program_path.read_text())["components"]) == 7, "delete not persisted"
print("OK: deleted designator R7, program now", len(tab.program["components"]), "components")

# --- delete a whole part number ---
tab.component_search.setText("")
app.processEvents()
row = next(i for i in range(tab.part_list.count())
           if tab.part_list.item(i).data(Qt.UserRole) == "PN-A")
tab.part_list.setCurrentRow(row)
app.processEvents()
assert tab.delete_part_btn.isEnabled()
tab.delete_selected_part()
app.processEvents()
assert all(c.get("part") != "PN-A" for c in tab.program["components"])
assert "PN-A" not in tab.program["unknown_parts"]
print("OK: deleted part PN-A, remaining parts ->",
      sorted({c["part"] for c in tab.program["components"]}))

win.grab().save(str(tmp / "program_manager.png"))


# =====================================================================
# Live Inspection: alignment via taught fiducials, and tuning
# =====================================================================
# Restore the full component list (the edits above removed half of it),
# but keep the fiducial definitions exactly as taught -- templates and
# definitions must agree or alignment silently skews the whole board.
taught_refs = [r.as_dict() for r in get_fiducial_refs(tab.program)]
program_path.write_text(json.dumps({**program, "fiducial_refs": taught_refs}))
live_win = QMainWindow()
live = LiveTab(log_path=str(tmp / "results.csv"), programs_dir=str(programs_dir),
               part_thresholds_path=str(programs_dir / "part_thresholds.json"))
live_win.setCentralWidget(live)
live_win.resize(1500, 900)
live_win.show()
app.processEvents()

live.set_program(json.loads(program_path.read_text()), PART_SIZES)
live.set_source(StillImageSource(image=frame))
app.processEvents()
assert live.fiducial_templates, "taught templates should load with the program"
print("OK: live tab loaded", len(live.fiducial_templates), "taught templates")

live.calibrate()
app.processEvents()
assert live.calibration and live.calibration.success, "alignment failed"
assert live.calibration.method == "template", \
    f"expected alignment via taught fiducials, got {live.calibration.method}"
# A 3-point fit is exact, so its RMS is structurally 0 and must not be
# presented as accuracy; the match score is the honest signal.
assert not live.calibration.rms_is_meaningful, "3 points cannot yield a meaningful RMS"
assert live.calibration.match_score > 0.5, live.calibration.match_score
assert "RMS" not in live.calib_label.text(), \
    f"3-point alignment should not quote an RMS: {live.calib_label.text()}"
print("OK: aligned via taught fiducials ->", live.calibration.message)
print("OK: quality reported honestly ->", live.calib_label.text())

live.run_inspection()
app.processEvents()
assert live.last_result.verdict == "PASS", live.last_result.message
print("OK: inspection ->", live.last_result.verdict)

# --- slider makes the same capture fail, with nothing re-measured ---
before = [(c.std, c.intensity_range) for u in live.last_result.units for c in u.components]
live.sensitivity_slider.setValue(_slider_from_sensitivity(3.0))
app.processEvents()
assert live.thresholds.sensitivity > 2.5, live.thresholds.sensitivity
assert live.last_result.verdict == "FAIL", "raising sensitivity should produce false calls"
after = [(c.std, c.intensity_range) for u in live.last_result.units for c in u.components]
assert before == after, "tuning must not re-measure the capture"
assert live.missing_list.count() > 0
print(f"OK: sensitivity {live.thresholds.sensitivity:.2f}x -> "
      f"{live.last_result.verdict}, {live.missing_list.count()} findings, no re-measure")

# --- mark one as a false call: that part is accepted, at this sensitivity ---
live.missing_list.setCurrentRow(0)
app.processEvents()
assert live.accept_btn.isEnabled(), "selecting a finding should enable the accept action"
accepted = live.missing_list.currentItem().data(Qt.UserRole)
live.accept_false_call()
app.processEvents()
assert accepted.part in live.part_thresholds, live.part_thresholds
assert all(c.part != accepted.part for c in live.last_result.missing), \
    "the accepted part still reads as missing"
print(f"OK: accepted {accepted.designator} ({accepted.part}) -> "
      f"{live.part_thresholds[accepted.part]}, verdict now {live.last_result.verdict}")

# --- tuning persists ---
live.save_tuning()
app.processEvents()
assert accepted.part in load_part_thresholds(str(programs_dir / "part_thresholds.json"))
print("OK: tuning saved to part_thresholds.json")

# --- reset clears it ---
live.reset_tuning()
app.processEvents()
assert live.part_thresholds == {}
assert abs(live.thresholds.sensitivity - 1.0) < 0.01, live.thresholds.sensitivity
assert live.last_result.verdict == "PASS", "back to defaults should pass again"
print("OK: reset -> 1.00x, no per-part overrides, verdict PASS")

# --- slider mapping is monotonic and hits 1.0 in the middle ---
assert _sensitivity_from_slider(0) < 0.2
assert _sensitivity_from_slider(100) > 4.0
values = [_sensitivity_from_slider(v) for v in range(0, 101, 10)]
assert all(b > a for a, b in zip(values, values[1:])), values
print("OK: slider maps monotonically", f"{values[0]:.2f}x .. {values[-1]:.2f}x")

live_win.grab().save(str(tmp / "live_tuning.png"))

assert not any(kind == "crit" for kind, _ in _dialogs), _dialogs
print("SCREENSHOTS:", tmp / "program_manager.png", tmp / "live_tuning.png")
print("\nALL CHECKS PASSED")
