"""Headless smoke test for ui/live_tab.py: drives the full live pipeline
(source -> calibrate -> inspect -> overlay -> CSV log) on a synthetic
board with one deliberately missing part."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from core.camera import StillImageSource
from core.result_log import read_results
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
)
from ui.live_tab import LiveTab

# A headless run must never be able to block on a modal: fail loudly
# instead of hanging if an unexpected dialog is raised.
_dialogs = []


def _record_dialog(kind):
    def stub(_parent, title, text, *a, **kw):
        _dialogs.append((kind, title, text))
        return QMessageBox.Ok
    return stub


QMessageBox.information = staticmethod(_record_dialog("info"))
QMessageBox.warning = staticmethod(_record_dialog("warning"))
QMessageBox.critical = staticmethod(_record_dialog("critical"))

tmpdir = Path(tempfile.mkdtemp())
log_path = tmpdir / "results.csv"

PART_SIZES = {"PN-1001": {"width_mm": 2.0, "height_mm": 1.2},
              "PN-2002": {"width_mm": 3.0, "height_mm": 2.0}}
MISSING = "R3"

comps = [{"designator": f"R{i + 1}", "x": 12.0 + (i % 4) * 18.0, "y": 12.0 + (i // 4) * 18.0,
          "rotation": 0.0, "library": "L", "part": "PN-1001" if i % 2 else "PN-2002"}
         for i in range(8)]
fiducials = [{"x": 4.0, "y": 4.0}, {"x": 74.0, "y": 7.0}, {"x": 8.0, "y": 40.0}]
program = {"name": "DEMO_BOARD", "is_panel": False, "components": comps,
           "fiducials": fiducials, "panel_offsets": []}

fid_mm = [(f["x"], f["y"]) for f in fiducials]
anchor = fid_mm + [(c["x"], c["y"]) for c in comps]
H = place_homography(anchor, scale=10.0, angle_deg=2.0)
frame, _ = make_synthetic_board_frame(fid_mm, H, image_size=autosize_canvas(anchor, H), noise_std=3.0)
draw_components(frame, comps, H, PART_SIZES, missing_designators=[MISSING])

app = QApplication(sys.argv)
win = QMainWindow()
tab = LiveTab(log_path=str(log_path))
win.setCentralWidget(tab)
win.resize(1200, 760)
win.show()
app.processEvents()

# --- 1. activate program + frame source ---
tab.set_program(program, PART_SIZES)
tab.set_source(StillImageSource(image=frame))
app.processEvents()
assert tab.live_frame is not None or tab.current_frame() is not None, "no frame from source"
print("OK: program activated and still-image source running")

# --- 2. calibrate (auto path should succeed on a clean synthetic board) ---
tab.calibrate()
app.processEvents()
assert tab.calibration is not None and tab.calibration.success, "auto-calibration failed"
print("OK: calibrated ->", tab.calibration.message)

# --- 3. inspect ---
tab.run_inspection()
app.processEvents()
result = tab.last_result
assert result is not None, "no inspection result"
assert result.verdict == "FAIL", f"expected FAIL (R3 is bare), got {result.verdict}: {result.message}"
found = {c.designator for c in result.missing}
assert found == {MISSING}, f"expected missing {{{MISSING}}}, got {found}"
print("OK: inspection ->", result.verdict, "missing:", sorted(found))

# --- 4. verdict label + missing list reflect the result ---
assert tab.verdict_label.text() == "FAIL"
assert tab.missing_list.count() == 1, tab.missing_list.count()
assert MISSING in tab.missing_list.item(0).text()
print("OK: UI shows ->", tab.verdict_label.text(), "|", tab.missing_list.item(0).text())

# --- 4b. the result overlay must survive further live-frame ticks ---
# (regression: the live timer used to overwrite the annotated capture
# ~33ms after each pass, so the operator never saw the ROI boxes)
shown_after_inspect = tab.canvas._frame_bgr.copy()
assert tab._frozen, "view should hold the result after an inspection"
win.grab().save(str(tmpdir / "live_tab_result.png"))
print("SCREENSHOT (result view):", tmpdir / "live_tab_result.png")
for _ in range(5):
    tab._grab_frame()
    app.processEvents()
assert np.array_equal(tab.canvas._frame_bgr, shown_after_inspect), \
    "live frames overwrote the result overlay"
assert tab.live_frame is not None, "capture must continue underneath the frozen view"
print("OK: result overlay held across", 5, "live ticks, capture still running")

# resuming must return to the live view
tab.resume_live()
tab._grab_frame()
app.processEvents()
assert not np.array_equal(tab.canvas._frame_bgr, shown_after_inspect), "Resume Live did not restore the feed"
print("OK: Resume Live returns to the camera feed")

# --- 5. result was appended to the CSV log ---
rows = read_results(str(log_path))
assert len(rows) == 1, rows
assert rows[0]["verdict"] == "FAIL"
assert rows[0]["missing"] == f"U1:{MISSING}", rows[0]["missing"]
assert rows[0]["program"] == "DEMO_BOARD"
print("OK: logged row ->", {k: rows[0][k] for k in ("verdict", "missing", "program", "barcode")})

# --- 6. a second pass appends rather than overwrites ---
tab.run_inspection()
app.processEvents()
assert len(read_results(str(log_path))) == 2
print("OK: second inspection appended (2 rows)")

# --- 7. activating a new program must drop the stale calibration ---
tab.set_program(program, PART_SIZES)
assert tab.calibration is None, "calibration must reset when the program changes"
print("OK: calibration cleared on program change")

assert not _dialogs, f"unexpected modal dialog(s) raised during the run: {_dialogs}"
print("OK: no blocking dialogs raised")

pix = win.grab()
out_png = tmpdir / "live_tab.png"
pix.save(str(out_png))
print("SCREENSHOT:", out_png)

print("\nALL CHECKS PASSED")
