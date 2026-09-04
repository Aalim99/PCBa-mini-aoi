"""Headless integration test for main.py: the three tabs wired together.

Covers the path a real session takes -- import an XY file in Program
Manager, size its parts, set it active, calibrate and inspect in Live
Inspection, and see the result land in Logs/History.
"""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openpyxl import Workbook
from PyQt5.QtWidgets import QApplication, QMessageBox

# Redirect app data into a scratch dir before importing main, so the
# test never writes into the working copy's programs/ or logs/.
tmpdir = Path(tempfile.mkdtemp())
import main as app_main

app_main.PROGRAMS_DIR = tmpdir / "programs"
app_main.PART_SIZES_PATH = app_main.PROGRAMS_DIR / "part_sizes.json"
app_main.LOG_PATH = tmpdir / "logs" / "results.csv"
app_main.PROGRAMS_DIR.mkdir(parents=True, exist_ok=True)

from core.camera import StillImageSource
from core.result_log import COLUMNS, read_results
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
)

_dialogs = []
QMessageBox.information = staticmethod(lambda *a, **k: _dialogs.append(a) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: _dialogs.append(a) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: _dialogs.append(a) or QMessageBox.Ok)

PART_SIZES = {"PN-1001": {"width_mm": 2.0, "height_mm": 1.2},
              "PN-2002": {"width_mm": 3.0, "height_mm": 2.0}}
MISSING = "R4"

# --- build a mounter XY file the way a real export looks ---
comps = [{"designator": f"R{i + 1}", "x": 12.0 + (i % 4) * 18.0, "y": 12.0 + (i // 4) * 18.0,
          "rotation": 0.0, "part": "PN-1001" if i % 2 else "PN-2002"} for i in range(8)]
fiducials = [(4.0, 4.0), (74.0, 7.0), (8.0, 40.0)]

wb = Workbook()
ws = wb.active
ws.append(["Mounter XY Export - Integration Test"])            # title row above the header
ws.append(["X", "Y", "Rotation", "Z", "Designator", "Library", "Part",
           "Skip No", "Type", "Pattern Type", "Pattern Group"])
for c in comps:
    ws.append([c["x"], c["y"], c["rotation"], 0, c["designator"], "LIB", c["part"],
               0, "Placement", "", ""])
ws.append([0, 0, 0, 0, "----", None, None, 0, "Placement", "", ""])   # junk row
for fx, fy in fiducials:
    ws.append([fx, fy, 0, 0, None, None, None, 0, "Pattern Fiducial", "", ""])
xy_path = tmpdir / "board.xlsx"
wb.save(xy_path)

# --- synthesize the camera frame for that same board, R4 left bare ---
anchor = list(fiducials) + [(c["x"], c["y"]) for c in comps]
H = place_homography(anchor, scale=10.0, angle_deg=2.0)
frame, _ = make_synthetic_board_frame(fiducials, H, image_size=autosize_canvas(anchor, H), noise_std=3.0)
draw_components(frame, comps, H, PART_SIZES, missing_designators=[MISSING])

app = QApplication(sys.argv)
win = app_main.MainWindow()
win.show()
app.processEvents()
assert win.tabs.count() == 3, "expected 3 tabs"
print("OK: main window up with tabs ->", [win.tabs.tabText(i) for i in range(3)])

# --- 1. import the XY file through the Program Manager tab ---
from core.program_parser import parse_program, save_program

program = parse_program(str(xy_path), "INTEG_BOARD")
program_path = save_program(program, str(app_main.PROGRAMS_DIR))
assert len(program["components"]) == len(comps), program["components"]
assert len(program["fiducials"]) == len(fiducials)
print("OK: XY parsed ->", len(program["components"]), "components,",
      len(program["fiducials"]), "fiducials,", len(program["unknown_parts"]), "parts to size")

win.program_tab.load_program(program_path)
app.processEvents()
assert win.program_tab.part_list.count() == 2, win.program_tab.part_list.count()
print("OK: Program Manager listed part numbers:",
      [win.program_tab.part_list.item(i).text() for i in range(win.program_tab.part_list.count())])

# --- 2. size the parts (as the operator would, via the size spinboxes) ---
for i in range(win.program_tab.part_list.count()):
    win.program_tab.part_list.setCurrentRow(i)
    app.processEvents()
    part = win.program_tab.current_part
    win.program_tab.width_spin.setValue(PART_SIZES[part]["width_mm"])
    win.program_tab.height_spin.setValue(PART_SIZES[part]["height_mm"])
    app.processEvents()
assert win.program_tab.part_sizes == PART_SIZES, win.program_tab.part_sizes
print("OK: part sizes set ->", win.program_tab.part_sizes)

# --- 3. set active: Live Inspection should pick it up and be focused ---
win.program_tab.activate_program()
app.processEvents()
assert win.live_tab.program is not None, "activation did not reach the live tab"
assert win.live_tab.program["name"] == "INTEG_BOARD"
assert win.live_tab.part_sizes == PART_SIZES
assert win.tabs.currentWidget() is win.live_tab, "activating should switch to Live Inspection"
assert json.loads(app_main.PART_SIZES_PATH.read_text()) == PART_SIZES, "sizes not persisted on activate"
print("OK: program activated, live tab focused, sizes persisted")

# --- 4. calibrate + inspect ---
win.live_tab.set_source(StillImageSource(image=frame))
app.processEvents()
win.live_tab.calibrate()
app.processEvents()
assert win.live_tab.calibration and win.live_tab.calibration.success, "calibration failed"
print("OK: calibrated ->", win.live_tab.calibration.message)

win.live_tab.run_inspection()
app.processEvents()
result = win.live_tab.last_result
assert result.verdict == "FAIL", f"{result.verdict}: {result.message}"
assert {c.designator for c in result.missing} == {MISSING}, [c.designator for c in result.missing]
print("OK: inspection ->", result.verdict, "missing", [c.designator for c in result.missing])

# --- 5. the result reaches Logs/History without a manual refresh ---
assert win.logs_tab.table.rowCount() == 1, win.logs_tab.table.rowCount()
verdict_col = COLUMNS.index("verdict")
assert win.logs_tab.table.item(0, verdict_col).text() == "FAIL", win.logs_tab.table.item(0, verdict_col).text()
assert read_results(str(app_main.LOG_PATH))[0]["missing"] == f"U1:{MISSING}"
print("OK: logs tab auto-refreshed ->", win.logs_tab.status_label.text())

# --- 6. filters work on the history ---
win.logs_tab.verdict_combo.setCurrentText("PASS")
app.processEvents()
assert win.logs_tab.table.rowCount() == 0, "PASS filter should hide the FAIL row"
win.logs_tab.verdict_combo.setCurrentText("All")
win.logs_tab.search_edit.setText(MISSING)
app.processEvents()
assert win.logs_tab.table.rowCount() == 1, "search should match the missing designator"
print("OK: log filters (verdict + text search) work")

win.grab().save(str(tmpdir / "main_live.png"))
win.tabs.setCurrentWidget(win.logs_tab)
app.processEvents()
win.grab().save(str(tmpdir / "main_logs.png"))
win.tabs.setCurrentWidget(win.program_tab)
app.processEvents()
win.grab().save(str(tmpdir / "main_program.png"))
print("SCREENSHOTS:", tmpdir / "main_live.png", tmpdir / "main_logs.png", tmpdir / "main_program.png")

print("\nALL CHECKS PASSED")
