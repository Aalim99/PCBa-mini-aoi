"""Render screenshots of every tab against a synthetic board, for
reviewing the UI without a camera. Not part of the app.

    QT_QPA_PLATFORM=offscreen python scripts/shoot_ui.py [outdir]
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

import main as app_main
from core.fiducials import FiducialRef, save_templates, set_fiducial_refs, teach_templates
from core.inspection import PresenceThresholds
from core.reference_image import save_reference
from core.result_log import append_result
from core.camera import StillImageSource
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
)
from ui.theme import apply_theme

out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
out_dir.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp())

app_main.PROGRAMS_DIR = work / "programs"
app_main.PART_SIZES_PATH = app_main.PROGRAMS_DIR / "part_sizes.json"
app_main.PART_THRESHOLDS_PATH = app_main.PROGRAMS_DIR / "part_thresholds.json"
app_main.LOG_PATH = work / "logs" / "results.csv"
app_main.PROGRAMS_DIR.mkdir(parents=True)

PART_SIZES = {"PN-1001": {"width_mm": 2.0, "height_mm": 1.2},
              "PN-2002": {"width_mm": 3.2, "height_mm": 2.0}}
comps = [{"designator": f"{'R' if i % 2 else 'C'}{i + 1}",
          "x": 14.0 + (i % 5) * 16.0, "y": 14.0 + (i // 5) * 15.0,
          "rotation": 90.0 if i % 3 == 0 else 0.0, "library": "LIB",
          "part": "PN-1001" if i % 2 else "PN-2002"} for i in range(15)]
fiducials = [{"x": 5.0, "y": 5.0}, {"x": 82.0, "y": 8.0}, {"x": 9.0, "y": 48.0}]
program = {"name": "DEMO_BOARD", "is_panel": False, "components": comps,
           "fiducials": fiducials, "panel_offsets": [], "unknown_parts": ["PN-1001", "PN-2002"]}
refs = [FiducialRef(f"F{i + 1}", f["x"], f["y"]) for i, f in enumerate(fiducials)]
set_fiducial_refs(program, refs)

program_path = app_main.PROGRAMS_DIR / "DEMO_BOARD.json"
program_path.write_text(json.dumps(program))
app_main.PART_SIZES_PATH.write_text(json.dumps(PART_SIZES))

fid_mm = [(f["x"], f["y"]) for f in fiducials]
anchor = fid_mm + [(c["x"], c["y"]) for c in comps]
H = place_homography(anchor, scale=13.0, angle_deg=1.5)
frame, _ = make_synthetic_board_frame(fid_mm, H, image_size=autosize_canvas(anchor, H), noise_std=3.0)
draw_components(frame, comps, H, PART_SIZES, missing_designators=["R6", "C10"], contrast=0.55)

photo_path = work / "reference.png"
cv2.imwrite(str(photo_path), frame)
save_reference("DEMO_BOARD", str(app_main.PROGRAMS_DIR), str(photo_path), H)
save_templates(teach_templates(frame, H, refs), "DEMO_BOARD", str(app_main.PROGRAMS_DIR))

app = apply_theme(QApplication(sys.argv))
win = app_main.MainWindow()
win.resize(1500, 900)
win.show()
app.processEvents()

# --- Program Manager ---
win.tabs.setCurrentWidget(win.program_tab)
win.program_tab.load_program(str(program_path))
app.processEvents()
win.program_tab.part_list.setCurrentRow(1)
app.processEvents()
win.grab().save(str(out_dir / "01_program_manager.png"))

# --- Live Inspection, after a pass ---
win.program_tab.activate_program()
app.processEvents()
win.live_tab.set_source(StillImageSource(image=frame))
app.processEvents()
win.live_tab.calibrate()
app.processEvents()
win.live_tab.run_inspection()
app.processEvents()
win.grab().save(str(out_dir / "02_live_inspection.png"))

# --- Live Inspection with a finding selected (false-call action live) ---
if win.live_tab.missing_list.count():
    win.live_tab.missing_list.setCurrentRow(0)
    app.processEvents()
    win.grab().save(str(out_dir / "03_false_call.png"))

# --- Logs ---
for _ in range(6):
    win.live_tab.run_inspection()
    app.processEvents()
win.tabs.setCurrentWidget(win.logs_tab)
win.logs_tab.refresh()
app.processEvents()
win.grab().save(str(out_dir / "04_logs.png"))

print("Screenshots in:", out_dir)
for p in sorted(out_dir.glob("*.png")):
    print(" ", p)
