"""Headless smoke test for the reference-image feature in the Program
Manager tab: a reference photo aligned to the program's fiducials must
put the real component behind the ROI box at true millimetre scale.
"""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from ui.program_tab import ProgramTab, PX_PER_MM
from core.reference_image import load_reference
from core.testutils import (
    make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
)

_dialogs = []
QMessageBox.information = staticmethod(lambda *a, **k: _dialogs.append(a) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: _dialogs.append(a) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: _dialogs.append(a) or QMessageBox.Ok)

tmpdir = Path(tempfile.mkdtemp())
programs_dir = tmpdir / "programs"
programs_dir.mkdir()

TRUE_SIZES = {"PN-BIG": {"width_mm": 4.0, "height_mm": 2.5}}
comps = [
    {"designator": "J1", "x": 20.0, "y": 20.0, "rotation": 0.0, "library": "L", "part": "PN-BIG"},
    {"designator": "J2", "x": 40.0, "y": 30.0, "rotation": 90.0, "library": "L", "part": "PN-BIG"},
]
fiducials = [{"x": 5.0, "y": 5.0}, {"x": 55.0, "y": 8.0}, {"x": 9.0, "y": 45.0}]
program = {"name": "REF_BOARD", "is_panel": False, "components": comps,
           "fiducials": fiducials, "panel_offsets": [], "unknown_parts": ["PN-BIG"]}
program_path = programs_dir / "REF_BOARD.json"
program_path.write_text(json.dumps(program))

# A photo of a known-good board, at a realistic camera resolution.
fid_mm = [(f["x"], f["y"]) for f in fiducials]
anchor = fid_mm + [(c["x"], c["y"]) for c in comps]
H_true = place_homography(anchor, scale=30.0, angle_deg=4.0)
photo, _ = make_synthetic_board_frame(fid_mm, H_true, image_size=autosize_canvas(anchor, H_true),
                                       noise_std=2.0)
draw_components(photo, comps, H_true, TRUE_SIZES)
photo_path = tmpdir / "good_board.png"
cv2.imwrite(str(photo_path), photo)

app = QApplication(sys.argv)
win = QMainWindow()
tab = ProgramTab(programs_dir=str(programs_dir), part_sizes_path=str(programs_dir / "part_sizes.json"))
win.setCentralWidget(tab)
win.resize(1200, 700)
win.show()
app.processEvents()

tab.load_program(str(program_path))
app.processEvents()
tab.part_list.setCurrentRow(0)
app.processEvents()
assert tab.current_part == "PN-BIG"
print("OK: program loaded, part selected ->", tab.current_part)

# --- 1. no reference yet: the editor still works, just without a backdrop ---
assert tab._reference_patch(160.0) is None
assert tab._backdrop_item is None
print("OK: no reference -> no backdrop, editor still usable")

# --- 2. align the reference (auto path, as loading the file would) ---
from core.calibration import auto_calibrate
from core.inspection import expanded_fiducials_mm

result = auto_calibrate(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), expanded_fiducials_mm(program))
assert result.success, result.message
tab._apply_reference(photo, result.homography)
app.processEvents()
assert tab._backdrop_item is not None, "reference did not render behind the ROI box"
assert "blind" not in tab.ref_label.text(), \
    f"status still claims no reference while one is loaded: {tab.ref_label.text()}"
assert tab.current_rect_item.brush().style() == Qt.NoBrush, \
    "ROI box must be outline-only over a reference, or it hides the part being sized"
print("OK: reference aligned ->", result.message, "|", tab.ref_label.text())

# --- 3. the backdrop is at true mm scale: measure the drawn part in it ---
BOARD_COLOR = np.array([60, 130, 60])


def measure(patch):
    mask = np.linalg.norm(patch.astype(np.float64) - BOARD_COLOR, axis=2) > 45
    cols, rows = np.where(mask.any(axis=0))[0], np.where(mask.any(axis=1))[0]
    return (cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1) if len(cols) and len(rows) else (0, 0)


patch = tab._reference_patch(tab._detail_half_extent)
w_px, h_px = measure(patch)
expect_w, expect_h = 4.0 * PX_PER_MM, 2.5 * PX_PER_MM
assert abs(w_px - expect_w) <= 6, f"backdrop part {w_px}px wide, expected ~{expect_w}px"
assert abs(h_px - expect_h) <= 6, f"backdrop part {h_px}px tall, expected ~{expect_h}px"
print(f"OK: backdrop shows the part at {w_px}x{h_px}px = "
      f"{w_px / PX_PER_MM:.2f}x{h_px / PX_PER_MM:.2f}mm (true 4.00x2.50mm)")

# --- 4. sizing the box to the part gives the true mm dimensions ---
tab.width_spin.setValue(round(w_px / PX_PER_MM, 2))
tab.height_spin.setValue(round(h_px / PX_PER_MM, 2))
app.processEvents()
sized = tab.part_sizes["PN-BIG"]
assert abs(sized["width_mm"] - 4.0) <= 0.2, sized
assert abs(sized["height_mm"] - 2.5) <= 0.2, sized
print("OK: box matched to the part reads", sized, "vs true 4.0x2.5mm")

# --- 5. the rotated instance must show the part the same way up, so one
# ROI size fits every placement ---
assert len(tab.instances) == 2, tab.instances
win.grab().save(str(tmpdir / "ref_instance1.png"))
tab.step_instance(1)
app.processEvents()
assert tab.instance_index == 1
rotated = tab._reference_patch(tab._detail_half_extent)
rw, rh = measure(rotated)
assert abs(rw - w_px) <= 6 and abs(rh - h_px) <= 6, \
    f"90-degree instance reads {rw}x{rh}px, first instance {w_px}x{h_px}px"
print(f"OK: 90-degree instance J2 also reads {rw}x{rh}px (de-rotated)")
win.grab().save(str(tmpdir / "ref_instance2.png"))

# --- 6. the backdrop keeps up when the box is resized ---
before = tab._backdrop_item
tab.width_spin.setValue(9.0)
app.processEvents()
assert tab._backdrop_item is not None and tab._backdrop_item is not before, \
    "backdrop not re-cut after the window grew"
grown = tab._reference_patch(tab._detail_half_extent)
assert grown.shape[0] > patch.shape[0], "patch should widen with the window"
print("OK: backdrop re-cut on resize ->", patch.shape[:2], "->", grown.shape[:2])
tab.width_spin.setValue(4.0)
app.processEvents()

# --- 7. persistence: saved on load, restored when the program reopens ---
from core.reference_image import save_reference

save_reference("REF_BOARD", str(programs_dir), str(photo_path), result.homography)
tab2 = ProgramTab(programs_dir=str(programs_dir), part_sizes_path=str(programs_dir / "part_sizes.json"))
tab2.load_program(str(program_path))
app.processEvents()
assert tab2.reference_image is not None, "saved reference not restored on reopen"
assert np.allclose(tab2.reference_homography, result.homography)
print("OK: reference restored on reopen ->", tab2.ref_label.text())

# --- 8. clearing removes it from disk too ---
tab2.part_list.setCurrentRow(0)
app.processEvents()
tab2.clear_reference_image()
app.processEvents()
assert tab2.reference_image is None and tab2._backdrop_item is None
assert load_reference("REF_BOARD", str(programs_dir)) is None
print("OK: clear removes the reference from disk")

assert not any("critical" in str(d) for d in _dialogs), _dialogs
print("SCREENSHOTS:", tmpdir / "ref_instance1.png", tmpdir / "ref_instance2.png")
print("\nALL CHECKS PASSED")
