"""Headless smoke test for the UI fixes a real 37MP board photo forced:

  * a big still photo must not be polled or converted at full size --
    that is what left the front page seemingly frozen with no result;
  * the ROI box must report its size in mm while it is being dragged,
    the way an AOI tuner does;
  * the grayscale controls must re-measure the held capture, not just
    redraw it;
  * an unsized program reports every component, so the findings list is
    capped rather than building hundreds of rows.

Run with an offscreen platform when there is no display:
    QT_QPA_PLATFORM=offscreen python tests/test_ui_readouts_smoke.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import (
    QApplication, QGraphicsSimpleTextItem, QMainWindow, QMessageBox,
)

from core.camera import StillImageSource
from core.calibration import manual_calibrate, mm_to_px_batch
from core.grayscale import load_grayscale_settings
from core.testutils import (
    autosize_canvas, draw_components, make_synthetic_board_frame, place_homography,
)
from ui.calibration_widget import ImageCanvas
from ui.live_tab import MAX_FINDINGS_SHOWN, LiveTab
from ui.program_tab import PX_PER_MM, ProgramTab

_dialogs = []


def _record_dialog(kind):
    def stub(_parent, title, text, *a, **kw):
        _dialogs.append((kind, title, text))
        return QMessageBox.Ok
    return stub


QMessageBox.information = staticmethod(_record_dialog("info"))
QMessageBox.warning = staticmethod(_record_dialog("warning"))
QMessageBox.critical = staticmethod(_record_dialog("critical"))


def send_mouse_event(widget, etype, pos, button, buttons):
    ev = QMouseEvent(etype, QPointF(pos), widget.mapToGlobal(pos), button, buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)


app = QApplication(sys.argv)
tmpdir = Path(tempfile.mkdtemp())
programs_dir = tmpdir / "programs"
programs_dir.mkdir()

FIDUCIALS = [{"x": 5.0, "y": 5.0}, {"x": 95.0, "y": 5.0}, {"x": 5.0, "y": 65.0}]
# Deliberately more components than the findings cap: an unsized program
# reports every one of them as unchecked, which is the case the cap exists
# for.
COMPONENTS = [
    {"designator": f"R{i}", "x": 12.0 + 4.0 * (i % 20), "y": 12.0 + 4.0 * (i // 20),
     "rotation": 0.0, "library": "RES", "part": f"PN-{1000 + i}"}
    for i in range(260)
]
program = {
    "name": "READOUT_BOARD", "source_file": "x.xlsx", "created": "now",
    "is_panel": False, "fiducials": FIDUCIALS, "panel_offsets": [],
    "components": COMPONENTS,
    "unknown_parts": sorted({c["part"] for c in COMPONENTS}),
}
prog_path = programs_dir / "READOUT_BOARD.json"
prog_path.write_text(json.dumps(program))


# =====================================================================
# 1. A big frame is displayed cheaply, and clicks still map to native px
# =====================================================================
canvas = ImageCanvas()
canvas.resize(640, 480)
canvas.show()
app.processEvents()

big = np.random.default_rng(0).integers(0, 255, (3000, 4000, 3), dtype=np.uint8)
start = time.time()
canvas.set_frame(big)
app.processEvents()
big_elapsed = time.time() - start

assert canvas.pixmap() is not None, "nothing was drawn"
assert canvas.pixmap().width() <= canvas.width() + 1, "pixmap must be built at display size"
assert canvas._display_scale < 0.25, (
    f"display scale {canvas._display_scale} must stay relative to the native 4000px frame")
print(f"OK: 12MP frame drawn in {big_elapsed:.3f}s at scale {canvas._display_scale:.4f}")

# a click near the middle of the widget must land near the middle of the
# *native* frame, not the scaled one
clicks = []
canvas.clicked.connect(lambda x, y: clicks.append((x, y)))
canvas.click_enabled = True
ox, oy = canvas._display_offset
mid = QPoint(int(ox + canvas.pixmap().width() / 2), int(oy + canvas.pixmap().height() / 2))
send_mouse_event(canvas, QEvent.MouseButtonPress, mid, Qt.LeftButton, Qt.LeftButton)
app.processEvents()
assert clicks, "click was not reported"
cx, cy = clicks[0]
assert abs(cx - 2000) < 40 and abs(cy - 1500) < 40, f"click mapped to {(cx, cy)}, expected ~(2000, 1500)"
print(f"OK: click on the scaled view maps back to native px {cx:.0f},{cy:.0f}")
canvas.close()


# =====================================================================
# 2. The ROI box reports its size in mm, live, while dragging
# =====================================================================
prog_win = QMainWindow()
prog_tab = ProgramTab(programs_dir=str(programs_dir),
                      part_sizes_path=str(programs_dir / "part_sizes.json"))
prog_win.setCentralWidget(prog_tab)
prog_win.resize(1100, 700)
prog_win.show()
prog_tab.load_program(str(prog_path))
prog_tab.part_list.setCurrentRow(0)
app.processEvents()
assert prog_tab.current_rect_item is not None

before_text = prog_tab.size_readout.text()
assert "mm" in before_text, before_text


def canvas_size_labels():
    return [i.text() for i in prog_tab.detail_scene.items()
            if isinstance(i, QGraphicsSimpleTextItem)]


assert len(canvas_size_labels()) == 1, \
    f"exactly one size label belongs on the canvas, got {canvas_size_labels()}"
print("OK: size shown on the panel and on the canvas ->", before_text, canvas_size_labels())

rect = prog_tab.current_rect_item.rect()
br_view = prog_tab.detail_view.mapFromScene(
    prog_tab.current_rect_item.mapToScene(rect.bottomRight()))
vp = prog_tab.detail_view.viewport()

send_mouse_event(vp, QEvent.MouseButtonPress, br_view, Qt.LeftButton, Qt.LeftButton)
app.processEvents()
mid_drag = QPoint(br_view.x() + 25, br_view.y() + 25)
send_mouse_event(vp, QEvent.MouseMove, mid_drag, Qt.NoButton, Qt.LeftButton)
app.processEvents()

during_text = prog_tab.size_readout.text()
assert during_text != before_text, \
    f"readout did not update mid-drag (still {during_text!r}) -- the whole point is live feedback"
assert len(canvas_size_labels()) == 1, \
    f"stale size labels left on the canvas: {canvas_size_labels()}"
during_spin = (prog_tab.width_spin.value(), prog_tab.height_spin.value())
print(f"OK: readout tracks the handle mid-drag -> {during_text} (spins {during_spin})")

send_mouse_event(vp, QEvent.MouseButtonRelease, mid_drag, Qt.LeftButton, Qt.NoButton)
app.processEvents()

final_rect = prog_tab.current_rect_item.rect()
final_w_mm = final_rect.width() / PX_PER_MM
assert prog_tab.size_readout.text().startswith(f"{final_w_mm:.2f}"), \
    f"after the drag the readout ({prog_tab.size_readout.text()}) must match the box ({final_w_mm:.2f}mm)"
saved = prog_tab.part_sizes[prog_tab.current_part]
assert abs(saved["width_mm"] - final_w_mm) < 0.01, (saved, final_w_mm)
print(f"OK: released size committed -> {prog_tab.size_readout.text()} / {saved}")

# typing a size must move the readout too, not just the box
prog_tab.width_spin.setValue(4.44)
prog_tab.height_spin.setValue(2.22)
app.processEvents()
assert prog_tab.size_readout.text().startswith("4.44"), prog_tab.size_readout.text()
assert "4.44" in canvas_size_labels()[0], canvas_size_labels()
print("OK: typed size updates both readouts ->", prog_tab.size_readout.text())


# =====================================================================
# 3. Live tab: still source, grayscale re-measure, capped findings
# =====================================================================
H = place_homography([(f["x"], f["y"]) for f in FIDUCIALS], scale=9.0, angle_deg=2.0)
frame, _truth = make_synthetic_board_frame(
    [(f["x"], f["y"]) for f in FIDUCIALS], H,
    image_size=autosize_canvas([(f["x"], f["y"]) for f in FIDUCIALS], H),
    noise_std=3.0, rng=np.random.default_rng(1),
)
part_sizes = {c["part"]: {"width_mm": 2.0, "height_mm": 1.2} for c in COMPONENTS}
draw_components(frame, COMPONENTS, H, part_sizes, missing_designators=("R3",))

image_path = tmpdir / "board.png"
cv2.imwrite(str(image_path), frame)

live_win = QMainWindow()
live = LiveTab(log_path=str(tmpdir / "logs" / "results.csv"),
               programs_dir=str(programs_dir),
               part_thresholds_path=str(programs_dir / "part_thresholds.json"))
live_win.setCentralWidget(live)
live_win.resize(1200, 760)
live_win.show()
app.processEvents()

live.set_source(StillImageSource(path=str(image_path)))
app.processEvents()
assert not live._timer.isActive(), "a still photo must not be polled -- that is the freeze"
assert live.live_frame is not None, "a still source must still be read once"
assert live.start_btn.text() == "Start Live"
print("OK: still source read once, timer left stopped")

live.set_program(program, part_sizes)
fiducials_mm = [(f["x"], f["y"]) for f in FIDUCIALS]
live.calibration = manual_calibrate(
    fiducials_mm,
    [tuple(p) for p in mm_to_px_batch(H, np.asarray(fiducials_mm, dtype=np.float64))])
live.run_inspection()
app.processEvents()
assert live.last_result is not None, "inspection produced no result"
print("OK: inspection ran on the still photo ->", live.last_result.verdict,
      f"({len(live.last_result.missing)} missing)")

# grayscale: changing the channel must re-measure, not just repaint
before = {c.designator: c.std for c in live.last_result.units[0].components}
live.gray_mode_combo.setCurrentIndex([live.gray_mode_combo.itemData(i)
                                      for i in range(live.gray_mode_combo.count())].index("red"))
app.processEvents()
assert live.gray_settings.mode == "red", live.gray_settings
after = {c.designator: c.std for c in live.last_result.units[0].components}
assert any(abs(after[d] - before[d]) > 1e-6 for d in before), \
    "switching channel left every measurement identical -- it was not re-measured"
print("OK: channel change re-measured the held capture ->", live.gray_settings.summary())

# preview draws the channel the operator is tuning
live.gray_preview_btn.setChecked(True)
live._on_preview_toggled(True)
app.processEvents()
assert live.canvas._frame_bgr is not None
live.gray_preview_btn.setChecked(False)
live._on_preview_toggled(False)
app.processEvents()
print("OK: grayscale preview toggles without disturbing the result")

live.reset_grayscale()
app.processEvents()
assert live.gray_settings.is_default, live.gray_settings
assert live.gray_mode_combo.currentData() == "luma"
print("OK: reset returns to plain luma")

# saved tuning must carry the grayscale settings, not only thresholds
live.gray_mode_combo.setCurrentIndex([live.gray_mode_combo.itemData(i)
                                      for i in range(live.gray_mode_combo.count())].index("green"))
app.processEvents()
live.save_tuning()
app.processEvents()
assert load_grayscale_settings(live.grayscale_path).mode == "green", \
    "grayscale settings were not persisted with the tuning"
print("OK: grayscale persisted with the tuning")

# an unsized program reports every component; the list must stay capped
held_calibration = live.calibration
live.set_program(program, {})       # same board, no part sizes at all
live.calibration = held_calibration  # the board has not moved
live.run_inspection()
app.processEvents()
unchecked = len(live.last_result.unchecked)
assert unchecked > MAX_FINDINGS_SHOWN, \
    f"test needs more than {MAX_FINDINGS_SHOWN} findings to exercise the cap, got {unchecked}"
rows = live.missing_list.count()
assert rows <= MAX_FINDINGS_SHOWN + 1, f"{rows} rows listed for {unchecked} findings"
last_row = live.missing_list.item(rows - 1).text()
assert "more" in last_row, f"the cap must say what was hidden, got {last_row!r}"
print(f"OK: {unchecked} findings listed as {rows} rows -> {last_row!r}")

assert not _dialogs or all(k == "info" for k, _t, _x in _dialogs), \
    f"unexpected modal dialog(s): {_dialogs}"

shot = tmpdir / "readouts.png"
prog_win.grab().save(str(shot))
print("SCREENSHOT:", shot)

print("\nALL CHECKS PASSED")
