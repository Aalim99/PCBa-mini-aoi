"""Headless smoke test for ui/calibration_widget.py -- not part of the
app, run manually to verify the auto-detect and manual-click paths."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QPointF, QEvent
from PyQt5.QtGui import QMouseEvent

from ui.calibration_widget import CalibrationWidget
from core.testutils import make_synthetic_board_frame, place_homography, autosize_canvas

tmpdir = Path(tempfile.mkdtemp())


def send_mouse_click(widget, pos):
    """A plain click (no held-button drag needed here, unlike program_tab's
    resize handles) -- press+release at the same point is sufficient."""
    press = QMouseEvent(QEvent.MouseButtonPress, QPointF(pos), widget.mapToGlobal(pos),
                         Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, press)
    release = QMouseEvent(QEvent.MouseButtonRelease, QPointF(pos), widget.mapToGlobal(pos),
                           Qt.LeftButton, Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(widget, release)


app = QApplication(sys.argv)
win = QMainWindow()
cw = CalibrationWidget()
win.setCentralWidget(cw)
win.resize(900, 700)
win.show()
app.processEvents()

# --- 1. auto-detect success path (clean synthetic board) ---
fiducials_mm = [(5.0, 5.0), (95.0, 8.0), (10.0, 70.0)]
H_true = place_homography(fiducials_mm, scale=8.0, angle_deg=4.0)
frame, truth_px = make_synthetic_board_frame(
    fiducials_mm, H_true, image_size=autosize_canvas(fiducials_mm, H_true),
    distractor_circles=15, noise_std=5.0,
)
cw.load_frame(frame, fiducials_mm)
app.processEvents()
cw.run_auto_calibration()
app.processEvents()
assert cw.result is not None and cw.result.success, f"auto-detect failed: {cw.result.message if cw.result else None}"
print("OK: auto-detect succeeded ->", cw.result.message)

pix = win.grab()
pix.save(str(tmpdir / "calib_auto_ok.png"))
print("SCREENSHOT:", tmpdir / "calib_auto_ok.png")

# --- 2. reset, then manual click path ---
cw.reset()
app.processEvents()
assert cw.result is None
cw.start_manual_calibration()
app.processEvents()
assert cw._manual_active

# figure out where each fiducial appears on-screen (canvas display coords)
# by mapping the known true pixel positions through the canvas's current
# display scale/offset -- same math the widget itself uses for clicks.
canvas = cw.canvas
ox, oy = canvas._display_offset
scale = canvas._display_scale
for i, (x_native, y_native) in enumerate(truth_px):
    display_pos = QPointF(x_native * scale + ox, y_native * scale + oy)
    send_mouse_click(canvas, display_pos.toPoint())
    app.processEvents()
    print(f"  clicked fiducial {i + 1} at native=({x_native:.1f},{y_native:.1f}) display={display_pos}")

assert cw.result is not None and cw.result.success, f"manual calibration failed: {cw.result.message if cw.result else None}"
print("OK: manual calibration succeeded ->", cw.result.message)

# sanity check the recovered homography against ground truth
pts = np.asarray(fiducials_mm).reshape(-1, 1, 2)
proj = cv2.perspectiveTransform(pts, cw.result.homography).reshape(-1, 2)
err = np.max(np.linalg.norm(proj - truth_px, axis=1))
assert err < 5.0, f"manual calibration reprojection error too high: {err}"
print("OK: manual calibration reprojection error(px) =", round(float(err), 3))

pix = win.grab()
pix.save(str(tmpdir / "calib_manual_ok.png"))
print("SCREENSHOT:", tmpdir / "calib_manual_ok.png")

print("\nALL CHECKS PASSED")
