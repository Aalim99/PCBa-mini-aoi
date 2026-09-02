"""
calibration_widget.py

Reusable PyQt5 widget for board calibration: given a captured frame and
a program's known fiducial mm positions, tries automatic fiducial
detection + homography fitting first (core.calibration.auto_calibrate);
if that fails or is ambiguous, switches to manual click-to-calibrate
mode where the operator clicks each fiducial on the image in order.

Run standalone for testing:
    python ui/calibration_widget.py [path/to/test_image.png]
(with no image given, a synthetic test board is generated so the
auto-detect and manual-click paths can both be exercised without a
camera or physical board).
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.calibration import auto_calibrate, manual_calibrate, CalibrationResult

Point = Tuple[float, float]


def _to_qpixmap(bgr_image: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())  # copy: rgb buffer must outlive this call


class ImageCanvas(QLabel):
    """Displays a frame scaled to fit, and reports clicks in native
    (full-resolution) pixel coordinates regardless of display scale."""

    clicked = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self._frame_bgr: Optional[np.ndarray] = None
        self._display_scale = 1.0
        self._display_offset = (0.0, 0.0)
        self.click_enabled = False

    def set_frame(self, frame_bgr: np.ndarray):
        self._frame_bgr = frame_bgr
        self._redraw()

    def _redraw(self):
        if self._frame_bgr is None:
            return
        pix = _to_qpixmap(self._frame_bgr)
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._display_scale = scaled.width() / pix.width() if pix.width() else 1.0
        self._display_offset = (
            (self.width() - scaled.width()) / 2.0,
            (self.height() - scaled.height()) / 2.0,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._redraw()

    def mousePressEvent(self, event):
        if not self.click_enabled or self._frame_bgr is None or self._display_scale == 0:
            return
        ox, oy = self._display_offset
        x_native = (event.pos().x() - ox) / self._display_scale
        y_native = (event.pos().y() - oy) / self._display_scale
        h, w = self._frame_bgr.shape[:2]
        if 0 <= x_native <= w and 0 <= y_native <= h:
            self.clicked.emit(x_native, y_native)


class CalibrationWidget(QWidget):
    """Emits `calibrated` with the winning CalibrationResult once
    either auto-detect succeeds outright, or the operator finishes
    clicking every fiducial in manual mode."""

    calibrated = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.frame_bgr: Optional[np.ndarray] = None
        self.fiducials_mm: List[Point] = []
        self.labels: List[str] = []
        self.result: Optional[CalibrationResult] = None
        self._manual_clicks: List[Point] = []
        self._manual_active = False
        self._build_ui()

    def _label_for(self, index: int) -> str:
        """Name of the point being clicked. Named fiducials (F1/F2/F3)
        are used when the program defines them, so the prompt matches
        what the operator set up rather than an anonymous number."""
        if index < len(self.labels):
            return self.labels[index]
        return f"fiducial {index + 1}"

    def _build_ui(self):
        root = QVBoxLayout(self)
        self.canvas = ImageCanvas()
        self.canvas.clicked.connect(self._on_canvas_clicked)
        root.addWidget(self.canvas, stretch=1)

        self.status_label = QLabel("No frame loaded.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.auto_btn = QPushButton("Run Auto-Detect")
        self.auto_btn.clicked.connect(self.run_auto_calibration)
        btn_row.addWidget(self.auto_btn)

        self.manual_btn = QPushButton("Start Manual Calibration")
        self.manual_btn.clicked.connect(self.start_manual_calibration)
        btn_row.addWidget(self.manual_btn)

        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset)
        btn_row.addWidget(self.reset_btn)
        root.addLayout(btn_row)

    # ---------- public API ----------
    def load_frame(self, frame_bgr: np.ndarray, fiducials_mm: List[Point],
                   labels: Optional[List[str]] = None):
        self.frame_bgr = frame_bgr
        self.fiducials_mm = list(fiducials_mm)
        self.labels = list(labels or [])
        self.result = None
        self._manual_clicks = []
        self._manual_active = False
        self.canvas.click_enabled = False
        self.canvas.set_frame(frame_bgr)
        self.status_label.setText(
            f"Frame loaded ({frame_bgr.shape[1]}x{frame_bgr.shape[0]}px). "
            f"{len(self.fiducials_mm)} fiducial(s) expected from the program."
        )

    # ---------- auto path ----------
    def run_auto_calibration(self):
        if self.frame_bgr is None:
            QMessageBox.warning(self, "No frame", "Load a frame first.")
            return
        gray = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2GRAY)
        self.result = auto_calibrate(gray, self.fiducials_mm)
        self._draw_overlay()
        if self.result.success:
            self.status_label.setText("AUTO OK: " + self.result.message)
            self.calibrated.emit(self.result)
        else:
            self.status_label.setText(
                "AUTO FAILED: " + self.result.message + " Use manual calibration."
            )

    # ---------- manual path ----------
    def start_manual_calibration(self):
        if self.frame_bgr is None:
            QMessageBox.warning(self, "No frame", "Load a frame first.")
            return
        if not self.fiducials_mm:
            QMessageBox.warning(self, "No program", "No fiducials defined for this program.")
            return
        self._manual_active = True
        self._manual_clicks = []
        self.canvas.click_enabled = True
        self.result = None
        self._draw_overlay()
        self._update_manual_prompt()

    def _update_manual_prompt(self):
        idx = len(self._manual_clicks)
        if idx < len(self.fiducials_mm):
            mm = self.fiducials_mm[idx]
            self.status_label.setText(
                f"Click <b>{self._label_for(idx)}</b> "
                f"({idx + 1} of {len(self.fiducials_mm)}) "
                f"&mdash; board position {mm[0]:.2f}, {mm[1]:.2f} mm."
            )
        else:
            self.status_label.setText("All fiducials clicked -- computing homography...")

    def _on_canvas_clicked(self, x_px, y_px):
        if not self._manual_active:
            return
        self._manual_clicks.append((x_px, y_px))
        self._draw_overlay()
        if len(self._manual_clicks) >= len(self.fiducials_mm):
            self._manual_active = False
            self.canvas.click_enabled = False
            self.result = manual_calibrate(self.fiducials_mm, self._manual_clicks)
            if self.result.success:
                self.status_label.setText("MANUAL OK: " + self.result.message)
                self.calibrated.emit(self.result)
            else:
                self.status_label.setText("MANUAL FAILED: " + self.result.message)
            self._draw_overlay()
        else:
            self._update_manual_prompt()

    # ---------- overlay drawing ----------
    def _draw_overlay(self):
        if self.frame_bgr is None:
            return
        overlay = self.frame_bgr.copy()
        for i, (x, y) in enumerate(self._manual_clicks):
            cv2.drawMarker(overlay, (int(x), int(y)), (0, 200, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(overlay, self._label_for(i), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        if self.result is not None:
            color = (0, 220, 0) if self.result.success else (0, 0, 255)
            for x, y in self.result.matched_px:
                cv2.circle(overlay, (int(x), int(y)), 10, color, 2)
        self.canvas.set_frame(overlay)

    def reset(self):
        self.result = None
        self._manual_clicks = []
        self._manual_active = False
        self.canvas.click_enabled = False
        if self.frame_bgr is not None:
            self.canvas.set_frame(self.frame_bgr)
        self.status_label.setText("Reset. Load a frame and run auto-detect or manual calibration.")


class CalibrationDialog(QDialog):
    """Modal wrapper around CalibrationWidget, for calibrating from the
    Live Inspection tab. Accepts once a calibration succeeds."""

    def __init__(self, frame_bgr, fiducials_mm, parent=None, labels=None):
        super().__init__(parent)
        self.setWindowTitle("Align Board")
        self.resize(900, 720)
        self.result_calibration: Optional[CalibrationResult] = None

        layout = QVBoxLayout(self)
        self.widget = CalibrationWidget()
        self.widget.load_frame(frame_bgr, fiducials_mm, labels=labels)
        self.widget.calibrated.connect(self._on_calibrated)
        layout.addWidget(self.widget)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.use_btn = QPushButton("Use This Calibration")
        self.use_btn.setEnabled(False)
        self.use_btn.clicked.connect(self.accept)
        buttons.addWidget(self.use_btn)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

    def _on_calibrated(self, result):
        self.result_calibration = result
        self.use_btn.setEnabled(True)


if __name__ == "__main__":
    from core.testutils import make_synthetic_board_frame, place_homography, autosize_canvas

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Calibration Widget - Standalone Test")
    widget = CalibrationWidget()
    win.setCentralWidget(widget)
    win.resize(900, 700)
    win.show()

    fiducials_mm = [(5.0, 5.0), (95.0, 8.0), (10.0, 70.0)]
    if len(sys.argv) > 1:
        frame = cv2.imread(sys.argv[1])
        if frame is None:
            print(f"Could not read image: {sys.argv[1]}")
            sys.exit(1)
    else:
        H_true = place_homography(fiducials_mm, scale=8.0, angle_deg=4.0)
        frame, _ = make_synthetic_board_frame(
            fiducials_mm, H_true, image_size=autosize_canvas(fiducials_mm, H_true),
            distractor_circles=15, noise_std=5.0,
        )

    widget.load_frame(frame, fiducials_mm)
    sys.exit(app.exec_())
