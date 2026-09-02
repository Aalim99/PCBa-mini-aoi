"""
live_tab.py

The Live Inspection tab: live camera view, calibrate against the active
program's fiducials, then trigger a single-frame capture + inspection
pass on operator command (button or Space) -- not continuous
frame-by-frame checking.

Shows PASS/FAIL/INCOMPLETE with the missing-component list, overlays
every ROI on the captured frame, reads the traceability barcode from
the same frame, and appends the result to the CSV log.

Run standalone for testing:
    python ui/live_tab.py
(with no camera attached it falls back to a synthetic board so the
whole pipeline can be exercised without hardware).
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QGroupBox, QListWidget, QSpinBox, QFileDialog,
    QShortcut,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.calibration import CalibrationResult, auto_calibrate
from core.camera import Camera, StillImageSource
from core.inspection import InspectionResult, PresenceThresholds, expanded_fiducials_mm, inspect
from core.barcode_reader import read_barcode
from core.result_log import append_result
from ui.calibration_widget import ImageCanvas, CalibrationDialog

VERDICT_STYLES = {
    "PASS": "background:#1b7f3b; color:white;",
    "FAIL": "background:#b3261e; color:white;",
    "INCOMPLETE": "background:#a8730a; color:white;",
    "": "background:#555; color:white;",
}


class LiveTab(QWidget):
    inspected = pyqtSignal(object)  # InspectionResult, so the logs tab can refresh

    def __init__(self, log_path="logs/results.csv"):
        super().__init__()
        self.log_path = log_path
        self.program: Optional[dict] = None
        self.part_sizes = {}
        self.calibration: Optional[CalibrationResult] = None
        self.source = None
        self.live_frame: Optional[np.ndarray] = None
        self.last_result: Optional[InspectionResult] = None
        self.thresholds = PresenceThresholds()
        # After a pass the view holds the annotated capture so the
        # operator can actually read it; frames keep being captured
        # underneath so the next trigger still inspects a fresh one.
        self._frozen = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab_frame)

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)

        left = QVBoxLayout()
        self.canvas = ImageCanvas()
        left.addWidget(self.canvas, stretch=1)

        controls = QHBoxLayout()
        self.camera_index = QSpinBox()
        self.camera_index.setRange(0, 9)
        controls.addWidget(QLabel("Camera:"))
        controls.addWidget(self.camera_index)

        self.start_btn = QPushButton("Start Live")
        self.start_btn.clicked.connect(self.toggle_live)
        controls.addWidget(self.start_btn)

        self.still_btn = QPushButton("Load Still Image...")
        self.still_btn.clicked.connect(self.load_still_image)
        controls.addWidget(self.still_btn)

        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.calibrate)
        controls.addWidget(self.calibrate_btn)

        self.resume_btn = QPushButton("Resume Live")
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self.resume_live)
        controls.addWidget(self.resume_btn)

        self.inspect_btn = QPushButton("INSPECT (Space)")
        self.inspect_btn.clicked.connect(self.run_inspection)
        controls.addWidget(self.inspect_btn, stretch=1)
        left.addLayout(controls)
        root.addLayout(left, stretch=3)

        right = QVBoxLayout()
        self.verdict_label = QLabel("NO RESULT")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setStyleSheet(VERDICT_STYLES[""] + "font-size:26px; font-weight:bold; padding:16px;")
        right.addWidget(self.verdict_label)

        self.program_label = QLabel("No active program")
        self.program_label.setWordWrap(True)
        right.addWidget(self.program_label)

        self.calib_label = QLabel("Not calibrated")
        self.calib_label.setWordWrap(True)
        right.addWidget(self.calib_label)

        self.barcode_label = QLabel("Barcode: -")
        right.addWidget(self.barcode_label)

        missing_box = QGroupBox("Missing / unchecked")
        box_layout = QVBoxLayout()
        self.missing_list = QListWidget()
        box_layout.addWidget(self.missing_list)
        missing_box.setLayout(box_layout)
        right.addWidget(missing_box, stretch=1)

        self.units_label = QLabel("")
        self.units_label.setWordWrap(True)
        right.addWidget(self.units_label)

        root.addLayout(right, stretch=2)

        trigger = QShortcut(QKeySequence(Qt.Key_Space), self)
        trigger.activated.connect(self.run_inspection)

    # ---------- program wiring ----------
    def set_program(self, program: dict, part_sizes: dict):
        """Called when the Program Manager tab activates a program.
        Calibration is dropped, since it belongs to the previous board."""
        self.program = program
        self.part_sizes = dict(part_sizes or {})
        self.calibration = None
        self._set_calibration_label()
        fid_count = len(expanded_fiducials_mm(program)) if program else 0
        self.program_label.setText(
            f"Active program: <b>{program.get('name', '?')}</b><br>"
            f"{len(program.get('components') or [])} component rows, {fid_count} fiducials"
            if program else "No active program"
        )

    def _set_calibration_label(self):
        if self.calibration and self.calibration.success:
            self.calib_label.setText(
                f"<b>Calibrated</b> ({self.calibration.method}): "
                f"{self.calibration.inlier_count} fiducials, RMS {self.calibration.rms_error_px:.2f}px"
            )
        else:
            self.calib_label.setText("<b>Not calibrated</b> - press Calibrate before inspecting")

    # ---------- camera ----------
    def toggle_live(self):
        if self._timer.isActive():
            self.stop_live()
            return
        cam = Camera(self.camera_index.value())
        if not cam.open():
            QMessageBox.warning(self, "No camera",
                                 f"Could not open camera #{self.camera_index.value()}.\n"
                                 "Attach a camera, or use 'Load Still Image...' instead.")
            return
        self.set_source(cam)

    def set_source(self, source):
        self.stop_live()
        self.resume_live()
        self.source = source
        if not source.is_open and not source.open():
            QMessageBox.warning(self, "Source unavailable", f"Could not open {source.description}.")
            self.source = None
            return
        self._timer.start(33)
        self.start_btn.setText("Stop Live")

    def stop_live(self):
        self._timer.stop()
        if self.source is not None:
            self.source.release()
        self.start_btn.setText("Start Live")

    def load_still_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load frame", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if path:
            self.set_source(StillImageSource(path=path))

    def _grab_frame(self):
        if self.source is None:
            return
        frame = self.source.read()
        if frame is None:
            return
        # Always keep the newest capture; only the display is frozen, so
        # the next trigger inspects a current frame, not the held one.
        self.live_frame = frame
        if not self._frozen:
            self.canvas.set_frame(frame)

    def resume_live(self):
        """Drop the held result view and follow the camera again."""
        self._frozen = False
        self.resume_btn.setEnabled(False)
        if self.live_frame is not None:
            self.canvas.set_frame(self.live_frame)

    # ---------- calibration ----------
    def calibrate(self):
        if not self._require(self.program, "Load a program in the Program Manager tab first."):
            return
        frame = self.current_frame()
        if not self._require(frame is not None, "No frame available - start the camera or load a still image."):
            return

        fiducials_mm = expanded_fiducials_mm(self.program)
        if not self._require(len(fiducials_mm) >= 2,
                             "This program has fewer than 2 fiducials, so the board cannot be aligned."):
            return

        result = auto_calibrate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), fiducials_mm)
        if result.success:
            # No confirmation popup: the operator calibrates then inspects
            # immediately, and the status label already reports the result.
            self.calibration = result
            self._set_calibration_label()
            return

        # Auto-detect failed or was ambiguous -> manual click fallback.
        dialog = CalibrationDialog(frame, fiducials_mm, parent=self)
        dialog.widget.status_label.setText("Auto-detect: " + result.message)
        if dialog.exec_() == dialog.Accepted and dialog.result_calibration:
            self.calibration = dialog.result_calibration
        self._set_calibration_label()

    def current_frame(self) -> Optional[np.ndarray]:
        if self.live_frame is not None:
            return self.live_frame
        if self.source is not None:
            return self.source.read()
        return None

    # ---------- inspection ----------
    def run_inspection(self):
        if not self._require(self.program, "No active program."):
            return
        if not self._require(self.calibration and self.calibration.success,
                             "Board is not calibrated yet - press Calibrate first."):
            return
        frame = self.current_frame()
        if not self._require(frame is not None, "No frame to inspect."):
            return

        frame = frame.copy()  # freeze this capture; live view keeps moving
        barcode = read_barcode(frame)
        result = inspect(frame, self.program, self.part_sizes, self.calibration.homography,
                         thresholds=self.thresholds, barcode=barcode)
        self.last_result = result

        self._frozen = True
        self.resume_btn.setEnabled(True)
        self.canvas.set_frame(self._overlay(frame, result))
        self._show_result(result)

        try:
            append_result(self.log_path, result)
        except OSError as exc:
            QMessageBox.warning(self, "Log write failed", f"Result not written to {self.log_path}:\n{exc}")

        self.inspected.emit(result)

    def _overlay(self, frame, result: InspectionResult):
        out = frame.copy()
        for unit in result.units:
            for comp in unit.components:
                if comp.roi_px is None:
                    continue
                x, y, w, h = comp.roi_px
                if comp.status != "checked":
                    color = (150, 150, 150)
                elif comp.present:
                    color = (0, 200, 0)
                else:
                    color = (0, 0, 255)
                cv2.rectangle(out, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                if comp.missing:
                    cv2.putText(out, f"{comp.unit}:{comp.designator}", (int(x), int(y) - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return out

    def _show_result(self, result: InspectionResult):
        self.verdict_label.setText(result.verdict)
        self.verdict_label.setStyleSheet(
            VERDICT_STYLES.get(result.verdict, VERDICT_STYLES[""])
            + "font-size:26px; font-weight:bold; padding:16px;"
        )
        self.barcode_label.setText(f"Barcode: {result.barcode or '(unreadable)'}")

        self.missing_list.clear()
        for comp in result.missing:
            self.missing_list.addItem(f"MISSING  {comp.unit}:{comp.designator}  ({comp.part})")
        for comp in result.unchecked:
            self.missing_list.addItem(f"UNCHECKED  {comp.unit}:{comp.designator}  ({comp.status})")

        self.units_label.setText(
            "Units: " + ", ".join(f"{u.label}={'PASS' if u.passed else 'FAIL'}" for u in result.units)
            + f"<br>{result.message}"
        )

    def _require(self, condition, message) -> bool:
        if condition:
            return True
        QMessageBox.warning(self, "Cannot continue", message)
        return False

    def closeEvent(self, event):
        self.stop_live()
        super().closeEvent(event)


if __name__ == "__main__":
    import json
    import tempfile
    from core.testutils import (
        make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
    )

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Live Inspection - Standalone Test")
    tab = LiveTab(log_path=str(Path(tempfile.mkdtemp()) / "results.csv"))
    win.setCentralWidget(tab)
    win.resize(1200, 760)
    win.show()

    # Synthesize a board so the tab is usable with no camera attached.
    part_sizes = {"PN-1001": {"width_mm": 2.0, "height_mm": 1.2},
                  "PN-2002": {"width_mm": 3.0, "height_mm": 2.0}}
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
    draw_components(frame, comps, H, part_sizes, missing_designators=["R3"])  # R3 left bare

    tab.set_program(program, part_sizes)
    tab.set_source(StillImageSource(image=frame))
    sys.exit(app.exec_())
