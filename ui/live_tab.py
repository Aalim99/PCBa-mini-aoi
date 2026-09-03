"""
live_tab.py

The Live Inspection tab: live camera view, align the board against the
program's fiducials, then trigger a single-frame capture + inspection
pass on operator command (button or Space) -- not continuous
frame-by-frame checking.

Alignment prefers the fiducials taught in Program Manager (F1/F2/F3 by
appearance and geometry), falls back to blob detection, then to manual
clicking. Whichever path ran is named on screen, because how the board
was aligned changes how much to trust a marginal call.

Tuning lives here rather than in a settings file: the sensitivity slider
and the "part is present" action re-decide the capture already on
screen, so the operator sees the effect of a change on the board that
provoked it.

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
    QPushButton, QMessageBox, QListWidget, QListWidgetItem, QComboBox,
    QFileDialog, QShortcut, QSlider, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.calibration import CalibrationResult, auto_calibrate
from core.camera import Camera, StillImageSource, list_cameras
from core.fiducials import align_with_templates, get_fiducial_refs, load_templates
from core.inspection import (
    InspectionResult, PresenceThresholds, expanded_fiducials_mm, inspect, reevaluate,
)
from core.barcode_reader import read_barcode
from core.grayscale import (
    MODES as GRAY_MODES, MODE_LABELS as GRAY_MODE_LABELS, GrayscaleSettings,
    load_grayscale_settings, save_grayscale_settings, to_gray as gray_convert,
)
from core.result_log import append_result
from core.thresholds import (
    MAX_SENSITIVITY, MIN_SENSITIVITY, clamp_sensitivity, load_part_thresholds,
    save_part_thresholds, thresholds_for_false_call,
)
from ui.calibration_widget import ImageCanvas, CalibrationDialog
from ui.theme import COLORS, Card, muted_label, verdict_style

SLIDER_STEPS = 100
# Cap on findings listed: an unsized program reports every component as
# unchecked, and hundreds of rows is slow to build and unreadable.
MAX_FINDINGS_SHOWN = 200


def _sensitivity_from_slider(value: int) -> float:
    """Slider position -> multiplier, spread logarithmically so the fine
    detail sits around 1.0 where tuning actually happens."""
    frac = value / SLIDER_STEPS
    return float(np.exp(np.log(MIN_SENSITIVITY) + frac * (np.log(MAX_SENSITIVITY) - np.log(MIN_SENSITIVITY))))


def _slider_from_sensitivity(sensitivity: float) -> int:
    sensitivity = clamp_sensitivity(sensitivity)
    frac = (np.log(sensitivity) - np.log(MIN_SENSITIVITY)) / (np.log(MAX_SENSITIVITY) - np.log(MIN_SENSITIVITY))
    return int(round(frac * SLIDER_STEPS))


class LiveTab(QWidget):
    inspected = pyqtSignal(object)  # InspectionResult, so the logs tab can refresh

    def __init__(self, log_path="logs/results.csv", programs_dir="programs",
                 part_thresholds_path="programs/part_thresholds.json",
                 grayscale_path=None):
        super().__init__()
        self.log_path = log_path
        self.programs_dir = programs_dir
        self.part_thresholds_path = part_thresholds_path
        self.grayscale_path = grayscale_path or str(
            Path(part_thresholds_path).with_name("grayscale.json"))
        self.program: Optional[dict] = None
        self.part_sizes = {}
        self.part_thresholds = load_part_thresholds(part_thresholds_path)
        self.gray_settings = load_grayscale_settings(self.grayscale_path)
        self.calibration: Optional[CalibrationResult] = None
        self.fiducial_templates = {}
        self.source = None
        self.cameras = []
        self.live_frame: Optional[np.ndarray] = None
        self.captured_frame: Optional[np.ndarray] = None
        self.last_result: Optional[InspectionResult] = None
        self.thresholds = PresenceThresholds()
        # After a pass the view holds the annotated capture so the
        # operator can read it; frames keep being captured underneath so
        # the next trigger still inspects a fresh one.
        self._frozen = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._grab_frame)

        self._build_ui()
        QTimer.singleShot(150, lambda: self.scan_cameras(show_result=False))

    # ---------- UI ----------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(12)
        root.addLayout(self._build_viewer(), stretch=5)
        root.addLayout(self._build_sidebar(), stretch=2)

    def _build_viewer(self):
        column = QVBoxLayout()
        column.setSpacing(10)

        self.canvas = ImageCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        column.addWidget(self.canvas, stretch=1)

        source_card = Card("Source")
        row = QHBoxLayout()
        row.setSpacing(8)
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(230)
        self.camera_combo.addItem("(not scanned yet)", None)
        row.addWidget(self.camera_combo, stretch=1)

        self.rescan_btn = QPushButton("Rescan")
        self.rescan_btn.setProperty("variant", "ghost")
        self.rescan_btn.clicked.connect(self.scan_cameras)
        row.addWidget(self.rescan_btn)

        self.start_btn = QPushButton("Start Live")
        self.start_btn.clicked.connect(self.toggle_live)
        row.addWidget(self.start_btn)

        self.still_btn = QPushButton("Open Image...")
        self.still_btn.setProperty("variant", "ghost")
        self.still_btn.clicked.connect(self.load_still_image)
        row.addWidget(self.still_btn)

        self.resume_btn = QPushButton("Resume Live")
        self.resume_btn.setProperty("variant", "ghost")
        self.resume_btn.setEnabled(False)
        self.resume_btn.clicked.connect(self.resume_live)
        row.addWidget(self.resume_btn)
        source_card.body.addLayout(row)

        # Only shown when something is actually wrong with the feed, so
        # a blank picture is explained rather than left to interpret.
        self.source_warning = QLabel("")
        self.source_warning.setWordWrap(True)
        self.source_warning.setStyleSheet(f"color: {COLORS['warn']};")
        self.source_warning.hide()
        source_card.body.addWidget(self.source_warning)
        column.addWidget(source_card)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.calibrate_btn = QPushButton("Align Board")
        self.calibrate_btn.setProperty("variant", "primary")
        self.calibrate_btn.setMinimumHeight(44)
        self.calibrate_btn.clicked.connect(self.calibrate)
        actions.addWidget(self.calibrate_btn, stretch=2)

        self.inspect_btn = QPushButton("INSPECT   (Space)")
        self.inspect_btn.setProperty("variant", "trigger")
        self.inspect_btn.setMinimumHeight(44)
        self.inspect_btn.clicked.connect(self.run_inspection)
        actions.addWidget(self.inspect_btn, stretch=3)
        column.addLayout(actions)

        trigger = QShortcut(QKeySequence(Qt.Key_Space), self)
        trigger.activated.connect(self.run_inspection)
        return column

    def _build_sidebar(self):
        column = QVBoxLayout()
        column.setSpacing(12)

        self.verdict_label = QLabel("READY")
        self.verdict_label.setAlignment(Qt.AlignCenter)
        self.verdict_label.setMinimumHeight(84)
        self._style_verdict("")
        column.addWidget(self.verdict_label)

        status = Card("Board")
        self.program_label = muted_label("No active program")
        self.calib_label = muted_label("Not aligned")
        self.barcode_label = muted_label("Barcode: -")
        status.body.addWidget(self.program_label)
        status.body.addWidget(self.calib_label)
        status.body.addWidget(self.barcode_label)
        column.addWidget(status)

        findings = Card("Findings")
        self.units_label = muted_label("")
        findings.body.addWidget(self.units_label)
        self.missing_list = QListWidget()
        self.missing_list.currentItemChanged.connect(self._on_finding_selected)
        findings.body.addWidget(self.missing_list, stretch=1)

        self.accept_btn = QPushButton("This part IS present  (false call)")
        self.accept_btn.setProperty("variant", "ghost")
        self.accept_btn.setEnabled(False)
        self.accept_btn.setToolTip(
            "Tell the station a component it called missing is actually fitted.\n"
            "Lowers the threshold for that part number only, and re-decides\n"
            "the capture on screen straight away."
        )
        self.accept_btn.clicked.connect(self.accept_false_call)
        findings.body.addWidget(self.accept_btn)
        column.addWidget(findings, stretch=1)

        tuning = Card("Sensitivity")
        self.sensitivity_label = muted_label("")
        tuning.body.addWidget(self.sensitivity_label)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(0, SLIDER_STEPS)
        self.sensitivity_slider.setValue(_slider_from_sensitivity(1.0))
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        tuning.body.addWidget(self.sensitivity_slider)

        scale_row = QHBoxLayout()
        left = QLabel("fewer false calls")
        left.setProperty("variant", "muted")
        right = QLabel("stricter")
        right.setProperty("variant", "muted")
        right.setAlignment(Qt.AlignRight)
        scale_row.addWidget(left)
        scale_row.addWidget(right)
        tuning.body.addLayout(scale_row)

        buttons = QHBoxLayout()
        self.reset_tuning_btn = QPushButton("Reset")
        self.reset_tuning_btn.setProperty("variant", "ghost")
        self.reset_tuning_btn.clicked.connect(self.reset_tuning)
        buttons.addWidget(self.reset_tuning_btn)
        self.save_tuning_btn = QPushButton("Save Tuning")
        self.save_tuning_btn.clicked.connect(self.save_tuning)
        buttons.addWidget(self.save_tuning_btn)
        tuning.body.addLayout(buttons)
        column.addWidget(tuning)
        self._update_sensitivity_label()

        column.addWidget(self._build_grayscale_card())
        return column

    def _build_grayscale_card(self):
        """Which channel the presence check measures, and how it is toned.
        On a green board the red channel often separates parts from the
        mask far better than luma, which changes every measurement."""
        card = Card("Grayscale")
        self.gray_summary = muted_label("")
        card.body.addWidget(self.gray_summary)

        row = QHBoxLayout()
        label = QLabel("Channel")
        label.setProperty("variant", "muted")
        row.addWidget(label)
        self.gray_mode_combo = QComboBox()
        for mode in GRAY_MODES:
            self.gray_mode_combo.addItem(GRAY_MODE_LABELS[mode], mode)
        self.gray_mode_combo.setCurrentIndex(
            max(0, GRAY_MODES.index(self.gray_settings.mode)))
        self.gray_mode_combo.currentIndexChanged.connect(self._on_grayscale_changed)
        row.addWidget(self.gray_mode_combo, stretch=1)
        card.body.addLayout(row)

        self.gamma_slider = self._tone_slider(
            card, "Gamma", 20, 300, int(self.gray_settings.gamma * 100))
        self.contrast_slider = self._tone_slider(
            card, "Contrast", 20, 300, int(self.gray_settings.contrast * 100))
        self.brightness_slider = self._tone_slider(
            card, "Brightness", -100, 100, int(self.gray_settings.brightness))

        actions = QHBoxLayout()
        self.gray_reset_btn = QPushButton("Reset")
        self.gray_reset_btn.setProperty("variant", "ghost")
        self.gray_reset_btn.clicked.connect(self.reset_grayscale)
        actions.addWidget(self.gray_reset_btn)
        self.gray_preview_btn = QPushButton("Preview")
        self.gray_preview_btn.setProperty("variant", "ghost")
        self.gray_preview_btn.setCheckable(True)
        self.gray_preview_btn.setToolTip(
            "Show the capture as the presence check actually sees it")
        self.gray_preview_btn.clicked.connect(self._on_preview_toggled)
        actions.addWidget(self.gray_preview_btn)
        card.body.addLayout(actions)

        self._update_grayscale_summary()
        return card

    def _tone_slider(self, card, name, low, high, value):
        row = QHBoxLayout()
        label = QLabel(name)
        label.setProperty("variant", "muted")
        label.setMinimumWidth(74)
        row.addWidget(label)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(value)
        slider.valueChanged.connect(self._on_grayscale_changed)
        row.addWidget(slider, stretch=1)
        card.body.addLayout(row)
        return slider

    def _style_verdict(self, verdict):
        fg, bg = verdict_style(verdict)
        self.verdict_label.setStyleSheet(
            f"background:{bg}; color:{fg}; border:2px solid {fg}; border-radius:10px;"
            f"font-size:30px; font-weight:800; letter-spacing:2px;"
        )

    # ---------- program wiring ----------
    def set_program(self, program: dict, part_sizes: dict):
        """Called when Program Manager activates a program. Alignment is
        dropped, since it belonged to the previous board."""
        self.program = program
        self.part_sizes = dict(part_sizes or {})
        self.calibration = None
        self.fiducial_templates = {}
        if program:
            self.fiducial_templates = load_templates(program.get("name", ""), self.programs_dir)
        self._set_calibration_label()

        if not program:
            self.program_label.setText("No active program")
            return
        refs = get_fiducial_refs(program)
        taught = sum(1 for r in refs if r.id in self.fiducial_templates)
        if refs:
            fid_note = (f"{len(refs)} defined ({', '.join(r.id for r in refs)}), "
                        f"{taught} taught")
        else:
            fid_note = f"{len(expanded_fiducials_mm(program))} from the XY file, none defined"
        self.program_label.setText(
            f"<b>{program.get('name', '?')}</b><br>"
            f"{len(program.get('components') or [])} components<br>"
            f"<span style='color:{COLORS['faint']}'>Fiducials: {fid_note}</span>"
        )

    def _set_calibration_label(self):
        if self.calibration and self.calibration.success:
            how = {"template": "taught fiducials", "auto": "auto-detect",
                   "manual": "manual clicks"}.get(self.calibration.method, self.calibration.method)
            # A 2- or 3-point fit is exact by construction, so quoting its
            # RMS would read as accuracy it cannot have. Show the match
            # score instead, which does say how well the marks were found.
            if self.calibration.rms_is_meaningful:
                quality = f"RMS {self.calibration.rms_error_px:.2f}px"
            elif self.calibration.match_score:
                quality = f"match {self.calibration.match_score:.2f}"
            else:
                quality = f"exact {self.calibration.inlier_count}-point fit"
            self.calib_label.setText(
                f"<span style='color:{COLORS['pass']}'><b>Aligned</b></span> via {how} - "
                f"{self.calibration.inlier_count} points, {quality}"
            )
        else:
            self.calib_label.setText(
                f"<span style='color:{COLORS['warn']}'><b>Not aligned</b></span> - "
                f"press Align Board before inspecting"
            )

    # ---------- camera ----------
    def scan_cameras(self, show_result=True):
        """Populate the camera list with devices that actually deliver a
        frame, so the operator picks from what exists instead of guessing
        an index."""
        self.rescan_btn.setEnabled(False)
        self.rescan_btn.setText("Scanning...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            cameras = list_cameras()
        finally:
            QApplication.restoreOverrideCursor()
            self.rescan_btn.setEnabled(True)
            self.rescan_btn.setText("Rescan")

        self.cameras = cameras
        self.camera_combo.clear()
        if cameras:
            for info in cameras:
                self.camera_combo.addItem(info["label"], info["index"])
        else:
            self.camera_combo.addItem("No cameras detected", None)

        if show_result and not cameras:
            QMessageBox.information(
                self, "No cameras detected",
                "No working camera was found on indices 0-5.\n\n"
                "Check that the camera is plugged in and not in use by another "
                "application (Teams, Zoom, the Windows Camera app), then press "
                "Rescan. You can also use 'Open Image...' to work from a saved capture."
            )
        return cameras

    def toggle_live(self):
        if self._timer.isActive():
            self.stop_live()
            return

        index = self.camera_combo.currentData()
        if index is None:
            cameras = self.scan_cameras(show_result=False)
            if not cameras:
                QMessageBox.warning(
                    self, "No camera",
                    "No working camera was found on indices 0-5.\n\n"
                    "Check the camera is plugged in and not already in use by "
                    "another application, then press Rescan -- or use "
                    "'Open Image...' to work from a saved capture."
                )
                return
            index = self.camera_combo.currentData()

        cam = Camera(index)
        if not cam.open():
            detected = ", ".join(str(c["index"]) for c in self.cameras) or "none"
            QMessageBox.warning(
                self, "Could not open camera",
                f"Camera {index} did not open.\n\nCurrently detected: {detected}.\n"
                "It may have been unplugged or claimed by another application. "
                "Press Rescan to refresh the list."
            )
            return
        self.set_source(cam)

    def set_source(self, source):
        self.stop_live()
        self.resume_live()
        self.source_warning.hide()
        self.source = source
        if not source.is_open and not source.open():
            QMessageBox.warning(self, "Source unavailable", f"Could not open {source.description}.")
            self.source = None
            return

        if getattr(source, "is_static", False):
            # A still image never changes: read it once. Polling it would
            # re-convert the whole frame every tick for no new information.
            self._grab_frame()
            self.start_btn.setText("Start Live")
        else:
            self._timer.start(33)
            self.start_btn.setText("Stop Live")
        self._warn_if_blank(source)

    def _warn_if_blank(self, source):
        """A camera whose pixel format could not be negotiated returns a
        flat colour rather than failing, which otherwise reaches the
        operator as an unexplained blank rectangle."""
        if not getattr(source, "delivers_blank_frames", False):
            return
        self.source_warning.setText(
            f"{source.description} opens but returns blank frames — no image data. "
            "Usually the resolution is more than the USB link can carry. Try another "
            "camera entry, close any other app using it, or plug it into a USB 3 port. "
            "Run scripts/camera_probe.py to see which formats this camera can deliver."
        )
        self.source_warning.show()

    def stop_live(self):
        self._timer.stop()
        if self.source is not None:
            self.source.release()
        self.start_btn.setText("Start Live")

    def load_still_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open frame", "",
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

    def current_frame(self) -> Optional[np.ndarray]:
        if self.live_frame is not None:
            return self.live_frame
        if self.source is not None:
            return self.source.read()
        return None

    # ---------- alignment ----------
    def calibrate(self):
        """Align the board. Taught fiducials first (specific marks, known
        geometry), then blob detection, then manual clicking."""
        if not self._require(self.program, "Load a program in the Program Manager tab first."):
            return
        frame = self.current_frame()
        if not self._require(frame is not None,
                             "No frame available - start the camera or open an image."):
            return

        refs = get_fiducial_refs(self.program)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        attempts = []

        if refs and self.fiducial_templates:
            result = align_with_templates(frame, refs, self.fiducial_templates)
            if result.success:
                self._accept_calibration(result)
                return
            attempts.append(f"taught fiducials: {result.message}")

        fallback_points = ([(r.x_mm, r.y_mm) for r in refs] if refs
                           else expanded_fiducials_mm(self.program))
        if not self._require(len(fallback_points) >= 2,
                             "This program has fewer than 2 fiducials, so the board cannot "
                             "be aligned. Define them in Program Manager."):
            return

        result = auto_calibrate(gray, fallback_points)
        if result.success:
            self._accept_calibration(result)
            return
        attempts.append(f"auto-detect: {result.message}")

        # Manual clicking, in the order the fiducials are defined.
        labels = [r.id for r in refs] if refs else None
        dialog = CalibrationDialog(frame, fallback_points, parent=self, labels=labels)
        dialog.widget.status_label.setText(" | ".join(attempts))
        if dialog.exec_() == dialog.Accepted and dialog.result_calibration:
            self._accept_calibration(dialog.result_calibration)
        else:
            self._set_calibration_label()

    def _accept_calibration(self, result):
        self.calibration = result
        self._set_calibration_label()

    # ---------- inspection ----------
    def run_inspection(self):
        if not self._require(self.program, "No active program."):
            return
        if not self._require(self.calibration and self.calibration.success,
                             "Board is not aligned yet - press Align Board first."):
            return
        frame = self.current_frame()
        if not self._require(frame is not None, "No frame to inspect."):
            return

        frame = frame.copy()  # freeze this capture; live view keeps moving
        self.captured_frame = frame
        barcode = read_barcode(frame)
        result = inspect(frame, self.program, self.part_sizes, self.calibration.homography,
                         thresholds=self.thresholds, barcode=barcode,
                         part_thresholds=self.part_thresholds,
                         gray_settings=self.gray_settings)
        self.last_result = result

        self._frozen = True
        self.resume_btn.setEnabled(True)
        self._show_result(result)

        try:
            append_result(self.log_path, result)
        except OSError as exc:
            QMessageBox.warning(self, "Log write failed", f"Result not written to {self.log_path}:\n{exc}")

        self.inspected.emit(result)

    def _overlay(self, frame, result: InspectionResult):
        out = frame.copy()
        # alignment points first, so they sit under the component boxes
        for (x, y) in (self.calibration.matched_px if self.calibration else []):
            cv2.drawMarker(out, (int(x), int(y)), (255, 200, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
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
        self._style_verdict(result.verdict)
        self.barcode_label.setText(f"Barcode: {result.barcode or '(unreadable)'}")

        if self.captured_frame is not None:
            self.canvas.set_frame(self._overlay(self.captured_frame, result))

        self.missing_list.clear()
        # A program whose parts are not yet sized reports every component
        # as unchecked; listing hundreds of rows is slow and tells the
        # operator nothing they cannot read from the counts.
        shown = 0
        for comp in result.missing:
            if shown >= MAX_FINDINGS_SHOWN:
                break
            shown += 1
            item = QListWidgetItem(
                f"MISSING   {comp.unit}:{comp.designator}   {comp.part or '-'}"
                f"      {comp.margin:.2f}x"
            )
            item.setData(Qt.UserRole, comp)
            item.setToolTip(
                f"Measured variation {comp.std:.1f} (needs {comp.std_min:.1f}), "
                f"range {comp.intensity_range:.1f} (needs {comp.range_min:.1f}).\n"
                f"{comp.margin:.2f}x means it reached {comp.margin * 100:.0f}% of what was required."
            )
            self.missing_list.addItem(item)
        for comp in result.unchecked:
            if shown >= MAX_FINDINGS_SHOWN:
                break
            shown += 1
            item = QListWidgetItem(f"UNCHECKED   {comp.unit}:{comp.designator}   ({comp.status})")
            item.setData(Qt.UserRole, None)
            self.missing_list.addItem(item)

        hidden = len(result.missing) + len(result.unchecked) - shown
        if hidden > 0:
            more = QListWidgetItem(f"… and {hidden} more (see the CSV log for the full list)")
            more.setData(Qt.UserRole, None)
            more.setForeground(QColor(COLORS["faint"]))
            self.missing_list.addItem(more)

        units = ", ".join(f"{u.label}={'PASS' if u.passed else 'FAIL'}" for u in result.units)
        self.units_label.setText(f"{result.message}<br>Units: {units}")

    def _on_finding_selected(self, current, _previous):
        comp = current.data(Qt.UserRole) if current else None
        self.accept_btn.setEnabled(comp is not None)

    # ---------- tuning ----------
    def _on_sensitivity_changed(self, value):
        self.thresholds.sensitivity = clamp_sensitivity(_sensitivity_from_slider(value))
        self._update_sensitivity_label()
        self._reevaluate_current()

    def _update_sensitivity_label(self):
        tuned = len(self.part_thresholds)
        note = f" - {tuned} part(s) individually tuned" if tuned else ""
        self.sensitivity_label.setText(f"<b>{self.thresholds.sensitivity:.2f}x</b>{note}")

    def _reevaluate_current(self):
        """Re-decide the capture on screen. Nothing is re-measured, so
        the operator sees the effect on the very board that prompted the
        change."""
        if not self.last_result:
            return
        result = reevaluate(self.last_result, self.thresholds, self.part_thresholds)
        self._show_result(result)

    def accept_false_call(self):
        """The operator says a component called missing is really there:
        lower that part number's threshold to just under what it actually
        measured, and re-decide immediately."""
        item = self.missing_list.currentItem()
        comp = item.data(Qt.UserRole) if item else None
        if comp is None:
            return
        if not comp.part:
            QMessageBox.information(
                self, "No part number",
                f"{comp.designator} has no part number, so there is nothing to tune "
                "against. Remove it in Program Manager if it should not be inspected."
            )
            return

        self.part_thresholds[comp.part] = thresholds_for_false_call(
            comp.std, comp.intensity_range, sensitivity=self.thresholds.sensitivity)
        self._update_sensitivity_label()
        self._reevaluate_current()

    # ---------- grayscale ----------
    def _on_grayscale_changed(self, _value=None):
        self.gray_settings = GrayscaleSettings(
            mode=self.gray_mode_combo.currentData() or "luma",
            gamma=self.gamma_slider.value() / 100.0,
            contrast=self.contrast_slider.value() / 100.0,
            brightness=float(self.brightness_slider.value()),
        )
        self._update_grayscale_summary()
        # Changing the channel changes what every ROI measures, so this
        # cannot be re-decided from stored numbers the way sensitivity is
        # -- the capture has to be measured again.
        self._remeasure_current()
        if self.gray_preview_btn.isChecked():
            self._show_grayscale_preview()

    def _update_grayscale_summary(self):
        self.gray_summary.setText(self.gray_settings.summary())

    def _remeasure_current(self):
        """Re-run the presence measurement over the held capture."""
        if self.captured_frame is None or not self.program or not self.calibration:
            return
        result = inspect(self.captured_frame, self.program, self.part_sizes,
                         self.calibration.homography, thresholds=self.thresholds,
                         barcode=self.last_result.barcode if self.last_result else None,
                         part_thresholds=self.part_thresholds,
                         gray_settings=self.gray_settings)
        self.last_result = result
        self._show_result(result)

    def _on_preview_toggled(self, checked):
        if checked:
            self._show_grayscale_preview()
        elif self.last_result is not None:
            self._show_result(self.last_result)
        elif self.live_frame is not None:
            self.canvas.set_frame(self.live_frame)

    def _show_grayscale_preview(self):
        frame = self.captured_frame if self.captured_frame is not None else self.live_frame
        if frame is None:
            return
        gray = gray_convert(frame, self.gray_settings)
        self.canvas.set_frame(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    def reset_grayscale(self):
        self.gray_settings = GrayscaleSettings()
        for widget, value in ((self.gray_mode_combo, 0),
                              (self.gamma_slider, 100),
                              (self.contrast_slider, 100),
                              (self.brightness_slider, 0)):
            widget.blockSignals(True)
            widget.setCurrentIndex(value) if widget is self.gray_mode_combo else widget.setValue(value)
            widget.blockSignals(False)
        self._update_grayscale_summary()
        self._remeasure_current()
        if self.gray_preview_btn.isChecked():
            self._show_grayscale_preview()

    def reset_tuning(self):
        if self.part_thresholds and QMessageBox.question(
            self, "Reset tuning",
            f"Discard the sensitivity setting and {len(self.part_thresholds)} "
            f"per-part threshold(s)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        self.part_thresholds = {}
        self.thresholds = PresenceThresholds()
        self.sensitivity_slider.blockSignals(True)
        self.sensitivity_slider.setValue(_slider_from_sensitivity(1.0))
        self.sensitivity_slider.blockSignals(False)
        self._update_sensitivity_label()
        self._reevaluate_current()

    def save_tuning(self):
        try:
            save_part_thresholds(self.part_thresholds_path, self.part_thresholds)
            save_grayscale_settings(self.grayscale_path, self.gray_settings)
        except OSError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        QMessageBox.information(
            self, "Tuning saved",
            f"Sensitivity {self.thresholds.sensitivity:.2f}x, "
            f"{len(self.part_thresholds)} per-part threshold(s) and grayscale "
            f"({self.gray_settings.summary()}) saved to\n{self.part_thresholds_path}"
        )

    # ---------- helpers ----------
    def _require(self, condition, message) -> bool:
        if condition:
            return True
        QMessageBox.warning(self, "Cannot continue", message)
        return False

    def closeEvent(self, event):
        self.stop_live()
        super().closeEvent(event)


if __name__ == "__main__":
    import tempfile
    from core.fiducials import FiducialRef, set_fiducial_refs
    from core.testutils import (
        make_synthetic_board_frame, place_homography, autosize_canvas, draw_components,
    )
    from ui.theme import apply_theme

    app = apply_theme(QApplication(sys.argv))
    win = QMainWindow()
    win.setWindowTitle("Live Inspection - Standalone Test")
    tmp = Path(tempfile.mkdtemp())
    tab = LiveTab(log_path=str(tmp / "results.csv"), programs_dir=str(tmp),
                  part_thresholds_path=str(tmp / "part_thresholds.json"))
    win.setCentralWidget(tab)
    win.resize(1400, 860)
    win.show()

    part_sizes = {"PN-1001": {"width_mm": 2.0, "height_mm": 1.2},
                  "PN-2002": {"width_mm": 3.0, "height_mm": 2.0}}
    comps = [{"designator": f"R{i + 1}", "x": 12.0 + (i % 4) * 18.0, "y": 12.0 + (i // 4) * 18.0,
              "rotation": 0.0, "library": "L", "part": "PN-1001" if i % 2 else "PN-2002"}
             for i in range(8)]
    fiducials = [{"x": 4.0, "y": 4.0}, {"x": 74.0, "y": 7.0}, {"x": 8.0, "y": 40.0}]
    program = {"name": "DEMO_BOARD", "is_panel": False, "components": comps,
               "fiducials": fiducials, "panel_offsets": []}
    set_fiducial_refs(program, [FiducialRef(f"F{i + 1}", f["x"], f["y"])
                                 for i, f in enumerate(fiducials)])

    fid_mm = [(f["x"], f["y"]) for f in fiducials]
    anchor = fid_mm + [(c["x"], c["y"]) for c in comps]
    H = place_homography(anchor, scale=12.0, angle_deg=2.0)
    frame, _ = make_synthetic_board_frame(fid_mm, H, image_size=autosize_canvas(anchor, H),
                                           noise_std=3.0)
    draw_components(frame, comps, H, part_sizes, missing_designators=["R3"], contrast=0.3)

    tab.set_program(program, part_sizes)
    tab.set_source(StillImageSource(image=frame))
    sys.exit(app.exec_())
