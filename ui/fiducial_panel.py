"""
fiducial_panel.py

Defining the board's alignment fiducials (F1, F2, F3) in Program
Manager, and teaching what each one looks like from the reference photo.

Three named points is the AOI convention and the useful maximum for a
flat board: they pin translation, rotation and scale, and the third side
of the triangle gives a redundant check that the right marks were found.

A point can be set three ways, in increasing order of freedom:
  * auto-suggested from the XY file's Pattern Fiducial rows,
  * chosen from that list explicitly,
  * clicked anywhere on the aligned reference photo -- which matters
    when the XY file's fiducial rows are not where the operator wants to
    align from, or when a panel lists many and only three should be used.
"""

import sys
from pathlib import Path
from typing import List, Optional

import cv2
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.calibration import px_to_mm
from core.fiducials import (
    DEFAULT_IDS, FiducialRef, delete_templates, get_fiducial_refs, load_templates,
    save_templates, set_fiducial_refs, suggest_fiducial_refs, teach_templates,
)
from ui.theme import COLORS, Card, muted_label


class PointPickerDialog(QDialog):
    """Click one point on the reference photo. Returns board mm, via the
    reference alignment, so a picked point means the same thing to the
    live camera as one taken from the XY file."""

    def __init__(self, image, homography, prompt, parent=None):
        super().__init__(parent)
        from ui.calibration_widget import ImageCanvas

        self.setWindowTitle(prompt)
        self.resize(920, 700)
        self.homography = homography
        self.picked_mm = None

        layout = QVBoxLayout(self)
        self.status = QLabel(prompt)
        layout.addWidget(self.status)

        self.canvas = ImageCanvas()
        self.canvas.set_frame(image)
        self.canvas.click_enabled = True
        self.canvas.clicked.connect(self._on_click)
        layout.addWidget(self.canvas, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.ok_btn = QPushButton("Use This Point")
        self.ok_btn.setProperty("variant", "primary")
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self.accept)
        buttons.addWidget(self.ok_btn)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)

        self._image = image

    def _on_click(self, x_px, y_px):
        self.picked_mm = px_to_mm(self.homography, x_px, y_px)
        marked = self._image.copy()
        cv2.drawMarker(marked, (int(x_px), int(y_px)), (0, 220, 255),
                       cv2.MARKER_CROSS, 26, 2)
        cv2.circle(marked, (int(x_px), int(y_px)), 16, (0, 220, 255), 1)
        self.canvas.set_frame(marked)
        self.status.setText(
            f"Picked board position {self.picked_mm[0]:.2f}, {self.picked_mm[1]:.2f} mm. "
            "Click again to move it."
        )
        self.ok_btn.setEnabled(True)


class FiducialPanel(QWidget):
    """The F1/F2/F3 card. Emits `changed` whenever the program's
    fiducial definitions or taught templates are modified."""

    changed = pyqtSignal()

    def __init__(self, programs_dir="programs"):
        super().__init__()
        self.programs_dir = programs_dir
        self.program: Optional[dict] = None
        self.reference_image = None
        self.reference_homography = None
        self.templates = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.card = Card("Alignment fiducials")
        self.help_label = muted_label(
            "Three points the station aligns every board from. Define them once here."
        )
        self.card.body.addWidget(self.help_label)

        self.rows = []
        for fid in DEFAULT_IDS:
            row = QHBoxLayout()
            row.setSpacing(6)

            tag = QLabel(fid)
            tag.setFixedWidth(26)
            tag.setStyleSheet(f"font-weight:700; color:{COLORS['accent']};")
            row.addWidget(tag)

            coord = QLabel("not set")
            coord.setProperty("variant", "muted")
            row.addWidget(coord, stretch=1)

            from_xy = QComboBox()
            from_xy.setMinimumWidth(120)
            from_xy.currentIndexChanged.connect(
                lambda _i, f=fid: self._on_combo_changed(f))
            row.addWidget(from_xy)

            pick = QPushButton("Pick")
            pick.setProperty("variant", "ghost")
            pick.setToolTip("Click this point directly on the reference photo")
            pick.clicked.connect(lambda _c, f=fid: self.pick_on_reference(f))
            row.addWidget(pick)

            self.card.body.addLayout(row)
            self.rows.append({"id": fid, "coord": coord, "combo": from_xy, "pick": pick})

        actions = QHBoxLayout()
        self.suggest_btn = QPushButton("Auto-suggest")
        self.suggest_btn.setProperty("variant", "ghost")
        self.suggest_btn.setToolTip("Pick a well-spread trio from the XY file's fiducial rows")
        self.suggest_btn.clicked.connect(self.auto_suggest)
        actions.addWidget(self.suggest_btn)

        self.teach_btn = QPushButton("Teach from Reference")
        self.teach_btn.setProperty("variant", "primary")
        self.teach_btn.setToolTip(
            "Cut a template of each fiducial from the reference photo.\n"
            "Live Inspection then finds these exact marks by appearance."
        )
        self.teach_btn.clicked.connect(self.teach_from_reference)
        actions.addWidget(self.teach_btn)
        self.card.body.addLayout(actions)

        self.status_label = muted_label("")
        self.card.body.addWidget(self.status_label)
        layout.addWidget(self.card)

    # ---------- state ----------
    def set_program(self, program: Optional[dict]):
        self.program = program
        self.templates = (load_templates(program.get("name", ""), self.programs_dir)
                          if program else {})
        self._populate_combos()
        self.refresh()

    def set_reference(self, image, homography):
        self.reference_image = image
        self.reference_homography = homography
        self.refresh()

    def _available_points(self):
        return [(float(f["x"]), float(f["y"])) for f in (self.program or {}).get("fiducials", [])]

    def _populate_combos(self):
        points = self._available_points()
        for row in self.rows:
            combo = row["combo"]
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("from XY...", None)
            for i, (x, y) in enumerate(points):
                combo.addItem(f"#{i + 1}  {x:.1f}, {y:.1f}", (x, y))
            combo.setEnabled(bool(points))
            combo.blockSignals(False)

    def _refs(self) -> List[FiducialRef]:
        return get_fiducial_refs(self.program) if self.program else []

    def _set_ref(self, fid: str, x_mm: float, y_mm: float):
        refs = {r.id: r for r in self._refs()}
        refs[fid] = FiducialRef(fid, float(x_mm), float(y_mm))
        ordered = [refs[i] for i in DEFAULT_IDS if i in refs]
        set_fiducial_refs(self.program, ordered)
        # A moved fiducial invalidates the template taught at the old spot.
        if fid in self.templates:
            self.templates.pop(fid, None)
            delete_templates(self.program.get("name", ""), self.programs_dir, ids=(fid,))
        self.refresh()
        self.changed.emit()

    # ---------- actions ----------
    def _on_combo_changed(self, fid: str):
        row = next(r for r in self.rows if r["id"] == fid)
        point = row["combo"].currentData()
        if point and self.program:
            self._set_ref(fid, point[0], point[1])
        row["combo"].blockSignals(True)
        row["combo"].setCurrentIndex(0)
        row["combo"].blockSignals(False)

    def auto_suggest(self):
        if not self.program:
            return
        suggested = suggest_fiducial_refs(self.program)
        if not suggested:
            QMessageBox.information(
                self, "No fiducials in the XY file",
                "This program has no Pattern Fiducial rows to suggest from.\n\n"
                "Load a reference image and use Pick to place the points by hand."
            )
            return
        set_fiducial_refs(self.program, suggested)
        for ref in suggested:
            delete_templates(self.program.get("name", ""), self.programs_dir, ids=(ref.id,))
            self.templates.pop(ref.id, None)
        self.refresh()
        self.changed.emit()

    def pick_on_reference(self, fid: str):
        if not self.program:
            return
        if self.reference_image is None or self.reference_homography is None:
            QMessageBox.information(
                self, "No reference image",
                "Load and align a reference image first -- a picked point is "
                "converted to board millimetres through that alignment."
            )
            return
        dialog = PointPickerDialog(self.reference_image, self.reference_homography,
                                   f"Click {fid} on the board", parent=self)
        if dialog.exec_() == QDialog.Accepted and dialog.picked_mm:
            self._set_ref(fid, *dialog.picked_mm)

    def teach_from_reference(self):
        if not self.program:
            return
        refs = self._refs()
        if len(refs) < 2:
            QMessageBox.information(
                self, "Define the fiducials first",
                "Set at least two of F1/F2/F3 before teaching their appearance."
            )
            return
        if self.reference_image is None or self.reference_homography is None:
            QMessageBox.information(
                self, "No reference image",
                "Teaching cuts a template of each fiducial out of the reference "
                "photo, so load and align one first."
            )
            return

        templates = teach_templates(self.reference_image, self.reference_homography, refs)
        if not templates:
            QMessageBox.warning(
                self, "Nothing to teach",
                "None of the defined fiducials fall inside the reference image."
            )
            return
        save_templates(templates, self.program.get("name", ""), self.programs_dir)
        self.templates = templates
        self.refresh()
        self.changed.emit()

    # ---------- display ----------
    def refresh(self):
        refs = {r.id: r for r in self._refs()}
        has_reference = self.reference_image is not None and self.reference_homography is not None

        for row in self.rows:
            ref = refs.get(row["id"])
            taught = row["id"] in self.templates
            if ref:
                mark = "✓ taught" if taught else "not taught"
                colour = COLORS["pass"] if taught else COLORS["warn"]
                row["coord"].setText(
                    f"{ref.x_mm:.2f}, {ref.y_mm:.2f} mm  "
                    f"<span style='color:{colour}'>{mark}</span>"
                )
            else:
                row["coord"].setText("not set")
            row["pick"].setEnabled(bool(self.program) and has_reference)

        self.suggest_btn.setEnabled(bool(self.program))
        self.teach_btn.setEnabled(bool(self.program) and len(refs) >= 2 and has_reference)

        if not self.program:
            self.status_label.setText("")
            return
        if not refs:
            self.status_label.setText(
                f"<span style='color:{COLORS['warn']}'>None defined.</span> "
                "Live Inspection will fall back to searching every fiducial in the XY file."
            )
        elif len(self.templates) >= min(2, len(refs)):
            self.status_label.setText(
                f"<span style='color:{COLORS['pass']}'>Ready.</span> "
                f"Live Inspection will align on {len(self.templates)} taught mark(s)."
            )
        elif not has_reference:
            self.status_label.setText(
                f"{len(refs)} defined. Load a reference image to teach their appearance."
            )
        else:
            self.status_label.setText(
                f"{len(refs)} defined, none taught yet - press Teach from Reference."
            )
