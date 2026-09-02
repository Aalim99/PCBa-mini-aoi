"""
program_tab.py

The "Program" tab: lets you load a parsed program JSON (from
core/program_parser.py) and set the ROI box size for each unique
Part number by dragging a resize handle on a zoomed canvas (or typing
exact mm values). Sizes are saved to a shared part_sizes.json that
persists across programs, since the same part number often reappears
on different boards.

Run this file directly to test just this tab on its own:
    python ui/program_tab.py
"""

import sys
import json
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QLabel, QPushButton, QDoubleSpinBox, QFormLayout, QFileDialog,
    QMessageBox, QGroupBox, QInputDialog, QGraphicsPixmapItem
)
from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QBrush, QPainter, QImage, QPixmap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


PX_PER_MM = 40
DEFAULT_SIZE_MM = 1.0


def bgr_to_qpixmap(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy())


def make_resizable_rect_item(rect, on_resize=None, color=QColor(255, 140, 0), filled=True):
    """Factory returning a QGraphicsRectItem subclass instance with
    corner-drag resizing. Position is NOT draggable — only size
    changes, since the item represents a component's ROI box size,
    not its board location.

    `filled=False` draws outline only, for when a reference photo sits
    behind the box: a translucent fill washes out the very component
    the box is being sized against.
    """

    from PyQt5.QtWidgets import QGraphicsRectItem

    class _ResizableRectItem(QGraphicsRectItem):
        HANDLE_SIZE = 10

        def __init__(self, rect, on_resize, color):
            super().__init__(rect)
            self.on_resize = on_resize
            if filled:
                self.setPen(QPen(color, 2))
                self.setBrush(QBrush(color.lighter(160)))
                self.setOpacity(0.65)
            else:
                self.setPen(QPen(color, 2))
                self.setBrush(QBrush(Qt.NoBrush))
                self.setOpacity(1.0)
            self.setAcceptHoverEvents(True)
            self._resize_corner = None
            self._drag_start = None
            self._start_rect = None

        def _corner_at(self, pos):
            r = self.rect()
            h = self.HANDLE_SIZE
            corners = {
                "tl": r.topLeft(), "tr": r.topRight(),
                "bl": r.bottomLeft(), "br": r.bottomRight(),
            }
            for name, pt in corners.items():
                if (pos - pt).manhattanLength() <= h:
                    return name
            return None

        def hoverMoveEvent(self, event):
            corner = self._corner_at(event.pos())
            if corner in ("tl", "br"):
                self.setCursor(Qt.SizeFDiagCursor)
            elif corner in ("tr", "bl"):
                self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
            super().hoverMoveEvent(event)

        def mousePressEvent(self, event):
            corner = self._corner_at(event.pos())
            if corner:
                self._resize_corner = corner
                self._drag_start = event.pos()
                self._start_rect = QRectF(self.rect())
                event.accept()
            else:
                super().mousePressEvent(event)

        def mouseMoveEvent(self, event):
            if self._resize_corner:
                delta = event.pos() - self._drag_start
                r = QRectF(self._start_rect)
                if self._resize_corner == "br":
                    r.setBottomRight(self._start_rect.bottomRight() + delta)
                elif self._resize_corner == "tl":
                    r.setTopLeft(self._start_rect.topLeft() + delta)
                elif self._resize_corner == "tr":
                    r.setTopRight(self._start_rect.topRight() + delta)
                elif self._resize_corner == "bl":
                    r.setBottomLeft(self._start_rect.bottomLeft() + delta)
                r = r.normalized()
                if r.width() > self.HANDLE_SIZE and r.height() > self.HANDLE_SIZE:
                    self.prepareGeometryChange()
                    self.setRect(r)
                event.accept()
            else:
                super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):
            if self._resize_corner:
                self._resize_corner = None
                if self.on_resize:
                    r = self.rect()
                    self.on_resize(r.width(), r.height())
                event.accept()
            else:
                super().mouseReleaseEvent(event)

    return _ResizableRectItem(rect, on_resize, color)


class ProgramTab(QWidget):
    # Emitted with (program dict, part_sizes dict) when a program becomes
    # the active one for Live Inspection.
    program_activated = pyqtSignal(object, object)

    def __init__(self, programs_dir="programs", part_sizes_path="programs/part_sizes.json"):
        super().__init__()
        self.programs_dir = programs_dir
        self.part_sizes_path = part_sizes_path
        self.program = None
        self.program_path = None
        self.part_sizes = self._load_part_sizes()
        self.current_part = None
        self.current_rect_item = None
        # Reference board photo, aligned to the program's fiducials, so
        # ROI sizes can be set against the real part instead of guessed.
        self.reference_image = None
        self.reference_homography = None
        self.instances = []
        self.instance_index = 0

        self._build_ui()

    def resizeEvent(self, event):
        """fitInView is a one-time snapshot, not a live binding -- refit
        both canvases whenever the tab is resized (e.g. maximizing the
        window) or their content would stay scaled to the old size."""
        super().resizeEvent(event)
        if self.program and self.overview_scene.items():
            self._fit_overview()
        if self.current_rect_item is not None:
            h = self._detail_half_extent
            self.detail_view.fitInView(-h, -h, 2 * h, 2 * h, Qt.KeepAspectRatio)

    # ---------- persistence ----------
    def _load_part_sizes(self):
        p = Path(self.part_sizes_path)
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {}

    def _save_part_sizes(self):
        Path(self.part_sizes_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.part_sizes_path, "w") as f:
            json.dump(self.part_sizes, f, indent=2)

    # ---------- UI layout ----------
    def _build_ui(self):
        root = QHBoxLayout(self)

        # Left column: load program + part list
        left = QVBoxLayout()
        self.import_btn = QPushButton("Import XY File...")
        self.import_btn.clicked.connect(self.import_xy_dialog)
        left.addWidget(self.import_btn)

        self.load_btn = QPushButton("Load Program JSON")
        self.load_btn.clicked.connect(self.load_program_dialog)
        left.addWidget(self.load_btn)

        self.program_label = QLabel("No program loaded")
        self.program_label.setWordWrap(True)
        left.addWidget(self.program_label)

        self.activate_btn = QPushButton("Set Active for Inspection")
        self.activate_btn.setEnabled(False)
        self.activate_btn.clicked.connect(self.activate_program)
        left.addWidget(self.activate_btn)

        self.part_list = QListWidget()
        self.part_list.currentItemChanged.connect(self.on_part_selected)
        left.addWidget(self.part_list, stretch=1)

        root.addLayout(left, stretch=2)

        # Middle column: read-only board overview, selected part highlighted
        mid = QVBoxLayout()
        mid.addWidget(QLabel("Board overview (orange = selected part)"))
        self.overview_scene = QGraphicsScene()
        self.overview_view = QGraphicsView(self.overview_scene)
        self.overview_view.setRenderHint(QPainter.Antialiasing)
        self.overview_view.setMinimumWidth(280)
        mid.addWidget(self.overview_view, stretch=1)
        root.addLayout(mid, stretch=2)

        # Right column: zoomed drag-to-resize canvas + precise mm entry
        right = QVBoxLayout()
        right.addWidget(QLabel("Drag a corner to resize this part's ROI box"))

        ref_row = QHBoxLayout()
        self.ref_btn = QPushButton("Load Reference Image...")
        self.ref_btn.clicked.connect(self.load_reference_image)
        ref_row.addWidget(self.ref_btn)
        self.clear_ref_btn = QPushButton("Clear")
        self.clear_ref_btn.setEnabled(False)
        self.clear_ref_btn.clicked.connect(self.clear_reference_image)
        ref_row.addWidget(self.clear_ref_btn)
        right.addLayout(ref_row)

        self.ref_label = QLabel("No reference image - sizes are set blind")
        self.ref_label.setWordWrap(True)
        right.addWidget(self.ref_label)

        self.detail_scene = QGraphicsScene()
        self.detail_view = QGraphicsView(self.detail_scene)
        self.detail_view.setRenderHint(QPainter.Antialiasing)
        self.detail_view.setMinimumSize(320, 320)
        right.addWidget(self.detail_view, stretch=1)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("< Prev")
        self.prev_btn.clicked.connect(lambda: self.step_instance(-1))
        nav.addWidget(self.prev_btn)
        self.instance_label = QLabel("-")
        self.instance_label.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.instance_label, stretch=1)
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(lambda: self.step_instance(1))
        nav.addWidget(self.next_btn)
        right.addLayout(nav)

        form_box = QGroupBox("Size (mm)")
        form = QFormLayout()
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(0.1, 50.0)
        self.width_spin.setSingleStep(0.1)
        self.width_spin.setDecimals(2)
        self.width_spin.valueChanged.connect(self.on_spin_changed)

        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.1, 50.0)
        self.height_spin.setSingleStep(0.1)
        self.height_spin.setDecimals(2)
        self.height_spin.valueChanged.connect(self.on_spin_changed)

        form.addRow("Width:", self.width_spin)
        form.addRow("Height:", self.height_spin)
        form_box.setLayout(form)
        right.addWidget(form_box)

        self.save_btn = QPushButton("Save Part Sizes")
        self.save_btn.clicked.connect(self.save_all)
        right.addWidget(self.save_btn)

        root.addLayout(right, stretch=3)

    # ---------- program loading ----------
    def import_xy_dialog(self):
        """Parse a mounter XY export into a new program JSON."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import mounter XY file", "", "Excel files (*.xlsx *.xls)"
        )
        if not path:
            return
        default_name = Path(path).stem
        name, ok = QInputDialog.getText(self, "Program name", "Name for this program:",
                                         text=default_name)
        if not ok or not name.strip():
            return

        from core.program_parser import parse_program, save_program
        try:
            program = parse_program(path, name.strip())
            out_path = save_program(program, self.programs_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", f"Could not parse {Path(path).name}:\n{exc}")
            return

        unsized = [p for p in program["unknown_parts"] if p not in self.part_sizes]
        QMessageBox.information(
            self, "Program imported",
            f"Saved to {out_path}\n\n"
            f"Components: {len(program['components'])}\n"
            f"Fiducials: {len(program['fiducials'])}\n"
            f"Panel offsets: {len(program['panel_offsets'])}\n"
            f"Part numbers still needing an ROI size: {len(unsized)}"
        )
        self.load_program(out_path)

    def load_program_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Program JSON", self.programs_dir, "JSON files (*.json)"
        )
        if path:
            self.load_program(path)

    def load_program(self, path):
        with open(path) as f:
            self.program = json.load(f)
        self.program_path = path
        self.program_label.setText(
            f"{self.program['name']}  |  {len(self.program['components'])} components"
        )
        self.activate_btn.setEnabled(True)
        self._load_saved_reference()
        self._populate_part_list()
        self._draw_overview()

    def activate_program(self):
        """Hand this program to the Live Inspection tab. Sizes are saved
        first so the inspection uses what is on screen, not the last
        explicitly-saved state."""
        if not self.program:
            return
        self._save_part_sizes()
        self.program_activated.emit(self.program, self.part_sizes)
        self.program_label.setText(
            f"{self.program['name']}  |  {len(self.program['components'])} components"
            f"  |  <b>ACTIVE</b>"
        )

    def _populate_part_list(self):
        self.part_list.clear()
        if not self.program:
            return
        parts = {}
        for c in self.program["components"]:
            part = c.get("part") or "UNSPECIFIED"
            parts.setdefault(part, []).append(c)

        for part, comps in sorted(parts.items()):
            size = self.part_sizes.get(part)
            size_str = f"{size['width_mm']:.2f}x{size['height_mm']:.2f}mm" if size else "NOT SIZED"
            item = QListWidgetItem(f"{part}  ({len(comps)}x)  [{size_str}]")
            item.setData(Qt.UserRole, part)
            item.setData(Qt.UserRole + 1, len(comps))
            if not size:
                item.setForeground(QColor("red"))
            self.part_list.addItem(item)

    # ---------- reference image ----------
    def load_reference_image(self):
        """Pick a photo of a known-good board and align it to this
        program's fiducials, so the ROI editor can show the real part."""
        if not self.program:
            QMessageBox.warning(self, "No program", "Load a program first.")
            return

        from core.inspection import expanded_fiducials_mm
        fiducials_mm = expanded_fiducials_mm(self.program)
        if len(fiducials_mm) < 2:
            QMessageBox.warning(self, "Not enough fiducials",
                                 "This program has fewer than 2 fiducials, so a reference "
                                 "image cannot be aligned to board coordinates.")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Reference board image", "",
                                               "Images (*.png *.jpg *.jpeg *.bmp *.tif)")
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            QMessageBox.critical(self, "Could not read image", f"Failed to open {Path(path).name}.")
            return

        from core.calibration import auto_calibrate
        result = auto_calibrate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), fiducials_mm)
        if not result.success:
            # Same manual click fallback the live tab uses.
            from ui.calibration_widget import CalibrationDialog
            dialog = CalibrationDialog(image, fiducials_mm, parent=self)
            dialog.widget.status_label.setText("Auto-align: " + result.message)
            if dialog.exec_() != dialog.Accepted or not dialog.result_calibration:
                return
            result = dialog.result_calibration

        self._apply_reference(
            image, result.homography,
            f"<b>Reference aligned</b> ({result.method}): {result.inlier_count} fiducials, "
            f"RMS {result.rms_error_px:.2f}px"
        )

        from core.reference_image import save_reference
        try:
            save_reference(self.program["name"], self.programs_dir, path, result.homography)
        except OSError as exc:
            QMessageBox.warning(self, "Reference not saved",
                                 f"The image is aligned for this session but could not be "
                                 f"stored with the program:\n{exc}")

    def _apply_reference(self, image, homography, message=None):
        self.reference_image = image
        self.reference_homography = homography
        self.clear_ref_btn.setEnabled(True)
        self.ref_label.setText(message or "<b>Reference aligned</b> - box is shown over the real part")
        if self.current_part:
            self._draw_detail(self.current_part)

    def _load_saved_reference(self):
        """Restore a previously aligned reference when a program opens."""
        self.reference_image = None
        self.reference_homography = None
        self.clear_ref_btn.setEnabled(False)
        self.ref_label.setText("No reference image - sizes are set blind")
        if not self.program:
            return

        from core.reference_image import load_reference
        record = load_reference(self.program["name"], self.programs_dir)
        if not record:
            return
        image = cv2.imread(record["image_path"])
        if image is None:
            return
        self.reference_image = image
        self.reference_homography = record["homography"]
        self.clear_ref_btn.setEnabled(True)
        self.ref_label.setText(f"<b>Reference loaded</b>: {Path(record['image_path']).name}")

    def clear_reference_image(self):
        if not self.program:
            return
        from core.reference_image import delete_reference
        delete_reference(self.program["name"], self.programs_dir)
        self.reference_image = None
        self.reference_homography = None
        self.clear_ref_btn.setEnabled(False)
        self.ref_label.setText("No reference image - sizes are set blind")
        if self.current_part:
            self._draw_detail(self.current_part)

    # ---------- component instances ----------
    def _load_instances(self, part):
        from core.reference_image import component_instances
        self.instances = component_instances(self.program, part) if self.program else []
        self.instance_index = 0
        self._update_instance_label()

    def _update_instance_label(self):
        total = len(self.instances)
        if not total:
            self.instance_label.setText("-")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return
        inst = self.instances[self.instance_index]
        self.instance_label.setText(
            f"{inst.get('unit', 'U1')}:{inst.get('designator', '?')}"
            f"  ({self.instance_index + 1}/{total})"
            + (f"  {inst.get('rotation', 0):g}°" if inst.get("rotation") else "")
        )
        self.prev_btn.setEnabled(total > 1)
        self.next_btn.setEnabled(total > 1)

    def step_instance(self, delta):
        """Check the same ROI box against another placement of the part."""
        if not self.instances:
            return
        self.instance_index = (self.instance_index + delta) % len(self.instances)
        self._update_instance_label()
        if self.current_part:
            self._draw_detail(self.current_part)

    def _reference_patch(self, half_extent):
        """The reference image around the selected instance, resampled
        into detail-canvas scene space (PX_PER_MM per mm, de-rotated)."""
        if self.reference_image is None or self.reference_homography is None or not self.instances:
            return None
        from core.reference_image import component_patch
        inst = self.instances[self.instance_index]
        return component_patch(
            self.reference_image, self.reference_homography,
            float(inst["x"]), float(inst["y"]), float(inst.get("rotation", 0.0) or 0.0),
            half_extent=half_extent, px_per_mm=PX_PER_MM,
        )

    # ---------- overview canvas (read-only context) ----------
    def _draw_overview(self):
        self.overview_scene.clear()
        if not self.program:
            return
        comps = self.program["components"]
        if not comps:
            return
        xs = [c["x"] for c in comps]
        ys = [c["y"] for c in comps]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        span = max(maxx - minx, maxy - miny, 1)
        scale = 260.0 / span

        for c in comps:
            px = (c["x"] - minx) * scale
            py = (maxy - c["y"]) * scale  # flip Y: board Y-up -> screen Y-down
            dot = self.overview_scene.addEllipse(
                px - 1, py - 1, 2, 2, QPen(Qt.gray), QBrush(Qt.gray)
            )
            dot.setData(0, c.get("part"))

        self._fit_overview()

    def _fit_overview(self):
        # Fit to the logical board canvas (positions are scaled into a fixed
        # 260-unit span in _draw_overview), not itemsBoundingRect(): fitting
        # to the dots' own ink extent zooms into whatever tiny area they
        # occupy -- e.g. one component, or several sharing an axis --
        # blowing a 2px marker up to fill the whole pane instead of showing
        # the board.
        margin = 20
        self.overview_view.fitInView(
            QRectF(-margin, -margin, 260 + 2 * margin, 260 + 2 * margin), Qt.KeepAspectRatio
        )

    def _highlight_part_in_overview(self, part):
        for item in self.overview_scene.items():
            if item.data(0) == part:
                item.setBrush(QBrush(QColor(255, 140, 0)))
                item.setPen(QPen(QColor(255, 140, 0)))
                item.setRect(item.rect().adjusted(-1, -1, 1, 1))
            else:
                item.setBrush(QBrush(Qt.gray))
                item.setPen(QPen(Qt.gray))
                item.setRect(QRectF(item.rect().center().x() - 1, item.rect().center().y() - 1, 2, 2))

    # ---------- detail / resize canvas ----------
    def on_part_selected(self, current, previous):
        if not current or not self.program:
            return
        part = current.data(Qt.UserRole)
        self.current_part = part
        self._load_instances(part)
        self._highlight_part_in_overview(part)
        self._draw_detail(part)

    def _draw_detail(self, part):
        self.detail_scene.clear()
        self._backdrop_item = None  # cleared with the scene; don't reuse the dead item
        size = self.part_sizes.get(part, {"width_mm": DEFAULT_SIZE_MM, "height_mm": DEFAULT_SIZE_MM})
        w_px = size["width_mm"] * PX_PER_MM
        h_px = size["height_mm"] * PX_PER_MM

        # Window first: the reference patch is cut to the same extent, so
        # the real part sits behind the box at matching scale.
        self._refit_detail_window(w_px, h_px)
        self._draw_reference_backdrop()

        rect = QRectF(-w_px / 2, -h_px / 2, w_px, h_px)
        self.current_rect_item = make_resizable_rect_item(
            rect, on_resize=self._on_canvas_resize, filled=self._backdrop_item is None
        )
        self.detail_scene.addItem(self.current_rect_item)

        cross_pen = QPen(QColor(80, 80, 80))
        self.detail_scene.addLine(-15, 0, 15, 0, cross_pen)
        self.detail_scene.addLine(0, -15, 0, 15, cross_pen)

        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(size["width_mm"])
        self.height_spin.setValue(size["height_mm"])
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)

    def _draw_reference_backdrop(self):
        """Put the aligned reference image behind the ROI box, centred on
        the selected component at PX_PER_MM scene pixels per millimetre."""
        existing = getattr(self, "_backdrop_item", None)
        if existing is not None and existing.scene() is self.detail_scene:
            self.detail_scene.removeItem(existing)
        self._backdrop_item = None

        half = getattr(self, "_detail_half_extent", 160.0)
        patch = self._reference_patch(half)
        if patch is None:
            return
        item = QGraphicsPixmapItem(bgr_to_qpixmap(patch))
        item.setOffset(-half, -half)      # scene origin = component centre
        item.setZValue(-10)               # behind the ROI box and crosshair
        self.detail_scene.addItem(item)
        self._backdrop_item = item

    def _refit_detail_window(self, w_px, h_px):
        """(Re)fit the detail view's window to comfortably contain a box
        of this pixel size, keeping its corner handles reachable. Must be
        re-run any time the box grows (drag or typed mm value), not just
        on initial draw, or handles can drift off-screen mid-resize."""
        margin = 40
        half = max(w_px, h_px, 80) / 2 + margin
        self._detail_half_extent = half
        self.detail_view.setSceneRect(-half, -half, 2 * half, 2 * half)
        self.detail_view.fitInView(-half, -half, 2 * half, 2 * half, Qt.KeepAspectRatio)

    def _on_canvas_resize(self, width_px, height_px):
        if not self.current_part:
            return
        w_mm = round(width_px / PX_PER_MM, 3)
        h_mm = round(height_px / PX_PER_MM, 3)
        self.part_sizes[self.current_part] = {"width_mm": w_mm, "height_mm": h_mm}

        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(w_mm)
        self.height_spin.setValue(h_mm)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)

        self._refit_detail_window(width_px, height_px)
        self._draw_reference_backdrop()  # window grew -> re-cut the patch to match
        self._refresh_current_list_item()

    def on_spin_changed(self, _value):
        if not self.current_part or not self.current_rect_item:
            return
        w_mm = self.width_spin.value()
        h_mm = self.height_spin.value()
        self.part_sizes[self.current_part] = {"width_mm": w_mm, "height_mm": h_mm}

        w_px = w_mm * PX_PER_MM
        h_px = h_mm * PX_PER_MM
        self.current_rect_item.prepareGeometryChange()
        self.current_rect_item.setRect(-w_px / 2, -h_px / 2, w_px, h_px)
        self._refit_detail_window(w_px, h_px)
        self._draw_reference_backdrop()  # window grew -> re-cut the patch to match

        self._refresh_current_list_item()

    def _refresh_current_list_item(self):
        item = self.part_list.currentItem()
        if not item:
            return
        part = item.data(Qt.UserRole)
        count = item.data(Qt.UserRole + 1)
        size = self.part_sizes.get(part)
        size_str = f"{size['width_mm']:.2f}x{size['height_mm']:.2f}mm"
        item.setText(f"{part}  ({count}x)  [{size_str}]")
        item.setForeground(QColor("black"))

    # ---------- save ----------
    def save_all(self):
        self._save_part_sizes()
        QMessageBox.information(self, "Saved", f"Part sizes saved to {self.part_sizes_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Program Tab - Standalone Test")
    tab = ProgramTab()
    win.setCentralWidget(tab)
    win.resize(1000, 600)
    win.show()
    sys.exit(app.exec_())
