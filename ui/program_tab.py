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

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QListWidget, QListWidgetItem, QGraphicsView, QGraphicsScene,
    QLabel, QPushButton, QDoubleSpinBox, QFormLayout, QFileDialog,
    QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QPen, QBrush, QPainter


PX_PER_MM = 40
DEFAULT_SIZE_MM = 1.0


def make_resizable_rect_item(rect, on_resize=None, color=QColor(255, 140, 0)):
    """Factory returning a QGraphicsRectItem subclass instance with
    corner-drag resizing. Position is NOT draggable — only size
    changes, since the item represents a component's ROI box size,
    not its board location."""

    from PyQt5.QtWidgets import QGraphicsRectItem

    class _ResizableRectItem(QGraphicsRectItem):
        HANDLE_SIZE = 10

        def __init__(self, rect, on_resize, color):
            super().__init__(rect)
            self.on_resize = on_resize
            self.setPen(QPen(color, 2))
            self.setBrush(QBrush(color.lighter(160)))
            self.setOpacity(0.65)
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
    def __init__(self, programs_dir="programs", part_sizes_path="programs/part_sizes.json"):
        super().__init__()
        self.programs_dir = programs_dir
        self.part_sizes_path = part_sizes_path
        self.program = None
        self.part_sizes = self._load_part_sizes()
        self.current_part = None
        self.current_rect_item = None

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
        self.load_btn = QPushButton("Load Program JSON")
        self.load_btn.clicked.connect(self.load_program_dialog)
        left.addWidget(self.load_btn)

        self.program_label = QLabel("No program loaded")
        left.addWidget(self.program_label)

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
        self.detail_scene = QGraphicsScene()
        self.detail_view = QGraphicsView(self.detail_scene)
        self.detail_view.setRenderHint(QPainter.Antialiasing)
        self.detail_view.setMinimumSize(320, 320)
        right.addWidget(self.detail_view, stretch=1)

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
    def load_program_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Program JSON", self.programs_dir, "JSON files (*.json)"
        )
        if path:
            self.load_program(path)

    def load_program(self, path):
        with open(path) as f:
            self.program = json.load(f)
        self.program_label.setText(
            f"{self.program['name']}  |  {len(self.program['components'])} components"
        )
        self._populate_part_list()
        self._draw_overview()

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
        self._highlight_part_in_overview(part)
        self._draw_detail(part)

    def _draw_detail(self, part):
        self.detail_scene.clear()
        size = self.part_sizes.get(part, {"width_mm": DEFAULT_SIZE_MM, "height_mm": DEFAULT_SIZE_MM})
        w_px = size["width_mm"] * PX_PER_MM
        h_px = size["height_mm"] * PX_PER_MM

        rect = QRectF(-w_px / 2, -h_px / 2, w_px, h_px)
        self.current_rect_item = make_resizable_rect_item(rect, on_resize=self._on_canvas_resize)
        self.detail_scene.addItem(self.current_rect_item)

        cross_pen = QPen(QColor(80, 80, 80))
        self.detail_scene.addLine(-15, 0, 15, 0, cross_pen)
        self.detail_scene.addLine(0, -15, 0, 15, cross_pen)

        self._refit_detail_window(w_px, h_px)

        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(size["width_mm"])
        self.height_spin.setValue(size["height_mm"])
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)

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
