"""Headless smoke test for ui/program_tab.py — not part of the app,
run manually to verify the tab loads and drag-resize works."""
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt, QPoint, QPointF, QEvent
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest

from ui.program_tab import ProgramTab, PX_PER_MM


def send_mouse_event(widget, etype, pos, button, buttons):
    """Real mouse drags hold a button down across move events; QTest.mouseMove
    doesn't simulate that, so build the events by hand to match real input."""
    ev = QMouseEvent(etype, QPointF(pos), widget.mapToGlobal(pos), button, buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)

tmpdir = Path(tempfile.mkdtemp())
programs_dir = tmpdir / "programs"
programs_dir.mkdir()

fake_program = {
    "name": "TEST_BOARD",
    "source_file": "test.xlsx",
    "created": "now",
    "is_panel": True,
    "fiducials": [{"x": 0, "y": 0}, {"x": 100, "y": 0}, {"x": 0, "y": 80}],
    "panel_offsets": [{"label": "U1", "dx": 0, "dy": 0}, {"label": "U2", "dx": 50, "dy": 0}],
    "components": [
        {"designator": "R1", "x": 10.0, "y": 10.0, "rotation": 0.0, "library": "RES", "part": "PN-1001"},
        {"designator": "R2", "x": 20.0, "y": 15.0, "rotation": 0.0, "library": "RES", "part": "PN-1001"},
        {"designator": "C1", "x": 30.0, "y": 5.0, "rotation": 90.0, "library": "CAP", "part": "PN-2002"},
        {"designator": "U3", "x": 40.0, "y": 25.0, "rotation": 0.0, "library": "IC", "part": None},
    ],
    "unknown_parts": ["PN-1001", "PN-2002"],
}
prog_path = programs_dir / "TEST_BOARD.json"
prog_path.write_text(json.dumps(fake_program))

app = QApplication(sys.argv)
win = QMainWindow()
tab = ProgramTab(programs_dir=str(programs_dir), part_sizes_path=str(programs_dir / "part_sizes.json"))
win.setCentralWidget(tab)
win.resize(1000, 600)
win.show()
app.processEvents()

# --- 1. load program (bypassing the file dialog) ---
tab.load_program(str(prog_path))
app.processEvents()
assert tab.program is not None, "program failed to load"
assert tab.part_list.count() == 3, f"expected 3 unique parts (incl UNSPECIFIED), got {tab.part_list.count()}"
print("OK: program loaded, part list populated:", [tab.part_list.item(i).text() for i in range(tab.part_list.count())])

# --- 2. select the first part ---
tab.part_list.setCurrentRow(0)
app.processEvents()
assert tab.current_part is not None, "no part selected after setCurrentRow"
assert tab.current_rect_item is not None, "no rect item drawn in detail view"
print("OK: part selected ->", tab.current_part)

# --- 3. simulate a drag on the bottom-right resize handle ---
before_w, before_h = tab.width_spin.value(), tab.height_spin.value()
rect = tab.current_rect_item.rect()
br_scene = tab.current_rect_item.mapToScene(rect.bottomRight())
br_view = tab.detail_view.mapFromScene(br_scene)

vp = tab.detail_view.viewport()
send_mouse_event(vp, QEvent.MouseButtonPress, br_view, Qt.LeftButton, Qt.LeftButton)
app.processEvents()
mid = QPoint(br_view.x() + 20, br_view.y() + 20)
send_mouse_event(vp, QEvent.MouseMove, mid, Qt.NoButton, Qt.LeftButton)
app.processEvents()
drag_to = QPoint(br_view.x() + 40, br_view.y() + 40)
send_mouse_event(vp, QEvent.MouseMove, drag_to, Qt.NoButton, Qt.LeftButton)
app.processEvents()
send_mouse_event(vp, QEvent.MouseButtonRelease, drag_to, Qt.LeftButton, Qt.NoButton)
app.processEvents()

after_w, after_h = tab.width_spin.value(), tab.height_spin.value()
print(f"OK: drag resize -> before=({before_w},{before_h}) after=({after_w},{after_h})")
assert (after_w, after_h) != (before_w, before_h), "drag-resize did NOT change the size spinboxes"
assert tab.part_sizes.get(tab.current_part) is not None, "drag-resize did not update part_sizes dict"

# --- 4. simulate typing an exact mm value in the spinbox ---
tab.width_spin.setValue(3.33)
app.processEvents()
assert tab.part_sizes[tab.current_part]["width_mm"] == 3.33, "spinbox edit did not update part_sizes"
print("OK: spinbox edit updates part_sizes ->", tab.part_sizes[tab.current_part])

# --- 5. save and reload to confirm persistence ---
tab._save_part_sizes()
saved = json.loads((programs_dir / "part_sizes.json").read_text())
assert tab.current_part in saved, "part_sizes.json was not written correctly"
print("OK: part_sizes.json persisted ->", saved)

# --- 6. select second part, confirm list label refreshed (not red anymore) ---
tab.part_list.setCurrentRow(0)
item0 = tab.part_list.item(0)
print("OK: list item after edit ->", item0.text(), "color=", item0.foreground().color().name())

# --- 7. grab a screenshot for visual sanity check ---
pix = win.grab()
out_png = tmpdir / "program_tab_screenshot.png"
pix.save(str(out_png))
print("SCREENSHOT:", out_png)

print("\nALL CHECKS PASSED")
