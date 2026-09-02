"""
logs_tab.py

The Logs / History tab: reads back the inspection result CSV and shows
it as a filterable table (verdict, barcode/free text), newest first.

Run standalone for testing:
    python ui/logs_tab.py [path/to/results.csv]
"""

import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.result_log import COLUMNS, filter_results, read_results

VERDICT_COLORS = {
    "PASS": QColor(210, 240, 215),
    "FAIL": QColor(250, 214, 210),
    "INCOMPLETE": QColor(250, 238, 205),
}


class LogsTab(QWidget):
    def __init__(self, log_path="logs/results.csv"):
        super().__init__()
        self.log_path = log_path
        self.rows = []
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QVBoxLayout(self)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Verdict:"))
        self.verdict_combo = QComboBox()
        self.verdict_combo.addItems(["All", "PASS", "FAIL", "INCOMPLETE"])
        self.verdict_combo.currentTextChanged.connect(self.apply_filters)
        filters.addWidget(self.verdict_combo)

        filters.addWidget(QLabel("Search:"))
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("barcode, program, designator...")
        self.search_edit.textChanged.connect(self.apply_filters)
        filters.addWidget(self.search_edit, stretch=1)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        filters.addWidget(self.refresh_btn)

        self.open_btn = QPushButton("Open CSV...")
        self.open_btn.clicked.connect(self.choose_csv)
        filters.addWidget(self.open_btn)
        root.addLayout(filters)

        self.table = QTableWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, stretch=1)

        self.status_label = QLabel("")
        root.addWidget(self.status_label)

    # ---------- data ----------
    def refresh(self):
        self.rows = read_results(self.log_path)
        self.apply_filters()

    def choose_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open results CSV", "", "CSV files (*.csv)")
        if path:
            self.log_path = path
            self.refresh()

    def apply_filters(self):
        verdict = self.verdict_combo.currentText()
        rows = filter_results(
            self.rows,
            verdict=None if verdict == "All" else verdict,
            text=self.search_edit.text().strip() or None,
        )
        self._populate(list(reversed(rows)))  # newest first
        total, shown = len(self.rows), len(rows)
        suffix = "" if total == shown else f" (filtered from {total})"
        self.status_label.setText(f"{shown} result(s){suffix} - {self.log_path}")

    def _populate(self, rows):
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            colour = VERDICT_COLORS.get(row.get("verdict"))
            for c, column in enumerate(COLUMNS):
                item = QTableWidgetItem(str(row.get(column, "")))
                if colour:
                    item.setBackground(colour)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()


if __name__ == "__main__":
    import tempfile
    from datetime import datetime, timedelta
    from core.inspection import ComponentResult, InspectionResult, UnitResult
    from core.result_log import append_result

    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setWindowTitle("Logs / History - Standalone Test")

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Seed a throwaway log so the tab has something to show.
        path = str(Path(tempfile.mkdtemp()) / "results.csv")
        base = datetime.now()
        for i, verdict in enumerate(["PASS", "FAIL", "PASS", "INCOMPLETE", "FAIL"]):
            units = [UnitResult(label="U1", components=[
                ComponentResult(designator="R1", part="PN-1", unit="U1", x_mm=1, y_mm=1,
                                present=verdict != "FAIL")])]
            append_result(path, InspectionResult(
                verdict=verdict, units=units, barcode=f"SN-{1000 + i}",
                program_name="DEMO_BOARD", message=f"sample row {i + 1}",
            ), timestamp=base + timedelta(minutes=i))

    tab = LogsTab(log_path=path)
    win.setCentralWidget(tab)
    win.resize(1100, 600)
    win.show()
    sys.exit(app.exec_())
