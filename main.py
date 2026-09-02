"""
main.py

PCB visual inspection station -- entry point.

Three tabs:
  Live Inspection  -- camera view, calibrate, trigger a pass, PASS/FAIL
  Program Manager  -- import mounter XY files, set part ROI sizes,
                      choose the active program
  Logs / History   -- past results from the CSV

Run:
    python main.py
"""

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ui.live_tab import LiveTab
from ui.logs_tab import LogsTab
from ui.program_tab import ProgramTab
from ui.theme import apply_theme, style_tabs

APP_DIR = Path(__file__).resolve().parent
PROGRAMS_DIR = APP_DIR / "programs"
PART_SIZES_PATH = PROGRAMS_DIR / "part_sizes.json"
PART_THRESHOLDS_PATH = PROGRAMS_DIR / "part_thresholds.json"
LOG_PATH = APP_DIR / "logs" / "results.csv"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Inspection Station")
        self.resize(1300, 820)

        self.tabs = QTabWidget()
        self.live_tab = LiveTab(log_path=str(LOG_PATH), programs_dir=str(PROGRAMS_DIR),
                                part_thresholds_path=str(PART_THRESHOLDS_PATH))
        self.program_tab = ProgramTab(programs_dir=str(PROGRAMS_DIR),
                                      part_sizes_path=str(PART_SIZES_PATH))
        self.logs_tab = LogsTab(log_path=str(LOG_PATH))

        self.tabs.addTab(self.live_tab, "Live Inspection")
        self.tabs.addTab(self.program_tab, "Program Manager")
        self.tabs.addTab(self.logs_tab, "Logs / History")
        style_tabs(self.tabs)
        self.setCentralWidget(self.tabs)

        # Program Manager owns which board is active; Live Inspection follows it.
        self.program_tab.program_activated.connect(self._on_program_activated)
        # A completed pass refreshes history, so the log tab is never stale.
        self.live_tab.inspected.connect(lambda _result: self.logs_tab.refresh())

        self.statusBar().showMessage(
            "Import or load a program in Program Manager, set it active, then align the board and inspect."
        )

    def _on_program_activated(self, program, part_sizes):
        self.live_tab.set_program(program, part_sizes)
        self.tabs.setCurrentWidget(self.live_tab)
        self.statusBar().showMessage(
            f"Active program: {program.get('name', '?')} - press Align Board, then INSPECT (Space)."
        )

    def closeEvent(self, event):
        self.live_tab.stop_live()
        super().closeEvent(event)


def main():
    app = apply_theme(QApplication(sys.argv))
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
