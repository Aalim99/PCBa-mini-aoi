"""
theme.py

Visual language for the station: a dark, low-glare palette suited to a
bench under work lighting, where the operator reads a verdict across the
bench rather than a paragraph up close.

Rules the rest of the UI follows:

  * One accent colour. Blue means interactive; it is never used for
    status, so a green/amber/red only ever means a verdict.
  * Verdicts carry weight. PASS/FAIL/INCOMPLETE get size and a solid
    fill, because that is the one thing read at a distance.
  * Panels are cards on a darker ground, so the eye can find the three
    regions (view, controls, results) without reading labels.

Widgets opt into variants with Qt dynamic properties, e.g.
`setProperty("variant", "primary")` on a button, or
`setProperty("verdict", "FAIL")` on the banner -- restyle after
changing one (see `restyle`).
"""

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QPalette
from PyQt5.QtWidgets import QFrame, QLabel, QVBoxLayout

COLORS = {
    "bg": "#12151a",
    "surface": "#1a1f27",
    "raised": "#222833",
    "border": "#2d3542",
    "border_strong": "#3a4453",
    "text": "#e7ecf3",
    "muted": "#98a3b3",
    "faint": "#6b7686",
    "accent": "#3d8bfd",
    "accent_hover": "#5599ff",
    "accent_press": "#2f74d9",
    "pass": "#2ea36b",
    "fail": "#e5484d",
    "warn": "#eaa53a",
    "pass_soft": "#173a2a",
    "fail_soft": "#3d1b1e",
    "warn_soft": "#3a2e14",
}

VERDICT_COLORS = {
    "PASS": (COLORS["pass"], COLORS["pass_soft"]),
    "FAIL": (COLORS["fail"], COLORS["fail_soft"]),
    "INCOMPLETE": (COLORS["warn"], COLORS["warn_soft"]),
}

STYLESHEET = f"""
QWidget {{
    background: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: "Segoe UI", "Inter", "Ubuntu", "DejaVu Sans", sans-serif;
    font-size: 13px;
}}

QLabel {{ background: transparent; }}
QLabel[variant="title"]    {{ font-size: 17px; font-weight: 600; }}
QLabel[variant="section"]  {{ font-size: 11px; font-weight: 700; color: {COLORS['faint']};
                              letter-spacing: 1px; }}
QLabel[variant="muted"]    {{ color: {COLORS['muted']}; }}
QLabel[variant="metric"]   {{ font-size: 20px; font-weight: 600; }}

QFrame[variant="card"] {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 10px;
}}
QFrame[variant="divider"] {{ background: {COLORS['border']}; max-height: 1px; border: none; }}

QPushButton {{
    background: {COLORS['raised']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 7px;
    padding: 7px 14px;
    color: {COLORS['text']};
}}
QPushButton:hover  {{ background: #29313e; border-color: {COLORS['accent']}; }}
QPushButton:pressed {{ background: #1d232c; }}
QPushButton:disabled {{ color: {COLORS['faint']}; border-color: {COLORS['border']};
                        background: #191d24; }}

QPushButton[variant="primary"] {{
    background: {COLORS['accent']}; border: 1px solid {COLORS['accent']};
    color: #ffffff; font-weight: 600;
}}
QPushButton[variant="primary"]:hover  {{ background: {COLORS['accent_hover']}; }}
QPushButton[variant="primary"]:pressed {{ background: {COLORS['accent_press']}; }}
QPushButton[variant="primary"]:disabled {{ background: #24303f; border-color: #24303f;
                                           color: {COLORS['faint']}; }}

QPushButton[variant="trigger"] {{
    background: {COLORS['accent']}; border: none; color: #ffffff;
    font-size: 15px; font-weight: 700; padding: 12px 20px; border-radius: 8px;
}}
QPushButton[variant="trigger"]:hover  {{ background: {COLORS['accent_hover']}; }}
QPushButton[variant="trigger"]:disabled {{ background: #24303f; color: {COLORS['faint']}; }}

QPushButton[variant="danger"] {{ border-color: #5d2b2e; color: #ff9a9d; }}
QPushButton[variant="danger"]:hover {{ background: {COLORS['fail_soft']};
                                       border-color: {COLORS['fail']}; }}

QPushButton[variant="ghost"] {{ background: transparent; border: 1px solid {COLORS['border']}; }}
QPushButton[variant="ghost"]:hover {{ background: {COLORS['raised']}; }}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
    background: {COLORS['raised']};
    border: 1px solid {COLORS['border_strong']};
    border-radius: 7px;
    padding: 6px 9px;
    selection-background-color: {COLORS['accent']};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {COLORS['accent']};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox::down-arrow {{
    image: none; border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {COLORS['muted']}; margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background: {COLORS['raised']}; border: 1px solid {COLORS['border_strong']};
    selection-background-color: {COLORS['accent']}; outline: none;
}}
QDoubleSpinBox::up-button, QSpinBox::up-button,
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    background: {COLORS['border']}; border: none; width: 15px;
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{ background: {COLORS['accent']}; }}

QListWidget, QTableWidget, QTreeWidget, QGraphicsView, QTextEdit {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    outline: none;
}}
QListWidget::item {{ padding: 6px 8px; border-radius: 5px; }}
QListWidget::item:selected {{ background: {COLORS['accent']}; color: #ffffff; }}
QListWidget::item:hover:!selected {{ background: {COLORS['raised']}; }}

QTableWidget {{ gridline-color: {COLORS['border']}; }}
QTableWidget::item {{ padding: 5px 7px; }}
QTableWidget::item:selected {{ background: {COLORS['accent']}; color: #ffffff; }}
QHeaderView::section {{
    background: {COLORS['raised']}; color: {COLORS['muted']};
    border: none; border-right: 1px solid {COLORS['border']};
    border-bottom: 1px solid {COLORS['border']};
    padding: 7px 8px; font-weight: 600;
}}
QTableCornerButton::section {{ background: {COLORS['raised']}; border: none; }}

QGroupBox {{
    border: 1px solid {COLORS['border']}; border-radius: 9px;
    margin-top: 16px; padding-top: 10px; background: {COLORS['surface']};
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 5px;
    color: {COLORS['faint']}; font-size: 11px; font-weight: 700; letter-spacing: 1px;
}}

QTabWidget::pane {{ border: 1px solid {COLORS['border']}; border-radius: 10px;
                    background: {COLORS['bg']}; top: -1px; }}
/* No font-weight here: Qt measures the tab with the widget font, so a
   bolder face set in the stylesheet overflows the width it reserved and
   clips the label. style_tabs() sets a bold font on the tab bar itself,
   which the measurement then accounts for. */
QTabBar::tab {{
    background: transparent; color: {COLORS['muted']};
    padding: 9px 20px; margin-right: 3px;
    border: 1px solid transparent; border-top-left-radius: 8px; border-top-right-radius: 8px;
}}
QTabBar::tab:hover {{ color: {COLORS['text']}; background: {COLORS['surface']}; }}
QTabBar::tab:selected {{
    background: {COLORS['surface']}; color: {COLORS['text']};
    border-color: {COLORS['border']}; border-bottom-color: {COLORS['surface']};
}}

QSlider::groove:horizontal {{
    height: 5px; background: {COLORS['border']}; border-radius: 3px;
}}
QSlider::sub-page:horizontal {{ background: {COLORS['accent']}; border-radius: 3px; }}
QSlider::handle:horizontal {{
    background: #ffffff; border: none; width: 16px; height: 16px;
    margin: -6px 0; border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{ background: {COLORS['accent_hover']}; }}

QScrollBar:vertical {{ background: transparent; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {COLORS['border_strong']}; border-radius: 5px;
                               min-height: 28px; }}
QScrollBar::handle:vertical:hover {{ background: {COLORS['faint']}; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {COLORS['border_strong']}; border-radius: 5px;
                                 min-width: 28px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {COLORS['surface']}; color: {COLORS['muted']};
              border-top: 1px solid {COLORS['border']}; }}
QToolTip {{ background: {COLORS['raised']}; color: {COLORS['text']};
            border: 1px solid {COLORS['border_strong']}; padding: 5px; border-radius: 5px; }}
QMessageBox, QDialog {{ background: {COLORS['surface']}; }}
QSplitter::handle {{ background: {COLORS['border']}; }}
QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px;
                        border: 1px solid {COLORS['border_strong']}; background: {COLORS['raised']}; }}
QCheckBox::indicator:checked {{ background: {COLORS['accent']}; border-color: {COLORS['accent']}; }}
"""


def apply_theme(app):
    """Apply the palette and stylesheet to the whole application."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["bg"]))
    palette.setColor(QPalette.Base, QColor(COLORS["surface"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["raised"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(COLORS["raised"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS["raised"]))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS["text"]))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)
    return app


def style_tabs(tab_widget):
    """Give a QTabWidget its bold tab labels without clipping them.

    The weight is set on the tab bar's own font rather than in the
    stylesheet so that Qt's tab measurement uses the same face it draws
    with; otherwise the label is measured light, drawn bold, and loses a
    character at each end.
    """
    bar = tab_widget.tabBar()
    font = bar.font()
    font.setBold(True)
    bar.setFont(font)
    bar.setElideMode(Qt.ElideNone)
    bar.setExpanding(False)
    return tab_widget


def restyle(widget):
    """Re-apply the stylesheet to one widget after changing a dynamic
    property -- Qt does not repolish automatically."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)
    widget.update()


# ---------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------

class Card(QFrame):
    """A titled panel. Content goes in `body`."""

    def __init__(self, title=None, parent=None):
        super().__init__(parent)
        self.setProperty("variant", "card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(9)
        if title:
            label = QLabel(title.upper())
            label.setProperty("variant", "section")
            outer.addWidget(label)
            self.title_label = label
        else:
            self.title_label = None
        self.body = QVBoxLayout()
        self.body.setSpacing(8)
        outer.addLayout(self.body)


def section_label(text):
    label = QLabel(text.upper())
    label.setProperty("variant", "section")
    return label


def muted_label(text=""):
    label = QLabel(text)
    label.setProperty("variant", "muted")
    label.setWordWrap(True)
    return label


def title_label(text):
    label = QLabel(text)
    label.setProperty("variant", "title")
    return label


def verdict_style(verdict):
    """(border/text colour, fill colour) for a verdict banner."""
    return VERDICT_COLORS.get(verdict, (COLORS["faint"], COLORS["raised"]))
