"""
inspection.py

Turns a captured frame + a calibrated mm->px homography into a
PASS/FAIL verdict: projects every component's ROI box onto the image,
crops it, and runs a presence heuristic to decide whether the part is
populated or the pad is bare.

Panel handling
--------------
A mounter XY export can describe a panel of identical units in one of
two ways, and the sample file available so far does not settle which:

  "expanded"  -- every component row already carries absolute
                 panel coordinates (the file lists all units), and the
                 Pattern Offset rows just mark where each unit sits.
  "replicate" -- the component rows describe ONE unit, and each
                 Pattern Offset is a delta at which that layout repeats.

Getting this wrong silently inspects the wrong places, so it is
detected from the data rather than assumed: if components spread across
several offset origins the file is already expanded, otherwise they
describe one unit and get replicated. A program JSON can pin the answer
explicitly with a "panel_mode" key, which always wins over detection.

Note also that offsets count units *besides* the base layout (a sample
file with 8 offsets covers 9 units), so an implicit origin unit is
added when no offset already sits at (0, 0).
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.calibration import mm_to_px_batch

Point = Tuple[float, float]


@dataclass
class PresenceThresholds:
    """Presence heuristic tuning. These defaults are placeholders --
    they MUST be re-tuned against real captures of a known-good and a
    known-missing board, since the right numbers depend on lighting,
    solder mask colour and component finish."""
    std_min: float = 8.0        # grayscale std dev inside the ROI
    range_min: float = 25.0     # robust intensity range (p95 - p5)
    mode: str = "and"           # "and" = stricter (favours false FAIL over false PASS)


@dataclass
class ComponentResult:
    designator: str
    part: Optional[str]
    unit: str
    x_mm: float
    y_mm: float
    roi_px: Optional[Tuple[float, float, float, float]] = None
    std: float = 0.0
    intensity_range: float = 0.0
    present: bool = False
    status: str = "checked"     # checked | unsized | off_frame

    @property
    def missing(self) -> bool:
        return self.status == "checked" and not self.present


@dataclass
class UnitResult:
    label: str
    components: List[ComponentResult] = field(default_factory=list)

    @property
    def missing(self) -> List[ComponentResult]:
        return [c for c in self.components if c.missing]

    @property
    def unchecked(self) -> List[ComponentResult]:
        return [c for c in self.components if c.status != "checked"]

    @property
    def passed(self) -> bool:
        return not self.missing and not self.unchecked


@dataclass
class InspectionResult:
    verdict: str                # PASS | FAIL | INCOMPLETE
    units: List[UnitResult] = field(default_factory=list)
    barcode: Optional[str] = None
    program_name: str = ""
    panel_mode: str = "single"
    message: str = ""

    @property
    def missing(self) -> List[ComponentResult]:
        return [c for u in self.units for c in u.missing]

    @property
    def unchecked(self) -> List[ComponentResult]:
        return [c for u in self.units for c in u.unchecked]

    @property
    def checked_count(self) -> int:
        return sum(1 for u in self.units for c in u.components if c.status == "checked")


# ---------------------------------------------------------------------
# Panel layout
# ---------------------------------------------------------------------

def panel_unit_origins(program: dict) -> List[Tuple[str, float, float]]:
    """Unit origins as (label, dx, dy). Pattern Offset rows count units
    *besides* the base layout, so an origin at (0, 0) is prepended when
    no offset already sits there -- otherwise a panel of N offsets would
    be inspected as N units instead of N+1."""
    offsets = program.get("panel_offsets") or []
    if not offsets:
        return [("U1", 0.0, 0.0)]

    origins = []
    has_base = any(abs(o.get("dx", 0.0)) < 1e-6 and abs(o.get("dy", 0.0)) < 1e-6 for o in offsets)
    if not has_base:
        origins.append(("U1", 0.0, 0.0))
    for i, o in enumerate(offsets):
        label = o.get("label") or f"U{len(origins) + 1}"
        origins.append((str(label), float(o.get("dx", 0.0)), float(o.get("dy", 0.0))))
    return origins


def detect_panel_mode(program: dict) -> str:
    """Return "single", "expanded" or "replicate" -- see module docstring.
    An explicit "panel_mode" key in the program always wins."""
    explicit = program.get("panel_mode")
    if explicit in ("expanded", "replicate", "single"):
        return explicit

    offsets = program.get("panel_offsets") or []
    components = program.get("components") or []
    if not offsets or not components:
        return "single"

    # Must include the implicit base origin: a single unit's components
    # sit near (0, 0), and without that origin to claim them they scatter
    # across whichever offsets happen to be nearest, faking a spread.
    unit_origins = panel_unit_origins(program)
    origins = np.array([[dx, dy] for _label, dx, dy in unit_origins], dtype=np.float64)
    pts = np.array([[c["x"], c["y"]] for c in components], dtype=np.float64)
    nearest = np.argmin(np.linalg.norm(pts[:, None, :] - origins[None, :, :], axis=2), axis=1)
    occupied = len(set(nearest.tolist()))

    # Components clustered at a single origin describe one unit;
    # components spread across the origins are already expanded.
    threshold = max(2, math.ceil(len(unit_origins) / 2))
    return "expanded" if occupied >= threshold else "replicate"


def expand_components(program: dict, mode: Optional[str] = None) -> List[dict]:
    """Flatten a program into the component instances to inspect, each
    tagged with its panel unit and carrying absolute board mm coords."""
    mode = mode or detect_panel_mode(program)
    components = program.get("components") or []

    if mode == "single":
        return [dict(c, unit="U1") for c in components]

    origins = panel_unit_origins(program)

    if mode == "replicate":
        out = []
        for label, dx, dy in origins:
            for c in components:
                out.append(dict(c, unit=label, x=c["x"] + dx, y=c["y"] + dy))
        return out

    # expanded: rows already absolute -- group them by nearest unit origin
    origin_arr = np.array([[dx, dy] for _label, dx, dy in origins], dtype=np.float64)
    pts = np.array([[c["x"], c["y"]] for c in components], dtype=np.float64)
    nearest = np.argmin(np.linalg.norm(pts[:, None, :] - origin_arr[None, :, :], axis=2), axis=1)
    return [dict(c, unit=origins[int(n)][0]) for c, n in zip(components, nearest)]


def expanded_fiducials_mm(program: dict, mode: Optional[str] = None) -> List[Point]:
    """Fiducial mm positions under the same panel interpretation used for
    components, so calibration and inspection can never disagree about
    what the coordinates mean."""
    mode = mode or detect_panel_mode(program)
    fiducials = [(float(f["x"]), float(f["y"])) for f in (program.get("fiducials") or [])]
    if mode != "replicate":
        return fiducials
    out = []
    for _label, dx, dy in panel_unit_origins(program):
        out.extend([(x + dx, y + dy) for x, y in fiducials])
    return out


# ---------------------------------------------------------------------
# ROI projection and presence check
# ---------------------------------------------------------------------

def project_roi(H, x_mm: float, y_mm: float, w_mm: float, h_mm: float,
                rotation_deg: float = 0.0) -> Tuple[float, float, float, float]:
    """Project a component's ROI box onto the image, returning the
    axis-aligned pixel bounding box (x, y, w, h). The box is rotated in
    board space first, so a part placed at 90 degrees gets the right
    footprint rather than its width and height swapped by accident."""
    hw, hh = w_mm / 2.0, h_mm / 2.0
    corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)
    t = math.radians(rotation_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], dtype=np.float64)
    corners = corners @ rot.T + np.array([x_mm, y_mm], dtype=np.float64)

    px = mm_to_px_batch(H, corners)
    x0, y0 = px.min(axis=0)
    x1, y1 = px.max(axis=0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def crop_roi(gray: np.ndarray, roi: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """Crop an ROI, clipped to the frame. Returns None when the box
    falls (mostly) outside the image -- an off-frame component must be
    reported as unchecked, never silently treated as present."""
    h_img, w_img = gray.shape[:2]
    x, y, w, h = roi
    x0, y0 = int(round(x)), int(round(y))
    x1, y1 = int(round(x + w)), int(round(y + h))
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w_img, x1), min(h_img, y1)
    if cx1 - cx0 < 2 or cy1 - cy0 < 2:
        return None
    return gray[cy0:cy1, cx0:cx1]


def check_presence(crop: np.ndarray, thresholds: PresenceThresholds) -> Tuple[bool, float, float]:
    """Heuristic populated-vs-bare decision for one ROI crop: a
    populated pad carries a component body and its edges, so it varies;
    a bare pad is comparatively flat. Returns (present, std, range)."""
    std = float(np.std(crop))
    p5, p95 = np.percentile(crop, [5, 95])
    rng = float(p95 - p5)

    std_ok = std >= thresholds.std_min
    range_ok = rng >= thresholds.range_min
    present = (std_ok and range_ok) if thresholds.mode == "and" else (std_ok or range_ok)
    return present, std, rng


# ---------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------

def to_gray(frame: np.ndarray) -> np.ndarray:
    return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def inspect(
    frame: np.ndarray,
    program: dict,
    part_sizes: Dict[str, dict],
    homography: np.ndarray,
    thresholds: Optional[PresenceThresholds] = None,
    panel_mode: Optional[str] = None,
    barcode: Optional[str] = None,
) -> InspectionResult:
    """Run one inspection pass over a captured frame.

    Components whose part number has no ROI size yet, and components
    whose ROI falls outside the frame, are reported as unchecked rather
    than assumed good -- a board with unchecked components can never
    come back a clean PASS.
    """
    thresholds = thresholds or PresenceThresholds()
    gray = to_gray(frame)
    mode = panel_mode or detect_panel_mode(program)
    items = expand_components(program, mode)

    units: Dict[str, UnitResult] = {}
    for item in items:
        unit_label = item.get("unit", "U1")
        unit = units.setdefault(unit_label, UnitResult(label=unit_label))

        part = item.get("part")
        result = ComponentResult(
            designator=item.get("designator", "?"),
            part=part,
            unit=unit_label,
            x_mm=float(item["x"]),
            y_mm=float(item["y"]),
        )

        size = part_sizes.get(part) if part else None
        if not size:
            result.status = "unsized"
            unit.components.append(result)
            continue

        roi = project_roi(homography, result.x_mm, result.y_mm,
                          float(size["width_mm"]), float(size["height_mm"]),
                          float(item.get("rotation", 0.0) or 0.0))
        result.roi_px = roi

        crop = crop_roi(gray, roi)
        if crop is None:
            result.status = "off_frame"
            unit.components.append(result)
            continue

        present, std, rng = check_presence(crop, thresholds)
        result.present, result.std, result.intensity_range = present, std, rng
        unit.components.append(result)

    ordered = [units[k] for k in sorted(units.keys())]
    missing = [c for u in ordered for c in u.missing]
    unchecked = [c for u in ordered for c in u.unchecked]

    if missing:
        verdict = "FAIL"
        message = f"{len(missing)} missing component(s) across {sum(1 for u in ordered if u.missing)} unit(s)."
    elif unchecked:
        verdict = "INCOMPLETE"
        message = (f"No missing components found, but {len(unchecked)} component(s) could not be "
                   f"checked (unsized part or ROI off frame).")
    else:
        verdict = "PASS"
        message = f"All {sum(len(u.components) for u in ordered)} components present."

    return InspectionResult(
        verdict=verdict, units=ordered, barcode=barcode,
        program_name=program.get("name", ""), panel_mode=mode, message=message,
    )
