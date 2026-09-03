"""
fiducials.py

AOI-style fiducial alignment: instead of hunting for any circular blob
and guessing which is which, the operator names the alignment points
(F1, F2, F3) once in Program Manager, a template of each is taught from
the reference board photo, and every inspection finds those specific
marks by appearance and checks the triangle they form.

Why this replaces blob detection as the primary path:

  * No correspondence guessing. Each template belongs to one named
    fiducial, so there is nothing to assign.
  * No panel aliasing. Blob matching on a panel whose fiducial pattern
    repeats at a regular pitch can align one pitch over and look
    equally good. Three specific marks with known mm spacing cannot:
    the triangle's side lengths only fit one way.
  * It sees the real mark. A taught template matches the fiducial's
    actual appearance rather than assuming "round and circular".

Three points is the useful maximum for a flat board -- it pins
translation, rotation and scale, and gives a redundant third side to
verify against. Two are supported (similarity only, no triangle to
check). The blob detector in calibration.py stays as the fallback for
programs with no reference image yet, and manual click-to-align stays
as the final fallback.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from core.calibration import CalibrationResult, fit_homography, mm_to_px_batch

Point = Tuple[float, float]

DEFAULT_IDS = ("F1", "F2", "F3")
DEFAULT_PATCH_MM = 6.0
DEFAULT_SCALES = (0.88, 0.94, 1.0, 1.06, 1.13)


@dataclass
class FiducialRef:
    """One named alignment point, in board millimetres."""
    id: str
    x_mm: float
    y_mm: float

    def as_dict(self) -> dict:
        return {"id": self.id, "x": self.x_mm, "y": self.y_mm}

    @staticmethod
    def from_dict(d: dict) -> "FiducialRef":
        return FiducialRef(str(d["id"]), float(d["x"]), float(d["y"]))


@dataclass
class FiducialMatch:
    id: str
    x_px: float
    y_px: float
    score: float
    scale: float


# ---------------------------------------------------------------------
# Program storage
# ---------------------------------------------------------------------

def get_fiducial_refs(program: dict) -> List[FiducialRef]:
    """The operator-chosen alignment points, if any have been set."""
    return [FiducialRef.from_dict(d) for d in (program.get("fiducial_refs") or [])]


def set_fiducial_refs(program: dict, refs: List[FiducialRef]) -> None:
    program["fiducial_refs"] = [r.as_dict() for r in refs]


def suggest_fiducial_refs(program: dict, count: int = 3) -> List[FiducialRef]:
    """Pick a sensible default trio from the XY file's Pattern Fiducial
    rows: the most spread-out, least collinear set available, since a
    long thin triangle pins rotation poorly.

    Only a suggestion -- the operator confirms or replaces it.
    """
    points = [(float(f["x"]), float(f["y"])) for f in (program.get("fiducials") or [])]
    if len(points) <= count:
        return [FiducialRef(DEFAULT_IDS[i] if i < len(DEFAULT_IDS) else f"F{i + 1}", x, y)
                for i, (x, y) in enumerate(points)]

    # Farthest-point seeding gives good spread cheaply, then a triangle
    # area check rejects a near-collinear pick.
    pts = np.asarray(points, dtype=np.float64)
    centre = pts.mean(axis=0)
    chosen = [int(np.argmax(np.linalg.norm(pts - centre, axis=1)))]
    while len(chosen) < min(count, len(pts)):
        d = np.min(np.linalg.norm(pts[:, None, :] - pts[chosen][None, :, :], axis=2), axis=1)
        d[chosen] = -1.0
        chosen.append(int(np.argmax(d)))

    picked = pts[chosen]
    if len(picked) == 3 and _triangle_area(picked) < 1.0:
        # Degenerate suggestion: fall back to the widest pair plus the
        # point furthest off that line.
        best, best_area = None, -1.0
        for i in range(len(pts)):
            area = _triangle_area(np.vstack([picked[:2], pts[i]]))
            if area > best_area:
                best, best_area = i, area
        if best is not None:
            chosen[2] = best

    return [FiducialRef(DEFAULT_IDS[i], float(pts[c][0]), float(pts[c][1]))
            for i, c in enumerate(chosen)]


def _triangle_area(pts: np.ndarray) -> float:
    (x1, y1), (x2, y2), (x3, y3) = pts[:3]
    return abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)) / 2.0


# ---------------------------------------------------------------------
# Template teaching
# ---------------------------------------------------------------------

def local_px_per_mm(homography: np.ndarray, x_mm: float, y_mm: float) -> float:
    """Image pixels per board millimetre near a point, read off the
    homography itself so it stays right under perspective."""
    probe = np.array([[x_mm, y_mm], [x_mm + 1.0, y_mm], [x_mm, y_mm + 1.0]], dtype=np.float64)
    p = mm_to_px_batch(homography, probe)
    dx = float(np.linalg.norm(p[1] - p[0]))
    dy = float(np.linalg.norm(p[2] - p[0]))
    return max((dx + dy) / 2.0, 1e-6)


def teach_templates(reference_image: np.ndarray, homography: np.ndarray,
                    refs: List[FiducialRef], patch_mm: float = DEFAULT_PATCH_MM
                    ) -> Dict[str, np.ndarray]:
    """Cut a template around each named fiducial from the aligned
    reference photo. Patch size is given in millimetres so every
    fiducial gets the same physical area whatever the camera scale."""
    templates = {}
    if reference_image is None or homography is None:
        return templates

    gray = reference_image if reference_image.ndim == 2 else cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape[:2]

    for ref in refs:
        scale = local_px_per_mm(homography, ref.x_mm, ref.y_mm)
        half = max(6, int(round(patch_mm * scale / 2.0)))
        centre = mm_to_px_batch(homography, np.array([[ref.x_mm, ref.y_mm]]))[0]
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        x1, y1 = min(w_img, cx + half), min(h_img, cy + half)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue   # fiducial sits outside the reference frame
        templates[ref.id] = gray[y0:y1, x0:x1].copy()
    return templates


def save_templates(templates: Dict[str, np.ndarray], program_name: str, programs_dir: str) -> List[str]:
    base = Path(programs_dir)
    base.mkdir(parents=True, exist_ok=True)
    written = []
    for fid, patch in templates.items():
        path = base / f"{program_name}.fid_{fid}.png"
        cv2.imwrite(str(path), patch)
        written.append(str(path))
    return written


def load_templates(program_name: str, programs_dir: str,
                   ids=DEFAULT_IDS) -> Dict[str, np.ndarray]:
    base = Path(programs_dir)
    templates = {}
    for fid in ids:
        path = base / f"{program_name}.fid_{fid}.png"
        if path.exists():
            patch = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if patch is not None:
                templates[fid] = patch
    return templates


def delete_templates(program_name: str, programs_dir: str, ids=DEFAULT_IDS) -> None:
    for fid in ids:
        Path(programs_dir, f"{program_name}.fid_{fid}.png").unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------

def _subpixel_offset(response: np.ndarray, x: int, y: int) -> Tuple[float, float]:
    """Sub-pixel correction of a correlation peak by fitting a parabola
    through its immediate neighbours on each axis.

    The correlation surface around a true match is smooth and roughly
    quadratic, so the vertex of that parabola locates the peak far more
    precisely than the integer cell that merely contains it.
    """
    h, w = response.shape[:2]
    if not (0 < x < w - 1 and 0 < y < h - 1):
        return 0.0, 0.0

    def vertex(before: float, centre: float, after: float) -> float:
        denominator = before - 2.0 * centre + after
        if abs(denominator) < 1e-12:
            return 0.0
        # Clamped: a flat or noisy surface can otherwise throw the vertex
        # outside the cell it was found in, which is never a refinement.
        return float(np.clip(0.5 * (before - after) / denominator, -0.5, 0.5))

    centre = float(response[y, x])
    dx = vertex(float(response[y, x - 1]), centre, float(response[y, x + 1]))
    dy = vertex(float(response[y - 1, x]), centre, float(response[y + 1, x]))
    return dx, dy


def match_template_peaks(gray: np.ndarray, template: np.ndarray, top_k: int = 6,
                         scales=DEFAULT_SCALES, min_score: float = 0.45
                         ) -> List[Tuple[float, float, float, float]]:
    """Best (x, y, score, scale) peaks for one template.

    Several scales are tried because the live working distance can
    differ slightly from the reference shot. Peaks are suppressed within
    roughly a template width of each other so one strong match doesn't
    return as a cluster of near-duplicates.

    Peak positions are refined to sub-pixel precision, which matters:
    the whole board's alignment is fitted from these few points, so a
    half-pixel bias at a fiducial becomes a bias at every ROI across the
    board -- and at 35 px/mm, one pixel is already 0.03mm of drift.
    """
    peaks: List[Tuple[float, float, float, float]] = []
    for scale in scales:
        tw = int(round(template.shape[1] * scale))
        th = int(round(template.shape[0] * scale))
        if tw < 6 or th < 6 or tw >= gray.shape[1] or th >= gray.shape[0]:
            continue
        resized = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        work = response.copy()
        for _ in range(top_k):
            _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(work)
            if max_v < min_score:
                break
            dx, dy = _subpixel_offset(response, max_l[0], max_l[1])
            cx = max_l[0] + dx + tw / 2.0
            cy = max_l[1] + dy + th / 2.0
            peaks.append((float(cx), float(cy), float(max_v), float(scale)))
            x0 = max(0, max_l[0] - tw // 2)
            y0 = max(0, max_l[1] - th // 2)
            x1 = min(work.shape[1], max_l[0] + tw // 2)
            y1 = min(work.shape[0], max_l[1] + th // 2)
            work[y0:y1, x0:x1] = -1.0

    # merge peaks from different scales that land on the same spot
    peaks.sort(key=lambda p: -p[2])
    merged: List[Tuple[float, float, float, float]] = []
    radius = max(template.shape) / 2.0
    for x, y, score, scale in peaks:
        if any(math.hypot(x - mx, y - my) < radius for mx, my, _s, _sc in merged):
            continue
        merged.append((x, y, score, scale))
        if len(merged) >= top_k:
            break
    return merged


def _best_consistent_set(refs: List[FiducialRef],
                         candidates: Dict[str, List[Tuple[float, float, float, float]]],
                         scale_tolerance: float = 0.08):
    """Choose one candidate per fiducial so the shape they form matches
    the known millimetre geometry.

    This is the step that makes the result trustworthy on a repetitive
    panel: a wrong-but-plausible match one panel pitch away changes the
    side lengths, so it cannot satisfy a single consistent mm-to-pixel
    scale across all three pairs.
    """
    ids = [r.id for r in refs]
    if any(not candidates.get(i) for i in ids):
        return None, 0.0

    mm = {r.id: np.array([r.x_mm, r.y_mm], dtype=np.float64) for r in refs}
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
    mm_dist = {(a, b): float(np.linalg.norm(mm[a] - mm[b])) for a, b in pairs}
    if any(d < 1e-6 for d in mm_dist.values()):
        return None, 0.0

    best, best_score = None, -1.0
    # Candidate lists are short (top_k per fiducial), so the exhaustive
    # product is cheap and avoids missing the right combination.
    def walk(index, chosen):
        nonlocal best, best_score
        if index == len(ids):
            scales = []
            for a, b in pairs:
                pa, pb = np.array(chosen[a][:2]), np.array(chosen[b][:2])
                scales.append(float(np.linalg.norm(pa - pb)) / mm_dist[(a, b)])
            if min(scales) <= 0:
                return
            spread = max(scales) / min(scales)
            if spread > 1.0 + scale_tolerance:
                return
            score = sum(chosen[i][2] for i in ids) - (spread - 1.0) * 2.0
            if score > best_score:
                best, best_score = dict(chosen), score
            return
        for cand in candidates[ids[index]]:
            chosen[ids[index]] = cand
            walk(index + 1, chosen)
        chosen.pop(ids[index], None)

    walk(0, {})
    return best, best_score


def align_with_templates(frame: np.ndarray, refs: List[FiducialRef],
                         templates: Dict[str, np.ndarray], top_k: int = 6,
                         min_score: float = 0.45, scale_tolerance: float = 0.08
                         ) -> CalibrationResult:
    """Locate the taught fiducials in a frame and fit the mm->px
    homography from them."""
    usable = [r for r in refs if r.id in templates]
    if len(usable) < 2:
        return CalibrationResult(
            success=False, method="template",
            message=f"Only {len(usable)} taught fiducial template(s); need at least 2.",
        )

    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    candidates = {r.id: match_template_peaks(gray, templates[r.id], top_k=top_k,
                                              min_score=min_score)
                  for r in usable}

    missing = [r.id for r in usable if not candidates[r.id]]
    if missing:
        return CalibrationResult(
            success=False, method="template",
            message=(f"Could not find fiducial(s) {', '.join(missing)} in the frame. "
                      "Check the board is fully in view and lit as it was when taught."),
        )

    chosen, _score = _best_consistent_set(usable, candidates, scale_tolerance)
    if chosen is None:
        return CalibrationResult(
            success=False, method="template",
            message=("Found candidate marks but they do not form the taught fiducial "
                      "geometry. The board may be the wrong program, or partly out of view."),
        )

    matched_mm = [(r.x_mm, r.y_mm) for r in usable]
    matched_px = [(chosen[r.id][0], chosen[r.id][1]) for r in usable]
    H, rms = fit_homography(matched_mm, matched_px)
    if H is None:
        return CalibrationResult(success=False, method="template",
                                  message="Homography fit failed on the located fiducials.")

    worst = min(chosen[r.id][2] for r in usable)
    quality = (f"RMS {rms:.2f}px" if len(usable) > 3
               else f"exact {len(usable)}-point fit")
    return CalibrationResult(
        success=True, method="template", homography=H,
        matched_mm=matched_mm, matched_px=matched_px,
        inlier_count=len(usable), rms_error_px=rms, match_score=worst,
        message=(f"Aligned on {len(usable)} taught fiducials "
                  f"({', '.join(r.id for r in usable)}), worst match {worst:.2f}, {quality}."),
    )


def matches_for_display(refs: List[FiducialRef], result: CalibrationResult) -> List[FiducialMatch]:
    """Pair up ids with located pixel positions, for drawing overlays."""
    out = []
    for ref, px in zip(refs, result.matched_px or []):
        out.append(FiducialMatch(ref.id, float(px[0]), float(px[1]), 1.0, 1.0))
    return out
