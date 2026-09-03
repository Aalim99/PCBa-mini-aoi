"""
calibration.py

Computes the mm -> pixel homography that aligns a PCB program's known
fiducial positions (from the parsed XY file) with a captured camera
frame, so component ROI positions (in board mm) can later be projected
onto the live image.

Two paths:
  - auto_calibrate(): detects circular fiducial pads in the frame via
    contour/circularity filtering, then solves the (initially unknown)
    correspondence between detected pixel positions and known mm
    positions using a RANSAC-style hypothesize-and-verify search over
    similarity transforms (scale + rotation + translation), and
    finally refits a full homography with cv2.findHomography on the
    inlier correspondences.
  - manual_calibrate(): given an ordered list of mm points and the
    matching ordered list of operator-clicked pixel points (no
    correspondence search needed since the operator clicks fiducials
    in the same order they're listed), fits the transform directly.

Known limitation: auto-detect correspondence matching can be ambiguous
on panels where many fiducials repeat at a regular pitch (e.g. 2
per-unit fiducials x 9 identical units) -- a translated/rotated guess
can look equally valid as the true one. auto_calibrate() detects this
by checking whether a meaningfully different assignment scores nearly
as well as the best one, and refuses to return success=True in that
case (an ambiguous-but-lucky match is still reported as a failure --
see the `ambiguous` field for a ready-to-use ambiguity flag).
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

# Blob search is done at no more than this on the long edge. A fiducial
# pad spans many pixels at any usable working distance, so the extra
# resolution only buys contour noise and seconds of runtime.
DETECT_MAX_DIMENSION = 2000


@dataclass
class CalibrationResult:
    success: bool
    homography: Optional[np.ndarray] = None
    method: str = ""
    matched_mm: List[Point] = field(default_factory=list)
    matched_px: List[Point] = field(default_factory=list)
    inlier_count: int = 0
    rms_error_px: float = float("inf")
    ambiguous: bool = False
    message: str = ""
    # Worst per-fiducial template match, when alignment came from taught
    # templates. 0.0 when that path was not used.
    match_score: float = 0.0

    @property
    def rms_is_meaningful(self) -> bool:
        """False when the fit is exactly determined by its points.

        A 2- or 3-point fit passes through every point by construction,
        so its RMS is always 0.00 and says nothing about how well the
        board was found -- reporting it as accuracy would be false
        confidence. With 3 fiducials, the match score is the honest
        quality signal.
        """
        return self.inlier_count > 3


# ---------------------------------------------------------------------
# 1. Fiducial candidate detection
# ---------------------------------------------------------------------

def detect_fiducial_candidates(
    gray: np.ndarray,
    min_radius_px: float = 3.0,
    max_radius_px: float = 60.0,
    min_circularity: float = 0.75,
) -> List[Tuple[float, float, float]]:
    """Find circular blob candidates (round copper fiducial pads) in a
    grayscale frame. Returns a list of (x_px, y_px, radius_px), most
    circular first.

    Both polarities are thresholded (Otsu) and searched, since a
    fiducial's brightness relative to the surrounding solder mask can
    go either way depending on finish and lighting.

    Large frames are searched at reduced resolution and the results
    scaled back: a 37MP capture otherwise spends seconds finding
    thousands of contours, and a fiducial pad is many pixels across at
    any sane working distance, so the detail is not needed to locate it.
    """
    shrink = 1.0
    if max(gray.shape[:2]) > DETECT_MAX_DIMENSION:
        shrink = DETECT_MAX_DIMENSION / max(gray.shape[:2])
        gray = cv2.resize(gray, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        min_radius_px *= shrink
        max_radius_px *= shrink

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    found = []
    for invert in (False, True):
        src = cv2.bitwise_not(blurred) if invert else blurred
        _, binary = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area <= 0:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < min_circularity:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if not (min_radius_px <= radius <= max_radius_px):
                continue
            found.append((float(x), float(y), float(radius), circularity))

    # de-duplicate near-identical detections from the two polarity passes
    found.sort(key=lambda c: -c[3])
    deduped = []
    for x, y, r, c in found:
        if any(math.hypot(x - dx, y - dy) < max(r, dr) for dx, dy, dr, _ in deduped):
            continue
        deduped.append((x, y, r, c))

    back = 1.0 / shrink
    return [(x * back, y * back, r * back) for x, y, r, _ in deduped]


# ---------------------------------------------------------------------
# 2. Correspondence solving (RANSAC over similarity transforms)
# ---------------------------------------------------------------------

def _similarity_from_pair(mm_a: Point, mm_b: Point, px_a: Point, px_b: Point):
    """Solve the uniform scale + rotation + translation that maps
    (mm_a -> px_a) and (mm_b -> px_b) exactly. Returns a 2x3 affine
    matrix, or None if the mm points are coincident."""
    mm_dx, mm_dy = mm_b[0] - mm_a[0], mm_b[1] - mm_a[1]
    px_dx, px_dy = px_b[0] - px_a[0], px_b[1] - px_a[1]
    mm_len = math.hypot(mm_dx, mm_dy)
    if mm_len < 1e-6:
        return None
    scale = math.hypot(px_dx, px_dy) / mm_len
    if scale < 1e-6:
        return None
    theta = math.atan2(px_dy, px_dx) - math.atan2(mm_dy, mm_dx)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    a, b = scale * cos_t, -scale * sin_t
    c, d = scale * sin_t, scale * cos_t
    tx = px_a[0] - (a * mm_a[0] + b * mm_a[1])
    ty = px_a[1] - (c * mm_a[0] + d * mm_a[1])
    return np.array([[a, b, tx], [c, d, ty]], dtype=np.float64)


def _apply_affine(affine: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    ones = np.ones((len(pts_mm), 1))
    homog = np.hstack([pts_mm, ones])
    return (affine @ homog.T).T


def find_correspondence_ransac(
    known_mm: List[Point],
    candidates_px: List[Point],
    inlier_thresh_px: float = 10.0,
    iterations: int = 3000,
    min_inliers: int = 3,
    seed: int = 0,
):
    """Solve the unknown correspondence between known fiducial mm
    positions and detected pixel candidates by hypothesize-and-verify:
    repeatedly pick a random pair of known points and a random pair of
    detected points, solve the similarity transform implied by that
    pairing, and count how many of the remaining known points land
    near a detected candidate under it. Keeps the best-scoring
    hypothesis found.

    Returns (affine, matched_mm, matched_px, inlier_count, ambiguous).
    `ambiguous` is True when a meaningfully different assignment scores
    nearly as well as the winner -- e.g. a repetitive panel pattern
    where shifting by one panel pitch looks just as good.
    """
    n_known = len(known_mm)
    n_cand = len(candidates_px)
    if n_known < 2 or n_cand < 2:
        return None, [], [], 0, False

    rng = random.Random(seed)
    known_arr = np.asarray(known_mm, dtype=np.float64)
    cand_arr = np.asarray(candidates_px, dtype=np.float64)

    scored = []
    seen = set()
    attempts = 0
    max_attempts = iterations * 5
    while len(scored) < iterations and attempts < max_attempts:
        attempts += 1
        mi, mj = rng.sample(range(n_known), 2)
        pi, pj = rng.sample(range(n_cand), 2)
        key = (mi, mj, pi, pj)
        if key in seen:
            continue
        seen.add(key)

        affine = _similarity_from_pair(known_mm[mi], known_mm[mj], candidates_px[pi], candidates_px[pj])
        if affine is None:
            continue
        projected = _apply_affine(affine, known_arr)
        diffs = projected[:, None, :] - cand_arr[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(n_known), nearest_idx]
        inlier_mask = nearest_dist <= inlier_thresh_px
        inlier_count = int(inlier_mask.sum())
        if inlier_count < min_inliers:
            continue
        sse = float(np.sum(nearest_dist[inlier_mask] ** 2))
        scored.append((inlier_count, -sse, affine, nearest_idx, inlier_mask))

    if not scored:
        return None, [], [], 0, False

    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    best_count, best_neg_sse, best_affine, best_idx, best_mask = scored[0]
    best_rms = math.sqrt(-best_neg_sse / best_count)

    # Ambiguous means a DIFFERENT alignment (e.g. a repetitive panel
    # pattern shifted by one pitch) fits *comparably well* -- not merely
    # that some other hypothesis also cleared the inlier-count bar. Two
    # failure modes must both be excluded here:
    #   1. A candidate that ties within the inlier threshold near the
    #      same true position (a stray distractor a couple of px closer
    #      than the real fiducial) implies almost the SAME transform --
    #      compare transforms geometrically, not raw candidate-index
    #      assignments, or this harmless tie-break gets flagged.
    #   2. A hypothesis that reaches the same inlier COUNT only by
    #      barely squeaking every point under the pixel threshold (much
    #      higher per-point residual than the best fit) is noise, not a
    #      competing alignment -- compare fit quality (RMS), not just count.
    best_projected = _apply_affine(best_affine, known_arr)
    transform_diff_thresh_px = max(3.0 * inlier_thresh_px, 30.0)
    quality_factor = 2.5
    ambiguous = False
    for count, neg_sse, affine, _nearest_idx, _inlier_mask in scored[1:30]:
        if count < best_count - 1:
            continue
        rms = math.sqrt(-neg_sse / count)
        if rms > best_rms * quality_factor:
            continue
        projected = _apply_affine(affine, known_arr)
        if np.max(np.linalg.norm(projected - best_projected, axis=1)) > transform_diff_thresh_px:
            ambiguous = True
            break

    best_indices = np.where(best_mask)[0]
    matched_mm = [tuple(known_arr[i]) for i in best_indices]
    matched_px = [tuple(cand_arr[best_idx[i]]) for i in best_indices]

    return best_affine, matched_mm, matched_px, best_count, ambiguous


# ---------------------------------------------------------------------
# 3. Homography fitting
# ---------------------------------------------------------------------

def is_usable_transform(H) -> bool:
    """Reject a transform that cannot actually be used to project points:
    non-finite, not invertible, or with a collapsed linear part (every
    point mapped onto a line)."""
    if H is None:
        return False
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        return False
    if abs(H[2, 2]) < 1e-12:
        return False
    if abs(np.linalg.det(H[:2, :2])) < 1e-9:
        return False
    try:
        np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return False
    return True


def fit_homography(matched_mm: List[Point], matched_px: List[Point]):
    """Fit the final mm->px transform from point correspondences, using
    the most general model the points can actually support.

    A full 8-DOF homography needs 4+ points with no three collinear.
    Real boards routinely break that: fiducials along one edge, or two
    rows across a panel, put every 4-point sample on a pair of lines and
    cv2.findHomography then fails outright. So each model is tried in
    turn -- homography, affine, similarity -- and the first that yields a
    usable transform wins. The result is always a 3x3 matrix so callers
    need not care which model was used.

    Returns (H, rms_error_px), or (None, inf) on failure.
    """
    n = len(matched_mm)
    if n < 2:
        return None, float("inf")

    mm = np.asarray(matched_mm, dtype=np.float64)
    px = np.asarray(matched_px, dtype=np.float64)

    candidates = []
    if n >= 4:
        H, _ = cv2.findHomography(mm, px, cv2.RANSAC, 5.0)
        candidates.append(H)
    if n >= 3:
        affine, _ = cv2.estimateAffine2D(mm, px, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if affine is not None:
            candidates.append(np.vstack([affine, [0, 0, 1]]))
    if n >= 2:
        similarity, _ = cv2.estimateAffinePartial2D(
            mm, px, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        if similarity is not None:
            candidates.append(np.vstack([similarity, [0, 0, 1]]))
        pair = _similarity_from_pair(tuple(mm[0]), tuple(mm[1]), tuple(px[0]), tuple(px[1]))
        if pair is not None:
            candidates.append(np.vstack([pair, [0, 0, 1]]))

    for H in candidates:
        if not is_usable_transform(H):
            continue
        projected = mm_to_px_batch(H, mm)
        rms = float(np.sqrt(np.mean(np.sum((projected - px) ** 2, axis=1))))
        if np.isfinite(rms):
            return H, rms

    return None, float("inf")


def mm_to_px_batch(H: np.ndarray, mm_points: np.ndarray) -> np.ndarray:
    pts = mm_points.reshape(-1, 1, 2).astype(np.float64)
    out = cv2.perspectiveTransform(pts, H)
    return out.reshape(-1, 2)


def mm_to_px(H: np.ndarray, x_mm: float, y_mm: float) -> Point:
    out = mm_to_px_batch(H, np.array([[x_mm, y_mm]]))
    return float(out[0, 0]), float(out[0, 1])


def px_to_mm(H: np.ndarray, x_px: float, y_px: float) -> Point:
    H_inv = np.linalg.inv(H)
    return mm_to_px(H_inv, x_px, y_px)


# ---------------------------------------------------------------------
# 4. High level entry points
# ---------------------------------------------------------------------

def auto_calibrate(
    gray_image: np.ndarray,
    fiducials_mm: List[Point],
    inlier_thresh_px: float = 10.0,
    min_radius_px: float = 3.0,
    max_radius_px: float = 60.0,
    ransac_iterations: int = 3000,
) -> CalibrationResult:
    if len(fiducials_mm) < 2:
        return CalibrationResult(
            success=False, method="auto",
            message=f"Program defines only {len(fiducials_mm)} fiducial(s); need at least 2.",
        )

    candidates = detect_fiducial_candidates(gray_image, min_radius_px, max_radius_px)
    if len(candidates) < 2:
        return CalibrationResult(
            success=False, method="auto",
            message=f"Only {len(candidates)} circular fiducial candidate(s) detected in frame.",
        )

    # Every candidate is kept on purpose. Trimming the list to speed the
    # search up also removes the rival hypotheses the ambiguity check
    # relies on, which turns an honest "ambiguous" into a confident
    # wrong answer -- the one failure this must never produce.
    candidates_px = [(x, y) for x, y, _r in candidates]
    affine, matched_mm, matched_px, inlier_count, ambiguous = find_correspondence_ransac(
        fiducials_mm, candidates_px, inlier_thresh_px=inlier_thresh_px, iterations=ransac_iterations,
    )

    min_needed = min(4, len(fiducials_mm))
    if affine is None or inlier_count < min_needed:
        return CalibrationResult(
            success=False, method="auto", ambiguous=ambiguous, inlier_count=inlier_count,
            message=f"Could not confidently match fiducials (best fit: {inlier_count}/{len(fiducials_mm)}).",
        )

    H, rms = fit_homography(matched_mm, matched_px)
    if H is None:
        return CalibrationResult(success=False, method="auto", ambiguous=ambiguous,
                                  message="Homography fit failed on matched fiducials.")

    if ambiguous:
        return CalibrationResult(
            success=False, method="auto", ambiguous=True,
            homography=H, matched_mm=matched_mm, matched_px=matched_px,
            inlier_count=inlier_count, rms_error_px=rms,
            message=("Fiducial match is ambiguous (a meaningfully different alignment scored nearly "
                      "as well -- likely a repetitive panel pattern). Falling back to manual calibration."),
        )

    return CalibrationResult(
        success=True, method="auto", homography=H,
        matched_mm=matched_mm, matched_px=matched_px, inlier_count=inlier_count, rms_error_px=rms,
        message=f"Auto-calibrated from {inlier_count}/{len(fiducials_mm)} fiducials, RMS error {rms:.2f}px.",
    )


def manual_calibrate(fiducials_mm_ordered: List[Point], clicked_px_ordered: List[Point]) -> CalibrationResult:
    if len(fiducials_mm_ordered) != len(clicked_px_ordered):
        return CalibrationResult(success=False, method="manual",
                                  message="Number of clicked points does not match number of fiducials.")
    if len(fiducials_mm_ordered) < 2:
        return CalibrationResult(success=False, method="manual",
                                  message="Need at least 2 fiducial points to calibrate.")

    H, rms = fit_homography(fiducials_mm_ordered, clicked_px_ordered)
    if H is None:
        return CalibrationResult(success=False, method="manual", message="Homography fit failed.")

    return CalibrationResult(
        success=True, method="manual", homography=H,
        matched_mm=list(fiducials_mm_ordered), matched_px=list(clicked_px_ordered),
        inlier_count=len(fiducials_mm_ordered), rms_error_px=rms,
        message=f"Manually calibrated from {len(fiducials_mm_ordered)} points, RMS error {rms:.2f}px.",
    )
