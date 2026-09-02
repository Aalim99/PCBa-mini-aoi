"""
testutils.py

Synthetic PCB frame generation for testing calibration/detection code
without a physical board or camera. Dev/test-only -- not used by the
running app.
"""
import cv2
import numpy as np


def make_synthetic_board_frame(
    fiducials_mm,
    homography,
    image_size=(800, 600),
    fiducial_radius_px=8,
    board_color=(60, 130, 60),
    fiducial_color=(210, 210, 60),
    distractor_circles=0,
    noise_std=4.0,
    rng=None,
):
    """Render a fake top-down PCB photo: a solid 'solder mask' colored
    background with circular fiducial pads drawn at the pixel positions
    implied by projecting `fiducials_mm` through `homography`, plus
    optional random distractor circles (to stress-test correspondence
    matching against false positives) and sensor noise.

    Returns (frame_bgr, fiducial_px_positions).
    """
    rng = rng or np.random.default_rng(0)
    w, h = image_size
    frame = np.empty((h, w, 3), dtype=np.uint8)
    frame[:, :] = board_color

    pts = np.asarray(fiducials_mm, dtype=np.float64).reshape(-1, 1, 2)
    px = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)

    for x, y in px:
        cv2.circle(frame, (int(round(x)), int(round(y))), fiducial_radius_px, fiducial_color, -1)
        cv2.circle(frame, (int(round(x)), int(round(y))), fiducial_radius_px, (30, 30, 30), 1)

    for _ in range(distractor_circles):
        x = rng.uniform(0, w)
        y = rng.uniform(0, h)
        r = rng.uniform(3, fiducial_radius_px * 1.5)
        color = tuple(int(c) for c in rng.uniform(40, 220, size=3))
        cv2.circle(frame, (int(x), int(y)), int(r), color, -1)

    if noise_std > 0:
        noise = rng.normal(0, noise_std, frame.shape)
        frame = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    return frame, px


def draw_components(frame, components, homography, part_sizes, missing_designators=(),
                    contrast=1.0):
    """Draw component bodies onto a synthetic board frame at the pixel
    positions implied by `homography`, skipping any designator listed in
    `missing_designators` so its pad is left bare. Populated parts get a
    dark textured body (varied interior + light terminations); bare pads
    get a flat low-contrast patch, which is what the presence heuristic
    is meant to tell apart.

    `contrast` scales how far the body and terminations sit from the
    board colour: 1.0 is a boldly visible part, lower values approach a
    subtle one that the presence heuristic finds genuinely hard.

    Mutates and returns `frame`.
    """
    from core.inspection import project_local_points

    board = np.array([60, 130, 60], dtype=np.float64)

    def shade(colour):
        c = board + (np.asarray(colour, dtype=np.float64) - board) * float(contrast)
        return tuple(int(round(v)) for v in np.clip(c, 0, 255))

    def fill(local_quad, colour, comp):
        pts = project_local_points(homography, comp["x"], comp["y"],
                                   comp.get("rotation", 0.0) or 0.0,
                                   np.asarray(local_quad, dtype=np.float64))
        # LINE_8, not LINE_AA: crisp edges keep measured extents exact in
        # tests, where an antialiased ramp reads as extra component width.
        cv2.fillConvexPoly(frame, np.round(pts).astype(np.int32), colour, lineType=cv2.LINE_8)

    missing = set(missing_designators)
    for c in components:
        size = part_sizes.get(c.get("part"))
        if not size:
            continue
        w_mm, h_mm = float(size["width_mm"]), float(size["height_mm"])
        hw, hh = w_mm / 2.0, h_mm / 2.0
        # Project the true rotated footprint rather than filling its
        # axis-aligned bounding box: a tilted camera would otherwise draw
        # every part fatter than it really is.
        body = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]

        if c["designator"] in missing:
            fill(body, (70, 140, 70), c)   # bare pad: flat, close to the solder mask
            continue

        fill(body, shade((35, 35, 40)), c)                             # dark body
        term = max(w_mm / 5.0, 0.05)
        light = shade((200, 200, 205))
        fill([(-hw, -hh), (-hw + term, -hh), (-hw + term, hh), (-hw, hh)], light, c)
        fill([(hw - term, -hh), (hw, -hh), (hw, hh), (hw - term, hh)], light, c)
        fill([(-hw, -hh * 0.12), (hw, -hh * 0.12), (hw, hh * 0.12), (-hw, hh * 0.12)],
             shade((120, 120, 125)), c)                                # body marking
    return frame


def autosize_canvas(fiducials_mm, homography, margin_px=100):
    """Pick a canvas size that comfortably contains every projected
    fiducial position, so tests don't silently clip fiducials off the
    edge of the synthetic frame (a clipped circle isn't circular).
    Assumes all projected coordinates are non-negative -- use
    place_homography() to build a homography that guarantees that."""
    pts = np.asarray(fiducials_mm, dtype=np.float64).reshape(-1, 1, 2)
    px = cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
    w = int(np.ceil(px[:, 0].max())) + margin_px
    h = int(np.ceil(px[:, 1].max())) + margin_px
    return max(w, 200), max(h, 200)


def make_ground_truth_homography(scale=8.0, angle_deg=3.0, tx=60.0, ty=40.0):
    """A synthetic mm->px homography (pure similarity: uniform scale +
    rotation + translation) standing in for a real fixed top-down
    camera's calibration, for use as ground truth in tests. Caller is
    responsible for tx/ty keeping fiducials on-canvas; prefer
    place_homography() when rendering a synthetic frame."""
    theta = np.radians(angle_deg)
    return np.array([
        [scale * np.cos(theta), -scale * np.sin(theta), tx],
        [scale * np.sin(theta), scale * np.cos(theta), ty],
        [0, 0, 1],
    ], dtype=np.float64)


def place_homography(fiducials_mm, scale=8.0, angle_deg=3.0, margin_px=80.0):
    """Build a mm->px similarity homography with translation chosen so
    EVERY fiducial projects to a positive-coordinate pixel with margin
    to spare in both axes -- avoids accidentally placing a fiducial
    off-canvas (negative x or y) purely as a side effect of the chosen
    rotation, which autosize_canvas (max-only) would not catch."""
    theta = np.radians(angle_deg)
    R = np.array([[scale * np.cos(theta), -scale * np.sin(theta)],
                  [scale * np.sin(theta), scale * np.cos(theta)]])
    pts = np.asarray(fiducials_mm, dtype=np.float64)
    rotated = pts @ R.T
    min_x, min_y = rotated.min(axis=0)
    tx = margin_px - min_x
    ty = margin_px - min_y
    return np.array([
        [R[0, 0], R[0, 1], tx],
        [R[1, 0], R[1, 1], ty],
        [0, 0, 1],
    ], dtype=np.float64)
