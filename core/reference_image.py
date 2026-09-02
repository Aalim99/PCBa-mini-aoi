"""
reference_image.py

A reference photo of a known-good board, aligned to the program's
fiducials, so ROI box sizes can be set against the real component
instead of guessed in millimetres.

Alignment reuses the same fiducial machinery as live calibration, so
the reference sits in board mm exactly like the live camera does. The
saved homography is what makes a stored reference reusable: reopen the
program later and the image still lines up without re-clicking.

`component_patch` returns the image around one component resampled into
the ROI editor's own scene space (a fixed pixels-per-mm), de-rotated by
the component's placement angle. So a part always appears at its native
width x height whatever its rotation on the board, and a box drawn to
2.0mm on screen is 2.0mm on the board.
"""

import json
import math
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

REFERENCE_SUFFIX = ".reference.json"
Point = Tuple[float, float]


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def reference_paths(program_name: str, programs_dir: str) -> Tuple[Path, Path]:
    """(sidecar json path, stored image path stem) for a program."""
    base = Path(programs_dir)
    return base / f"{program_name}{REFERENCE_SUFFIX}", base / f"{program_name}.reference"


def save_reference(program_name: str, programs_dir: str, image_path: str,
                   homography: np.ndarray, copy_image: bool = True) -> dict:
    """Store the reference alongside the program. The image is copied
    into the programs directory by default so a program keeps working
    after the original photo is moved or deleted."""
    json_path, image_stem = reference_paths(program_name, programs_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    stored_path = Path(image_path)
    if copy_image:
        stored_path = image_stem.with_suffix(Path(image_path).suffix or ".png")
        if Path(image_path).resolve() != stored_path.resolve():
            shutil.copyfile(image_path, stored_path)

    record = {
        "image": stored_path.name if copy_image else str(stored_path),
        "homography": np.asarray(homography, dtype=np.float64).tolist(),
    }
    json_path.write_text(json.dumps(record, indent=2))
    return record


def load_reference(program_name: str, programs_dir: str) -> Optional[dict]:
    """Returns {"image_path", "homography", "image"} or None. The image
    is not decoded here -- callers load it when they need pixels."""
    json_path, _ = reference_paths(program_name, programs_dir)
    if not json_path.exists():
        return None
    try:
        record = json.loads(json_path.read_text())
        homography = np.asarray(record["homography"], dtype=np.float64)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None

    image_path = Path(record.get("image", ""))
    if not image_path.is_absolute():
        image_path = Path(programs_dir) / image_path
    if not image_path.exists():
        return None
    return {"image_path": str(image_path), "homography": homography}


def delete_reference(program_name: str, programs_dir: str) -> None:
    json_path, _ = reference_paths(program_name, programs_dir)
    record = load_reference(program_name, programs_dir)
    json_path.unlink(missing_ok=True)
    if record:
        stored = Path(record["image_path"])
        if stored.parent.resolve() == Path(programs_dir).resolve():
            stored.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------

def scene_to_board_matrix(x_mm: float, y_mm: float, rotation_deg: float,
                          half_extent: float, px_per_mm: float) -> np.ndarray:
    """Map ROI-editor scene pixels to board millimetres.

    Scene space has the component centred, `px_per_mm` scene pixels per
    millimetre, and y running down the screen -- hence the y flip, which
    keeps the patch the same way up as the camera sees it.
    """
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    recentre = np.array([[1, 0, -half_extent], [0, 1, -half_extent], [0, 0, 1]], dtype=np.float64)
    to_mm = np.array([[1 / px_per_mm, 0, 0], [0, -1 / px_per_mm, 0], [0, 0, 1]], dtype=np.float64)
    rotate = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float64)
    translate = np.array([[1, 0, x_mm], [0, 1, y_mm], [0, 0, 1]], dtype=np.float64)
    return translate @ rotate @ to_mm @ recentre


def component_patch(image: np.ndarray, homography: np.ndarray, x_mm: float, y_mm: float,
                    rotation_deg: float = 0.0, half_extent: float = 160.0,
                    px_per_mm: float = 40.0) -> Optional[np.ndarray]:
    """Square crop of the reference image around one component, in the
    ROI editor's scene space: the component sits at the centre,
    de-rotated, at `px_per_mm` scene pixels per millimetre.

    Returns None if the transform is degenerate or the requested patch
    is unusably small.
    """
    if image is None or homography is None:
        return None
    size = int(round(2 * half_extent))
    if size < 8 or size > 4096:
        return None

    scene_to_board = scene_to_board_matrix(x_mm, y_mm, rotation_deg, half_extent, px_per_mm)
    scene_to_px = np.asarray(homography, dtype=np.float64) @ scene_to_board
    try:
        px_to_scene = np.linalg.inv(scene_to_px)
    except np.linalg.LinAlgError:
        return None

    return cv2.warpPerspective(
        image, px_to_scene, (size, size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(40, 40, 40),
    )


def component_instances(program: dict, part: str) -> List[dict]:
    """Every placement of one part number, in board mm. Panel-expanded,
    so a panel offers each unit's copy rather than only the base unit."""
    from core.inspection import expand_components

    return [c for c in expand_components(program) if c.get("part") == part]
