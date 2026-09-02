"""
barcode_reader.py

Reads a board's traceability barcode out of the same PCB camera frame
used for inspection. Used for logging only -- never for choosing which
program to load, which stays a manual operator choice.

Tries OpenCV's built-in 1D barcode detector and QR detector. Decoding
is best effort by design: a frame with no readable code logs a blank
barcode rather than failing the inspection, since an unreadable label
is not a board defect.
"""

from typing import List, Optional

import cv2
import numpy as np


def _decode_1d(frame: np.ndarray) -> List[str]:
    """OpenCV's 1D barcode detector. Its exact return shape has varied
    across releases, so unpack defensively rather than pinning one."""
    try:
        detector = cv2.barcode.BarcodeDetector()
    except Exception:
        return []

    try:
        out = detector.detectAndDecodeWithType(frame)
    except Exception:
        return []

    # Shapes seen across versions: (ok, info, type, points) or (info, type, points)
    decoded = None
    for element in out:
        if isinstance(element, (list, tuple)) and element and isinstance(element[0], str):
            decoded = element
            break
    if decoded is None:
        return []
    return [s for s in decoded if s]


def _decode_qr(frame: np.ndarray) -> List[str]:
    try:
        detector = cv2.QRCodeDetector()
    except Exception:
        return []

    # Multi first (a panel can carry one label per unit), then single.
    try:
        ok, infos, _pts, _straight = detector.detectAndDecodeMulti(frame)
        if ok and infos:
            found = [s for s in infos if s]
            if found:
                return found
    except Exception:
        pass

    try:
        data, _pts, _straight = detector.detectAndDecode(frame)
        return [data] if data else []
    except Exception:
        return []


def read_barcodes(frame: np.ndarray) -> List[str]:
    """All codes readable in the frame (1D barcodes first, then QR).
    Returns an empty list when nothing decodes."""
    if frame is None or frame.size == 0:
        return []
    results = _decode_1d(frame)
    results.extend(s for s in _decode_qr(frame) if s not in results)
    return results


def read_barcode(frame: np.ndarray) -> Optional[str]:
    """The single barcode to log for this board, or None if unreadable."""
    codes = read_barcodes(frame)
    return codes[0] if codes else None
