"""
camera.py

Thin wrapper over the inspection camera, with a still-image source as a
drop-in stand-in. Two reasons for the stand-in: the station should stay
usable for reviewing a saved capture when no camera is attached, and it
lets the whole live pipeline be exercised without hardware.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class CameraSource:
    """Common interface: open() -> bool, read() -> frame or None, release()."""

    def open(self) -> bool:
        raise NotImplementedError

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    @property
    def is_open(self) -> bool:
        return False

    @property
    def description(self) -> str:
        return "unknown source"


class Camera(CameraSource):
    """A live V4L2/DirectShow camera by device index."""

    def __init__(self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        self.index = index
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            self._cap = None
            return False
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        return True

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok and frame is not None else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def description(self) -> str:
        return f"camera #{self.index}"


class StillImageSource(CameraSource):
    """Serves one image over and over, standing in for a live camera."""

    def __init__(self, image: np.ndarray = None, path: str = None):
        self._path = path
        self._frame = image
        self._open = False

    def open(self) -> bool:
        if self._frame is None and self._path:
            self._frame = cv2.imread(self._path)
        self._open = self._frame is not None
        return self._open

    def read(self) -> Optional[np.ndarray]:
        return None if not self._open or self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def description(self) -> str:
        return f"still image ({Path(self._path).name})" if self._path else "still image"


def open_camera(index: int = 0) -> Optional[Camera]:
    """Open a live camera, or None when the device isn't available."""
    cam = Camera(index)
    return cam if cam.open() else None


def list_cameras(max_index: int = 4) -> list:
    """Device indices that can actually be opened. Probing is noisy on
    some backends, so failures are silently skipped."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                found.append(i)
        finally:
            cap.release()
    return found
