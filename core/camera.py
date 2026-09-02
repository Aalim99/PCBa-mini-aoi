"""
camera.py

Thin wrapper over the inspection camera, with a still-image source as a
drop-in stand-in. Two reasons for the stand-in: the station should stay
usable for reviewing a saved capture when no camera is attached, and it
lets the whole live pipeline be exercised without hardware.
"""

import platform
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def backend_candidates() -> List[int]:
    """Capture backends to try, best-first for this platform.

    Windows matters most here: the default backend (MSMF) often fails to
    open cameras that DirectShow opens fine, which looks to the operator
    like "no camera" on a machine with a working camera.
    """
    system = platform.system()
    if system == "Windows":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if system == "Darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def backend_name(backend: int) -> str:
    return {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "MediaFoundation",
            cv2.CAP_V4L2: "V4L2", cv2.CAP_AVFOUNDATION: "AVFoundation",
            cv2.CAP_ANY: "default"}.get(backend, str(backend))


@contextmanager
def _quiet_opencv():
    """Silence OpenCV's console spew while probing absent devices."""
    try:
        previous = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            try:
                cv2.utils.logging.setLogLevel(previous)
            except Exception:
                pass


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
        self.backend: Optional[int] = None
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Try each platform backend in turn until one actually delivers
        a frame. isOpened() alone is not enough -- some backends report
        success on a device that then never returns an image."""
        for backend in backend_candidates():
            with _quiet_opencv():
                cap = cv2.VideoCapture(self.index, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                if self.width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                if self.height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    continue
            self._cap = cap
            self.backend = backend
            return True
        self._cap = None
        self.backend = None
        return False

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
        via = f" via {backend_name(self.backend)}" if self.backend is not None else ""
        return f"camera #{self.index}{via}"


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


def probe_camera(index: int) -> Optional[dict]:
    """Details for a camera that can actually deliver a frame, else None.
    Returns {index, backend, backend_name, width, height, label}."""
    cam = Camera(index)
    if not cam.open():
        return None
    try:
        frame = cam.read()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        return {
            "index": index,
            "backend": cam.backend,
            "backend_name": backend_name(cam.backend),
            "width": w,
            "height": h,
            "label": f"Camera {index}  ({w}x{h}, {backend_name(cam.backend)})",
        }
    finally:
        cam.release()


def list_cameras(max_index: int = 6) -> List[dict]:
    """Detect usable cameras by index. A device counts only if it hands
    back a real frame, so indices that merely open but never deliver
    aren't offered to the operator as a choice."""
    found = []
    for i in range(max_index):
        info = probe_camera(i)
        if info:
            found.append(info)
    return found
