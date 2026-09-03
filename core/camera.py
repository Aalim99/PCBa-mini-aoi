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


# Many UVC cameras hand back a few junk frames right after opening.
WARMUP_FRAMES = 5
# A frame this flat carries no picture. A camera whose pixel format was
# mis-negotiated typically returns a solid fill rather than failing the
# read, which looks to the operator like a blank coloured rectangle.
# Set above what pure sensor noise survives the downsample below (~1.5)
# and far below any real scene, so noise alone still counts as blank.
BLANK_FRAME_STD = 3.0
# Tried in order when the native format gives nothing usable. MJPG first
# because an uncompressed high-resolution stream exceeds USB bandwidth on
# many cameras, which is exactly when the driver starts emitting filler.
FALLBACK_MODES = [
    ("MJPG", None, None),
    ("MJPG", 1920, 1080),
    ("MJPG", 1280, 720),
    (None, 1280, 720),
]


def normalise_frame(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Coerce whatever the driver returned into 3-channel BGR."""
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def frame_is_blank(frame: Optional[np.ndarray], threshold: float = BLANK_FRAME_STD) -> bool:
    """True when a frame is essentially one flat colour, i.e. carries no
    image at all.

    Variation is measured within each channel separately and the largest
    taken. A single std over the whole array would call a solid green
    frame "varied" purely because its channels differ from each other --
    which is exactly the frame this is meant to catch.

    Downsampled first so this stays cheap on a 4K frame.
    """
    if frame is None or frame.size == 0:
        return True
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        return float(small.std()) < threshold
    per_channel = small.reshape(-1, small.shape[2]).std(axis=0)
    return float(per_channel.max()) < threshold


def fourcc_of(cap) -> str:
    try:
        value = int(cap.get(cv2.CAP_PROP_FOURCC))
    except Exception:
        return ""
    if value <= 0:
        return ""
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


class Camera(CameraSource):
    """A live V4L2/DirectShow camera by device index."""

    def __init__(self, index: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        self.index = index
        self.width = width
        self.height = height
        self.backend: Optional[int] = None
        self.fourcc: str = ""
        self.frame_size = (0, 0)
        # True when the camera opened but only ever produced flat frames,
        # so the caller can say so instead of showing a blank rectangle.
        self.delivers_blank_frames = False
        self._cap: Optional[cv2.VideoCapture] = None

    # ---------- opening ----------
    def _start(self, backend, fourcc, width, height):
        """Open one candidate configuration, or None if it won't start."""
        with _quiet_opencv():
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                return None
            if fourcc:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap

    @staticmethod
    def _warm_up(cap):
        """Read past the junk frames a camera emits on start-up, and
        return the first frame that looks like a picture (or the last one
        read, so the caller can judge)."""
        frame = None
        with _quiet_opencv():
            for _ in range(WARMUP_FRAMES):
                ok, candidate = cap.read()
                if not ok or candidate is None:
                    continue
                frame = normalise_frame(candidate)
                if not frame_is_blank(frame):
                    return frame
        return frame

    def _adopt(self, cap, backend, blank):
        self._cap = cap
        self.backend = backend
        self.fourcc = fourcc_of(cap)
        self.frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.delivers_blank_frames = blank

    def open(self, thorough: bool = True) -> bool:
        """Open the camera, negotiating a format that actually produces a
        picture.

        isOpened() is not enough -- some backends report success on a
        device that never returns an image, and some return frames that
        decode to a flat colour. So each candidate is warmed up and the
        frame inspected, and a configuration that only yields flat frames
        is kept only as a last resort, flagged.

        `thorough=False` tries just the native format on each backend,
        for scanning which devices exist without a long negotiation.
        """
        fallback = None
        for backend in backend_candidates():
            modes = [(None, self.width, self.height)]
            if thorough:
                modes += [m for m in FALLBACK_MODES
                          if (m[1], m[2]) != (self.width, self.height) or m[0]]
            for fourcc, width, height in modes:
                cap = self._start(backend, fourcc, width, height)
                if cap is None:
                    continue
                frame = self._warm_up(cap)
                if frame is None:
                    cap.release()
                    continue
                if not frame_is_blank(frame):
                    self._adopt(cap, backend, blank=False)
                    return True
                if fallback is None:
                    fallback = (cap, backend)
                else:
                    cap.release()
                if not thorough:
                    break

        if fallback is not None:
            self._adopt(fallback[0], fallback[1], blank=True)
            return True

        self._cap = None
        self.backend = None
        return False

    # ---------- use ----------
    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return normalise_frame(frame) if ok and frame is not None else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def description(self) -> str:
        parts = [f"camera #{self.index}"]
        if self.frame_size[0]:
            parts.append(f"{self.frame_size[0]}x{self.frame_size[1]}")
        if self.fourcc:
            parts.append(self.fourcc)
        if self.backend is not None:
            parts.append(backend_name(self.backend))
        return f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0]


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

    Deliberately not thorough: scanning several indices should be quick,
    and format negotiation is left to the real open when the operator
    starts the live view.
    """
    cam = Camera(index)
    if not cam.open(thorough=False):
        return None
    try:
        frame = cam.read()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        note = "  [blank frames]" if cam.delivers_blank_frames else ""
        return {
            "index": index,
            "backend": cam.backend,
            "backend_name": backend_name(cam.backend),
            "width": w,
            "height": h,
            "blank": cam.delivers_blank_frames,
            "label": f"Camera {index}  ({w}x{h}, {backend_name(cam.backend)}){note}",
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
