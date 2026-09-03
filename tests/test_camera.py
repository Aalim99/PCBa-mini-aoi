"""Tests for core/camera.py -- frame validation and format negotiation.

The failure these guard against: a camera that opens successfully and
returns frames, but whose frames decode to a flat colour. That reaches
the operator as an unexplained blank rectangle unless it is detected.

Run directly:
    python tests/test_camera.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

import core.camera as camera_module
from core.camera import (
    Camera, StillImageSource, backend_candidates, frame_is_blank, normalise_frame,
)


def real_frame(w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (200, 160), (20, 20, 20), -1)
    return frame


def test_blank_frame_detection():
    green = np.zeros((480, 640, 3), np.uint8)
    green[:, :] = (0, 128, 0)          # exactly the reported symptom
    assert frame_is_blank(green), "a solid green frame must read as blank"
    assert frame_is_blank(np.zeros((480, 640, 3), np.uint8)), "black is blank"
    assert frame_is_blank(np.full((480, 640, 3), 255, np.uint8)), "white is blank"
    assert frame_is_blank(None)
    assert not frame_is_blank(real_frame()), "a real picture must not read as blank"
    print("OK test_blank_frame_detection")


def test_blank_detection_ignores_noise_but_sees_structure():
    """Sensor noise alone is not a picture, but a dim scene with real
    structure in it is -- even when it is very dark."""
    noise_only = np.clip(np.random.default_rng(1).normal(18, 6, (240, 320, 3)),
                         0, 255).astype(np.uint8)
    assert frame_is_blank(noise_only), "noise with no structure is not a picture"

    dim_scene = noise_only.copy()
    cv2.rectangle(dim_scene, (60, 50), (240, 180), (70, 70, 70), -1)
    assert not frame_is_blank(dim_scene), "a dim frame with structure is a picture"

    almost_flat = np.full((240, 320, 3), 90, np.uint8)
    almost_flat[0, 0] = 91
    assert frame_is_blank(almost_flat), "one stray pixel does not make a picture"
    print("OK test_blank_detection_ignores_noise_but_sees_structure")


def test_normalise_frame_channels():
    gray = np.full((10, 10), 120, np.uint8)
    assert normalise_frame(gray).shape == (10, 10, 3), "gray must become BGR"
    bgra = np.zeros((10, 10, 4), np.uint8)
    assert normalise_frame(bgra).shape == (10, 10, 3), "BGRA must become BGR"
    bgr = np.zeros((10, 10, 3), np.uint8)
    assert normalise_frame(bgr).shape == (10, 10, 3)
    assert normalise_frame(None) is None
    print("OK test_normalise_frame_channels")


class FakeCapture:
    """Stands in for cv2.VideoCapture: returns whatever frames it is told
    to, so negotiation can be tested without hardware."""

    def __init__(self, frames, opened=True):
        self._frames = frames
        self._opened = opened
        self.released = False
        self.props = {}

    def isOpened(self):
        return self._opened and not self.released

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames[0]

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        if prop == cv2.CAP_PROP_FOURCC:
            return float(cv2.VideoWriter_fourcc(*"MJPG"))
        return 0

    def release(self):
        self.released = True


def run_with_fake(factory, fn):
    """Patch cv2.VideoCapture inside core.camera for one call."""
    original = camera_module.cv2.VideoCapture
    camera_module.cv2.VideoCapture = factory
    try:
        return fn()
    finally:
        camera_module.cv2.VideoCapture = original


def test_open_rejects_blank_and_retries_other_formats():
    """Native format gives a flat frame; a fallback format gives a real
    one. The real one must win, and the camera must not be flagged."""
    green = np.full((480, 640, 3), 0, np.uint8)
    green[:, :] = (0, 128, 0)
    calls = []

    def factory(index, backend):
        calls.append(backend)
        # first configuration attempted is blank, later ones are fine
        return FakeCapture([green] if len(calls) == 1 else [real_frame()])

    cam = Camera(0)
    assert run_with_fake(factory, cam.open) is True
    assert cam.delivers_blank_frames is False, "a good format was available"
    assert len(calls) >= 2, "should have tried past the blank configuration"
    print(f"OK test_open_rejects_blank_and_retries_other_formats: {len(calls)} attempt(s)")


def test_open_flags_camera_that_is_blank_in_every_format():
    """Nothing works: still open (so the operator sees something) but
    flagged, so the UI can explain rather than show a blank rectangle."""
    green = np.full((480, 640, 3), 0, np.uint8)
    green[:, :] = (0, 128, 0)

    cam = Camera(0)
    assert run_with_fake(lambda i, b: FakeCapture([green]), cam.open) is True
    assert cam.delivers_blank_frames is True, "must be flagged as blank"
    print("OK test_open_flags_camera_that_is_blank_in_every_format")


def test_open_fails_when_no_frames_at_all():
    cam = Camera(0)
    assert run_with_fake(lambda i, b: FakeCapture([]), cam.open) is False
    assert cam.is_open is False
    print("OK test_open_fails_when_no_frames_at_all")


def test_open_fails_when_device_absent():
    cam = Camera(9)
    assert run_with_fake(lambda i, b: FakeCapture([], opened=False), cam.open) is False
    print("OK test_open_fails_when_device_absent")


def test_quick_open_does_not_negotiate():
    """Scanning many indices must stay quick: one attempt per backend."""
    attempts = []

    def factory(index, backend):
        attempts.append(backend)
        green = np.full((480, 640, 3), 0, np.uint8)
        green[:, :] = (0, 128, 0)
        return FakeCapture([green])

    cam = Camera(0)
    run_with_fake(factory, lambda: cam.open(thorough=False))
    assert len(attempts) == len(backend_candidates()), attempts
    print(f"OK test_quick_open_does_not_negotiate: {len(attempts)} attempt(s) "
          f"for {len(backend_candidates())} backend(s)")


def test_description_reports_what_was_negotiated():
    cam = Camera(2)
    run_with_fake(lambda i, b: FakeCapture([real_frame()]), cam.open)
    text = cam.description
    assert "camera #2" in text and "640x480" in text and "MJPG" in text, text
    print("OK test_description_reports_what_was_negotiated:", text)


def test_read_normalises_channels():
    gray_frames = [np.full((100, 100), 90, np.uint8)]
    cam = Camera(0)
    # a flat gray frame is blank, so the camera opens flagged -- fine here,
    # the point is that read() hands back 3 channels either way
    run_with_fake(lambda i, b: FakeCapture(gray_frames), cam.open)
    frame = cam.read()
    assert frame is not None and frame.shape == (100, 100, 3), frame.shape
    print("OK test_read_normalises_channels")


def test_still_image_source_unaffected():
    frame = real_frame()
    source = StillImageSource(image=frame)
    assert source.open() and source.is_open
    assert source.read().shape == frame.shape
    assert not getattr(source, "delivers_blank_frames", False)
    print("OK test_still_image_source_unaffected")


if __name__ == "__main__":
    test_blank_frame_detection()
    test_blank_detection_ignores_noise_but_sees_structure()
    test_normalise_frame_channels()
    test_open_rejects_blank_and_retries_other_formats()
    test_open_flags_camera_that_is_blank_in_every_format()
    test_open_fails_when_no_frames_at_all()
    test_open_fails_when_device_absent()
    test_quick_open_does_not_negotiate()
    test_description_reports_what_was_negotiated()
    test_read_normalises_channels()
    test_still_image_source_unaffected()
    print("\nALL CAMERA TESTS PASSED")
