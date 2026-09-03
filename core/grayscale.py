"""
grayscale.py

How a colour frame is reduced to the single channel the presence check
measures. This is a genuine tuning knob, not a formality: the heuristic
decides on variation within an ROI, so whatever makes a component stand
out from the bare board directly decides how well it works.

Channel choice matters most on a PCB. A green solder mask is bright in
the green channel and dark in the red one, so picking red often doubles
the contrast between the board and the parts sitting on it, while plain
luma averages much of that away. Gamma then lifts detail out of dark
component bodies, and contrast stretches what is left.

Settings are stored beside part_sizes.json and shared across programs,
since they describe the station's lighting and camera rather than any
one board.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MODES = ("luma", "red", "green", "blue", "max", "min")
MODE_LABELS = {
    "luma": "Luma (standard)",
    "red": "Red channel",
    "green": "Green channel",
    "blue": "Blue channel",
    "max": "Brightest channel",
    "min": "Darkest channel",
}


@dataclass
class GrayscaleSettings:
    mode: str = "luma"
    gamma: float = 1.0        # >1 lifts shadows, <1 deepens them
    contrast: float = 1.0     # gain about mid-grey
    brightness: float = 0.0   # flat offset, -100..100

    def as_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "GrayscaleSettings":
        base = GrayscaleSettings()
        if not isinstance(data, dict):
            return base
        mode = str(data.get("mode", base.mode))
        return GrayscaleSettings(
            mode=mode if mode in MODES else base.mode,
            gamma=float(data.get("gamma", base.gamma)),
            contrast=float(data.get("contrast", base.contrast)),
            brightness=float(data.get("brightness", base.brightness)),
        )

    @property
    def is_default(self) -> bool:
        return (self.mode == "luma" and abs(self.gamma - 1.0) < 1e-6
                and abs(self.contrast - 1.0) < 1e-6 and abs(self.brightness) < 1e-6)

    def summary(self) -> str:
        bits = [MODE_LABELS.get(self.mode, self.mode)]
        if abs(self.gamma - 1.0) > 1e-6:
            bits.append(f"gamma {self.gamma:.2f}")
        if abs(self.contrast - 1.0) > 1e-6:
            bits.append(f"contrast {self.contrast:.2f}")
        if abs(self.brightness) > 1e-6:
            bits.append(f"brightness {self.brightness:+.0f}")
        return ", ".join(bits)


def _channel(frame: np.ndarray, mode: str) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if mode == "blue":
        return frame[:, :, 0]
    if mode == "green":
        return frame[:, :, 1]
    if mode == "red":
        return frame[:, :, 2]
    if mode == "max":
        return frame.max(axis=2)
    if mode == "min":
        return frame.min(axis=2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _tone_lut(settings: GrayscaleSettings) -> Optional[np.ndarray]:
    """A 256-entry lookup for gamma/contrast/brightness.

    Built once and applied with cv2.LUT rather than per-pixel maths,
    which matters when a frame is tens of megapixels.
    """
    if (abs(settings.gamma - 1.0) < 1e-6 and abs(settings.contrast - 1.0) < 1e-6
            and abs(settings.brightness) < 1e-6):
        return None
    values = np.arange(256, dtype=np.float64)
    gamma = max(settings.gamma, 1e-3)
    values = 255.0 * np.power(values / 255.0, 1.0 / gamma)
    values = (values - 128.0) * settings.contrast + 128.0 + settings.brightness
    return np.clip(values, 0, 255).astype(np.uint8)


def to_gray(frame: np.ndarray, settings: Optional[GrayscaleSettings] = None) -> np.ndarray:
    """Reduce a frame to the single channel the presence check measures."""
    settings = settings or GrayscaleSettings()
    gray = _channel(frame, settings.mode)
    lut = _tone_lut(settings)
    return cv2.LUT(gray, lut) if lut is not None else gray


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

def load_grayscale_settings(path: str) -> GrayscaleSettings:
    p = Path(path)
    if not p.exists():
        return GrayscaleSettings()
    try:
        return GrayscaleSettings.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return GrayscaleSettings()


def save_grayscale_settings(path: str, settings: GrayscaleSettings) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(settings.as_dict(), indent=2))
