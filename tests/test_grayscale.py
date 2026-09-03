"""Tests for core/grayscale.py and the way it feeds the presence check.

The claim under test: the grayscale controls are a real tuning knob, not
decoration. Choosing a channel has to change what an ROI measures, and
the change has to reach core.inspection, or the sliders are theatre.

Run directly:
    python tests/test_grayscale.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.grayscale import (
    MODES, GrayscaleSettings, load_grayscale_settings, save_grayscale_settings,
    _tone_lut, to_gray,
)
from core.inspection import measure_roi, to_gray as inspect_to_gray


def green_board_with_part():
    """A green solder mask with a dark part on it -- the case the red
    channel is supposed to handle much better than luma."""
    frame = np.zeros((60, 60, 3), dtype=np.uint8)
    frame[:, :] = (40, 170, 40)          # BGR: green mask, little red
    frame[20:40, 20:40] = (60, 60, 150)  # a part body: strong in red
    return frame


def test_every_mode_returns_a_single_channel():
    frame = green_board_with_part()
    for mode in MODES:
        gray = to_gray(frame, GrayscaleSettings(mode=mode))
        assert gray.ndim == 2, mode
        assert gray.shape == frame.shape[:2], mode
        assert gray.dtype == np.uint8, mode
    print(f"OK test_every_mode_returns_a_single_channel: {len(MODES)} modes")


def test_channel_choice_changes_contrast_on_a_green_board():
    frame = green_board_with_part()

    def separation(mode):
        gray = to_gray(frame, GrayscaleSettings(mode=mode))
        return abs(float(gray[20:40, 20:40].mean()) - float(gray[0:10, 0:10].mean()))

    luma = separation("luma")
    red = separation("red")
    assert red > luma * 1.5, f"red {red:.1f} should beat luma {luma:.1f} on a green board"
    print(f"OK test_channel_choice_changes_contrast_on_a_green_board: "
          f"luma {luma:.1f} -> red {red:.1f}")


def test_defaults_match_plain_luma():
    frame = green_board_with_part()
    import cv2
    assert np.array_equal(to_gray(frame, GrayscaleSettings()),
                          cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    assert GrayscaleSettings().is_default
    assert _tone_lut(GrayscaleSettings()) is None, "default tone must skip the LUT entirely"
    print("OK test_defaults_match_plain_luma")


def test_gamma_lifts_shadows_and_contrast_stretches():
    frame = green_board_with_part()
    base = to_gray(frame).astype(float).mean()
    lifted = to_gray(frame, GrayscaleSettings(gamma=2.2)).astype(float).mean()
    assert lifted > base + 5, f"gamma 2.2 should brighten ({base:.1f} -> {lifted:.1f})"

    plain = to_gray(frame).astype(float)
    stretched = to_gray(frame, GrayscaleSettings(contrast=2.0)).astype(float)
    assert stretched.std() > plain.std(), "contrast 2.0 should widen the spread"

    brighter = to_gray(frame, GrayscaleSettings(brightness=40)).astype(float).mean()
    assert brighter > base + 20, f"brightness +40 should lift the mean ({base:.1f} -> {brighter:.1f})"
    print(f"OK test_gamma_lifts_shadows_and_contrast_stretches: "
          f"mean {base:.1f} -> gamma {lifted:.1f} -> bright {brighter:.1f}")


def test_lut_is_256_entries_and_clamped():
    lut = _tone_lut(GrayscaleSettings(contrast=4.0, brightness=100))
    assert lut is not None and lut.shape == (256,) and lut.dtype == np.uint8
    assert lut.min() >= 0 and lut.max() <= 255, "extreme settings must not wrap around"
    print("OK test_lut_is_256_entries_and_clamped")


def test_extreme_settings_do_not_crash_or_wrap():
    frame = green_board_with_part()
    for settings in (GrayscaleSettings(gamma=0.0), GrayscaleSettings(gamma=100.0),
                     GrayscaleSettings(contrast=0.0), GrayscaleSettings(brightness=-1000)):
        gray = to_gray(frame, settings)
        assert gray.dtype == np.uint8 and gray.shape == frame.shape[:2]
    print("OK test_extreme_settings_do_not_crash_or_wrap")


def test_measurement_actually_moves_with_the_setting():
    """The point of the knob: the numbers the presence check decides on
    have to change when the operator changes the channel."""
    frame = green_board_with_part()
    luma_std, luma_range = measure_roi(to_gray(frame, GrayscaleSettings(mode="luma")))
    red_std, red_range = measure_roi(to_gray(frame, GrayscaleSettings(mode="red")))
    assert red_std > luma_std, f"red std {red_std:.1f} vs luma {luma_std:.1f}"
    assert red_range > luma_range
    print(f"OK test_measurement_actually_moves_with_the_setting: "
          f"std {luma_std:.1f} -> {red_std:.1f}, range {luma_range:.0f} -> {red_range:.0f}")


def test_inspection_delegates_to_the_settings():
    frame = green_board_with_part()
    settings = GrayscaleSettings(mode="red", gamma=1.4)
    assert np.array_equal(inspect_to_gray(frame, settings), to_gray(frame, settings))
    assert np.array_equal(inspect_to_gray(frame), to_gray(frame))
    print("OK test_inspection_delegates_to_the_settings")


def test_grey_input_passes_through():
    gray_in = np.full((20, 20), 128, dtype=np.uint8)
    assert np.array_equal(to_gray(gray_in, GrayscaleSettings(mode="red")), gray_in)
    print("OK test_grey_input_passes_through")


def test_roundtrip_and_bad_file():
    tmp = Path(tempfile.mkdtemp()) / "nested" / "grayscale.json"
    settings = GrayscaleSettings(mode="red", gamma=1.8, contrast=1.3, brightness=-12)
    save_grayscale_settings(str(tmp), settings)
    back = load_grayscale_settings(str(tmp))
    assert back == settings, back

    assert load_grayscale_settings(str(tmp.parent / "missing.json")) == GrayscaleSettings()
    tmp.write_text("{ not json")
    assert load_grayscale_settings(str(tmp)) == GrayscaleSettings(), "bad file must fall back to defaults"

    assert GrayscaleSettings.from_dict({"mode": "ultraviolet"}).mode == "luma", \
        "an unknown mode must not be trusted into to_gray"
    assert GrayscaleSettings.from_dict(None) == GrayscaleSettings()
    print("OK test_roundtrip_and_bad_file")


def test_summary_reads_like_a_setting():
    assert GrayscaleSettings().summary() == "Luma (standard)"
    text = GrayscaleSettings(mode="red", gamma=1.5, brightness=10).summary()
    assert "Red" in text and "gamma 1.50" in text and "+10" in text, text
    print("OK test_summary_reads_like_a_setting:", text)


if __name__ == "__main__":
    test_every_mode_returns_a_single_channel()
    test_channel_choice_changes_contrast_on_a_green_board()
    test_defaults_match_plain_luma()
    test_gamma_lifts_shadows_and_contrast_stretches()
    test_lut_is_256_entries_and_clamped()
    test_extreme_settings_do_not_crash_or_wrap()
    test_measurement_actually_moves_with_the_setting()
    test_inspection_delegates_to_the_settings()
    test_grey_input_passes_through()
    test_roundtrip_and_bad_file()
    test_summary_reads_like_a_setting()
    print("\nALL GRAYSCALE TESTS PASSED")
