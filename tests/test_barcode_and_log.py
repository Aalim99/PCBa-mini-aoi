"""Automated tests for core/barcode_reader.py and core/result_log.py.
Run directly:
    python tests/test_barcode_and_log.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.barcode_reader import read_barcode, read_barcodes
from core.result_log import (
    append_result, read_results, filter_results, parse_missing, parse_units, COLUMNS,
)
from core.inspection import InspectionResult, UnitResult, ComponentResult


def make_qr_frame(text, scale=8, pad=40, background=200):
    """A QR code rendered onto a larger frame, standing in for a label
    in the camera's field of view."""
    enc = cv2.QRCodeEncoder.create()
    code = enc.encode(text)
    code = cv2.resize(code, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    h, w = code.shape[:2]
    frame = np.full((h + 2 * pad, w + 2 * pad), background, dtype=np.uint8)
    frame[pad:pad + h, pad:pad + w] = code
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_qr_roundtrip():
    frame = make_qr_frame("PCB-SN-00427")
    assert read_barcode(frame) == "PCB-SN-00427", read_barcodes(frame)
    print("OK test_qr_roundtrip:", read_barcode(frame))


def test_no_barcode_returns_none():
    """An unreadable label must not raise -- it logs blank, since a bad
    label is not a board defect."""
    frame = np.full((300, 400, 3), 120, dtype=np.uint8)
    assert read_barcode(frame) is None
    assert read_barcodes(frame) == []
    print("OK test_no_barcode_returns_none")


def test_barcode_on_noisy_frame():
    frame = make_qr_frame("SN-9981-B")
    rng = np.random.default_rng(0)
    noisy = np.clip(frame.astype(np.float64) + rng.normal(0, 8, frame.shape), 0, 255).astype(np.uint8)
    assert read_barcode(noisy) == "SN-9981-B"
    print("OK test_barcode_on_noisy_frame")


def test_empty_frame_is_safe():
    assert read_barcodes(None) == []
    assert read_barcodes(np.array([], dtype=np.uint8)) == []
    print("OK test_empty_frame_is_safe")


def _fake_result(verdict="FAIL"):
    u1 = UnitResult(label="U1", components=[
        ComponentResult(designator="R1", part="PN-1", unit="U1", x_mm=1, y_mm=1, present=True),
    ])
    u2 = UnitResult(label="U2", components=[
        ComponentResult(designator="C3", part="PN-2", unit="U2", x_mm=2, y_mm=2, present=False),
        ComponentResult(designator="R7", part="PN-1", unit="U2", x_mm=3, y_mm=3, present=False),
    ])
    return InspectionResult(verdict=verdict, units=[u1, u2], barcode="SN-123",
                            program_name="BOARD_A", message="2 missing")


def test_append_and_read():
    tmp = Path(tempfile.mkdtemp()) / "logs" / "results.csv"
    append_result(str(tmp), _fake_result())
    append_result(str(tmp), _fake_result(verdict="PASS"))
    rows = read_results(str(tmp))
    assert len(rows) == 2, rows
    assert list(rows[0].keys()) == COLUMNS, rows[0].keys()
    assert rows[0]["verdict"] == "FAIL"
    assert rows[0]["missing_count"] == "2"
    assert rows[0]["units_failed"] == "1"
    assert rows[1]["verdict"] == "PASS"
    print("OK test_append_and_read:", len(rows), "rows,", rows[0]["missing"])


def test_missing_and_unit_cells_roundtrip():
    tmp = Path(tempfile.mkdtemp()) / "results.csv"
    append_result(str(tmp), _fake_result())
    row = read_results(str(tmp))[0]
    assert parse_missing(row["missing"]) == [("U2", "C3"), ("U2", "R7")], row["missing"]
    assert parse_units(row["unit_verdicts"]) == [("U1", "PASS"), ("U2", "FAIL")], row["unit_verdicts"]
    print("OK test_missing_and_unit_cells_roundtrip:", row["unit_verdicts"])


def test_filters():
    tmp = Path(tempfile.mkdtemp()) / "results.csv"
    append_result(str(tmp), _fake_result(verdict="FAIL"))
    append_result(str(tmp), _fake_result(verdict="PASS"))
    rows = read_results(str(tmp))
    assert len(filter_results(rows, verdict="PASS")) == 1
    assert len(filter_results(rows, barcode="sn-123")) == 2
    assert len(filter_results(rows, program="BOARD_A")) == 2
    assert len(filter_results(rows, program="NOPE")) == 0
    assert len(filter_results(rows, text="C3")) == 2
    assert len(filter_results(rows, verdict="PASS", text="C3")) == 1
    print("OK test_filters")


def test_read_missing_file_is_empty():
    assert read_results(str(Path(tempfile.mkdtemp()) / "nope.csv")) == []
    print("OK test_read_missing_file_is_empty")


def test_csv_survives_commas_in_fields():
    """Message text contains commas -- the reader must not shear columns."""
    tmp = Path(tempfile.mkdtemp()) / "results.csv"
    result = _fake_result()
    result.message = "missing: C3, R7, and more"
    append_result(str(tmp), result)
    row = read_results(str(tmp))[0]
    assert row["message"] == "missing: C3, R7, and more", row
    assert row["verdict"] == "FAIL"
    print("OK test_csv_survives_commas_in_fields")


if __name__ == "__main__":
    test_qr_roundtrip()
    test_no_barcode_returns_none()
    test_barcode_on_noisy_frame()
    test_empty_frame_is_safe()
    test_append_and_read()
    test_missing_and_unit_cells_roundtrip()
    test_filters()
    test_read_missing_file_is_empty()
    test_csv_survives_commas_in_fields()
    print("\nALL BARCODE + LOG TESTS PASSED")
