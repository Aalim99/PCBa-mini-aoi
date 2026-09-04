"""
result_log.py

Appends inspection results to a local CSV (no database) and reads them
back for the Logs/History tab.

One row per inspection pass. Per-unit verdicts and the missing-component
list are flattened into single cells so the file stays a plain,
spreadsheet-openable CSV; `parse_missing` and `parse_units` turn them
back into structured data for display.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

COLUMNS = [
    "timestamp",
    "barcode",
    "program",
    "verdict",
    "units_total",
    "units_failed",
    "checked_count",
    "missing_count",
    "unchecked_count",
    "missing",        # "U1:R5|U2:C3"
    "unit_verdicts",  # "U1=PASS|U2=FAIL"
    "message",
]

LIST_SEP = "|"
PAIR_SEP = ":"


def _fmt_missing(result) -> str:
    # c.label carries the unit and, where a designator repeats within it,
    # the board position that tells two placements apart.
    return LIST_SEP.join(c.label for c in result.missing)


def _fmt_unit_verdicts(result) -> str:
    return LIST_SEP.join(f"{u.label}={'PASS' if u.passed else 'FAIL'}" for u in result.units)


def append_result(csv_path: str, result, timestamp: Optional[datetime] = None) -> str:
    """Append one InspectionResult. Creates the file with a header row
    on first use. Returns the path written."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0

    row = {
        "timestamp": (timestamp or datetime.now()).isoformat(timespec="seconds"),
        "barcode": result.barcode or "",
        "program": result.program_name,
        "verdict": result.verdict,
        "units_total": len(result.units),
        "units_failed": sum(1 for u in result.units if not u.passed),
        "checked_count": result.checked_count,
        "missing_count": len(result.missing),
        "unchecked_count": len(result.unchecked),
        "missing": _fmt_missing(result),
        "unit_verdicts": _fmt_unit_verdicts(result),
        "message": result.message,
    }

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return str(path)


def read_results(csv_path: str) -> List[Dict[str, str]]:
    """All logged rows, oldest first. Missing file -> empty list."""
    path = Path(csv_path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_results(rows: List[Dict[str, str]], verdict: Optional[str] = None,
                   barcode: Optional[str] = None, program: Optional[str] = None,
                   text: Optional[str] = None) -> List[Dict[str, str]]:
    """Filter logged rows. All criteria are optional and combine with
    AND; `text` is a case-insensitive substring match over every field."""
    out = rows
    if verdict:
        out = [r for r in out if r.get("verdict") == verdict]
    if barcode:
        out = [r for r in out if barcode.lower() in (r.get("barcode") or "").lower()]
    if program:
        out = [r for r in out if r.get("program") == program]
    if text:
        needle = text.lower()
        out = [r for r in out if any(needle in str(v).lower() for v in r.values())]
    return out


def parse_missing(cell: str) -> List[tuple]:
    """"U1:R5|U2:C3" -> [("U1", "R5"), ("U2", "C3")]"""
    if not cell:
        return []
    pairs = []
    for chunk in cell.split(LIST_SEP):
        unit, _, designator = chunk.partition(PAIR_SEP)
        if designator:
            pairs.append((unit, designator))
    return pairs


def parse_units(cell: str) -> List[tuple]:
    """"U1=PASS|U2=FAIL" -> [("U1", "PASS"), ("U2", "FAIL")]"""
    if not cell:
        return []
    pairs = []
    for chunk in cell.split(LIST_SEP):
        label, _, verdict = chunk.partition("=")
        if verdict:
            pairs.append((label, verdict))
    return pairs
