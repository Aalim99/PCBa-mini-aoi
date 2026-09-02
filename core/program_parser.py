"""
program_parser.py

Parses a pick-and-place (XY) file exported from the SMT mounter into a
structured "program" definition used by the PCB inspection app.

Handles:
  - locating the real header row (mounter exports usually have a title
    row above the actual column headers)
  - splitting rows by their `Type` column into:
        Placement          -> components to inspect
        Pattern Fiducial   -> alignment reference points
        Pattern Offset     -> panel repeat positions (if the file is a
                               panel of multiple identical board units)
  - filtering out junk/marker rows (e.g. a placeholder row with a
    dashed Designator and no Part/Library)

Output is a plain dict, ready to be saved as programs/<name>.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime


REQUIRED_COLUMNS = {"X", "Y", "Rotation", "Designator", "Library", "Part", "Type"}


def _find_header_row(raw: pd.DataFrame, max_scan_rows: int = 10) -> int:
    """Scan the first few rows to find the one that contains the real
    column headers (mounter exports often have a title row first)."""
    for i in range(min(max_scan_rows, len(raw))):
        row_values = set(str(v).strip() for v in raw.iloc[i].tolist())
        if {"X", "Y", "Designator", "Type"}.issubset(row_values):
            return i
    raise ValueError(
        "Could not locate header row (expected columns like X, Y, "
        "Designator, Type within the first rows of the sheet)."
    )


def _is_junk_row(row: pd.Series) -> bool:
    """Flags placeholder/marker rows that aren't real components.
    e.g. Designator made only of dashes/punctuation, or empty Part
    and Library on a 'Placement' row."""
    designator = str(row.get("Designator", "")).strip()
    if designator == "" or designator.lower() == "nan":
        return True
    if all(ch in "-_= " for ch in designator):
        return True
    return False


def load_mounter_xy(filepath: str) -> pd.DataFrame:
    """Load the raw XY export and return a clean DataFrame with the
    real header row applied."""
    raw = pd.read_excel(filepath, header=None)
    header_row = _find_header_row(raw)
    df = pd.read_excel(filepath, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"XY file is missing expected columns: {missing}")

    # drop fully blank rows and rows where Type itself is blank/garbage
    df = df.dropna(subset=["Type"])
    df = df[df["Type"] != "Type"]  # stray repeated header row, if any
    return df


def parse_program(filepath: str, program_name: str, is_panel: bool = None) -> dict:
    """Parse a mounter XY export into a structured program dict.

    Args:
        filepath: path to the .xlsx XY export
        program_name: name to store this program under
        is_panel: True/False if known; if None, inferred from whether
                  any 'Pattern Offset' rows exist

    Returns:
        dict with keys: name, created, is_panel, fiducials,
        panel_offsets, components, unknown_parts
    """
    df = load_mounter_xy(filepath)

    placements = df[df["Type"] == "Placement"].copy()
    fiducials = df[df["Type"] == "Pattern Fiducial"].copy()
    offsets = df[df["Type"] == "Pattern Offset"].copy()

    # filter junk rows out of placements only (fiducials/offsets don't
    # carry a meaningful Designator anyway)
    placements = placements[~placements.apply(_is_junk_row, axis=1)]

    if is_panel is None:
        is_panel = len(offsets) > 0

    components = []
    unknown_parts = set()
    for _, r in placements.iterrows():
        part = None if pd.isna(r.get("Part")) else str(r["Part"]).strip()
        if part:
            unknown_parts.add(part)
        components.append({
            "designator": str(r["Designator"]).strip(),
            "x": float(r["X"]),
            "y": float(r["Y"]),
            "rotation": float(r["Rotation"]) if not pd.isna(r["Rotation"]) else 0.0,
            "library": None if pd.isna(r.get("Library")) else str(r["Library"]).strip(),
            "part": part,
        })

    fiducial_list = [
        {"x": float(r["X"]), "y": float(r["Y"])}
        for _, r in fiducials.iterrows()
    ]

    panel_offsets = [
        {
            "label": str(r["Designator"]).strip() if not pd.isna(r.get("Designator")) else None,
            "dx": float(r["X"]),
            "dy": float(r["Y"]),
        }
        for _, r in offsets.iterrows()
    ]

    program = {
        "name": program_name,
        "source_file": Path(filepath).name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "is_panel": bool(is_panel),
        "fiducials": fiducial_list,
        "panel_offsets": panel_offsets,
        "components": components,
        "unknown_parts": sorted(unknown_parts),  # parts not yet sized in the lookup table
    }
    return program


def save_program(program: dict, programs_dir: str) -> str:
    Path(programs_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(programs_dir) / f"{program['name']}.json"
    with open(out_path, "w") as f:
        json.dump(program, f, indent=2)
    return str(out_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python program_parser.py <xy_file.xlsx> <program_name>")
        sys.exit(1)

    program = parse_program(sys.argv[1], sys.argv[2])
    out = save_program(program, "programs")

    print(f"Parsed program saved to: {out}")
    print(f"  Panel: {program['is_panel']}")
    print(f"  Components: {len(program['components'])}")
    print(f"  Fiducials: {len(program['fiducials'])}")
    print(f"  Panel offsets: {len(program['panel_offsets'])}")
    print(f"  Unique part numbers needing size lookup: {len(program['unknown_parts'])}")
