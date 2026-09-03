"""
program_parser.py

Parses a pick-and-place (XY) file exported from the SMT mounter into a
structured "program" definition used by the PCB inspection app.

Accepts Excel (.xlsx/.xls) and delimited text (.csv/.tsv/.txt) exports,
since which one a mounter produces varies by machine and by how the
operator exported it.

Handles:
  - locating the real header row (mounter exports usually have a title
    row above the actual column headers)
  - resolving column names across dialects, so "RefDes", "Center-X" and
    "Rot" land on Designator, X and Rotation
  - splitting rows by their `Type` column into:
        Placement          -> components to inspect
        Pattern Fiducial   -> alignment reference points
        Pattern Offset     -> panel repeat positions (if the file is a
                               panel of multiple identical board units)
  - filtering out junk/marker rows (e.g. a placeholder row with a
    dashed Designator and no Part/Library)

Only X, Y and Designator are genuinely required: without them there is
nothing to inspect. Type, Rotation, Part and Library are optional and
fall back to sensible defaults, because plenty of exports are a plain
list of placements with no record-type column at all. `parse_program`
reports what it had to assume in the returned dict's `notes`, so the
caller can tell the operator rather than quietly guessing.

Output is a plain dict, ready to be saved as programs/<name>.json
"""

import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TEXT_SUFFIXES = {".csv", ".tsv", ".txt"}

# The minimum needed to inspect anything.
REQUIRED_COLUMNS = {"X", "Y", "Designator"}
# Understood but optional, with defaults applied when absent.
OPTIONAL_COLUMNS = {"Type", "Rotation", "Part", "Library"}

# Column aliases across mounter/CAD dialects. Keys are canonicalised
# (lower case, non-alphanumerics stripped), so "Center-X (mm)" and
# "center x" both arrive as "centerxmm" / "centerx".
COLUMN_ALIASES = {
    "x": "X", "xmm": "X", "posx": "X", "xpos": "X", "positionx": "X",
    "centerx": "X", "centerxmm": "X", "midx": "X", "refx": "X",
    "y": "Y", "ymm": "Y", "posy": "Y", "ypos": "Y", "positiony": "Y",
    "centery": "Y", "centerymm": "Y", "midy": "Y", "refy": "Y",
    "rotation": "Rotation", "rot": "Rotation", "angle": "Rotation",
    "theta": "Rotation", "rotationdeg": "Rotation", "orientation": "Rotation",
    "designator": "Designator", "refdes": "Designator", "ref": "Designator",
    "reference": "Designator", "referencedesignator": "Designator",
    "component": "Designator", "componentname": "Designator", "name": "Designator",
    "library": "Library", "footprint": "Library", "package": "Library",
    "pattern": "Library", "shape": "Library",
    "part": "Part", "partnumber": "Part", "partno": "Part", "partnum": "Part",
    "itemnumber": "Part", "materialnumber": "Part", "feedername": "Part",
    "type": "Type", "recordtype": "Type", "rowtype": "Type",
}

PLACEMENT = "Placement"
PATTERN_FIDUCIAL = "Pattern Fiducial"
PATTERN_OFFSET = "Pattern Offset"


def _canonical(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _resolve_columns(header):
    """Map a header row onto canonical column names, keeping unknown
    columns under their original name so nothing is silently lost."""
    resolved, seen = [], {}
    for raw in header:
        name = COLUMN_ALIASES.get(_canonical(raw), str(raw).strip())
        # Duplicate headers are legal in these exports; suffix rather
        # than let pandas hand back a DataFrame on a single-column lookup.
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        resolved.append(name)
    return resolved


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # A title row above the header often defeats the sniffer; fall
        # back to whichever candidate actually appears most.
        counts = {d: sample.count(d) for d in ",;\t|"}
        best = max(counts, key=counts.get)
        return best if counts[best] else ","


def _read_rows(filepath: str):
    """Every cell of the file as a ragged list of rows.

    Read this way rather than straight into a DataFrame because a title
    row above the header has a different width to the data, which the
    CSV parsers reject outright.
    """
    path = Path(filepath)
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        delimiter = _sniff_delimiter(text[:8192])
        return [row for row in csv.reader(io.StringIO(text), delimiter=delimiter)]
    raw = pd.read_excel(filepath, header=None, dtype=object)
    return raw.values.tolist()


def _find_header_row(rows, max_scan_rows: int = 20) -> int:
    """Index of the row holding the real column headers."""
    for i in range(min(max_scan_rows, len(rows))):
        names = set(_resolve_columns(rows[i]))
        if REQUIRED_COLUMNS.issubset(names):
            return i
    raise ValueError(
        "Could not find a header row containing X, Y and Designator "
        f"(or recognised equivalents) in the first {max_scan_rows} rows. "
        "Check the file is a pick-and-place export."
    )


def _frame_from_rows(rows, header_index: int) -> pd.DataFrame:
    header = _resolve_columns(rows[header_index])
    width = len(header)
    body = []
    for row in rows[header_index + 1:]:
        cells = list(row[:width])
        cells += [None] * (width - len(cells))    # ragged short rows
        body.append(cells)
    df = pd.DataFrame(body, columns=header)

    # Text files give "" for an empty cell where Excel gives NaN; make
    # them agree so the isna() checks below mean one thing.
    return df.map(lambda v: np.nan if (v is None or (isinstance(v, str) and not v.strip())) else v)


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


def _to_float(value, field, designator) -> float:
    """Parse a coordinate, tolerating a decimal comma and a stray unit
    suffix, and naming the offending row when it cannot."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace("mm", "").strip()
    if "," in text and "." not in text:
        text = text.replace(",", ".")     # 12,5 -> 12.5
    try:
        return float(text)
    except ValueError:
        raise ValueError(
            f"{field} is not a number for {designator or 'a row'}: {value!r}"
        ) from None


def load_mounter_xy(filepath: str) -> pd.DataFrame:
    """Load an XY export (Excel or delimited text) and return a clean
    DataFrame with canonical column names."""
    rows = _read_rows(filepath)
    if not rows:
        raise ValueError(f"{Path(filepath).name} is empty.")

    df = _frame_from_rows(rows, _find_header_row(rows))

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"XY file is missing required column(s): {', '.join(sorted(missing))}. "
            f"Columns found: {', '.join(str(c) for c in df.columns)}"
        )

    if "Type" in df.columns:
        df = df.dropna(subset=["Type"])
        df = df[df["Type"] != "Type"]     # stray repeated header row, if any
    # Rows with no coordinates are separators, not placements.
    df = df.dropna(subset=["X", "Y"], how="any")
    return df


def parse_program(filepath: str, program_name: str, is_panel: bool = None) -> dict:
    """Parse a mounter XY export into a structured program dict.

    Args:
        filepath: path to the .xlsx/.xls/.csv/.tsv XY export
        program_name: name to store this program under
        is_panel: True/False if known; if None, inferred from whether
                  any 'Pattern Offset' rows exist

    Returns:
        dict with keys: name, created, is_panel, fiducials,
        panel_offsets, components, unknown_parts, notes
    """
    df = load_mounter_xy(filepath)
    notes = []

    if "Type" in df.columns:
        types = df["Type"].astype(str).str.strip()
        placements = df[types.str.casefold() == PLACEMENT.casefold()].copy()
        fiducials = df[types.str.casefold() == PATTERN_FIDUCIAL.casefold()].copy()
        offsets = df[types.str.casefold() == PATTERN_OFFSET.casefold()].copy()
        if placements.empty and not df.empty:
            # A Type column whose values this parser doesn't recognise:
            # inspect everything rather than produce an empty program.
            seen = sorted({t for t in types.unique() if t and t != "nan"})[:6]
            notes.append(f"No rows marked '{PLACEMENT}' (saw: {', '.join(seen)}); "
                         "treating every row as a component.")
            placements, fiducials, offsets = df.copy(), df.iloc[0:0], df.iloc[0:0]
    else:
        notes.append("No Type column, so every row is treated as a component "
                     "and no fiducials were found. Define F1/F2/F3 by picking "
                     "them on a reference image.")
        placements, fiducials, offsets = df.copy(), df.iloc[0:0], df.iloc[0:0]

    # filter junk rows out of placements only (fiducials/offsets don't
    # carry a meaningful Designator anyway)
    if not placements.empty:
        placements = placements[~placements.apply(_is_junk_row, axis=1)]

    if is_panel is None:
        is_panel = len(offsets) > 0

    has_part = "Part" in df.columns
    has_rotation = "Rotation" in df.columns
    if not has_part:
        notes.append("No Part column, so components carry no part number. "
                     "ROI sizes are set per part number, so tell the app which "
                     "column holds it, or sizes cannot be shared across boards.")
    if not has_rotation:
        notes.append("No Rotation column; every component is treated as 0 degrees.")

    components = []
    unknown_parts = set()
    for _, r in placements.iterrows():
        designator = str(r["Designator"]).strip()
        part = None
        if has_part and not pd.isna(r.get("Part")):
            part = str(r["Part"]).strip() or None
        if part:
            unknown_parts.add(part)
        rotation = 0.0
        if has_rotation and not pd.isna(r.get("Rotation")):
            rotation = _to_float(r["Rotation"], "Rotation", designator)
        components.append({
            "designator": designator,
            "x": _to_float(r["X"], "X", designator),
            "y": _to_float(r["Y"], "Y", designator),
            "rotation": rotation,
            "library": (None if not ("Library" in df.columns) or pd.isna(r.get("Library"))
                        else str(r["Library"]).strip()),
            "part": part,
        })

    fiducial_list = [
        {"x": _to_float(r["X"], "X", "a fiducial"), "y": _to_float(r["Y"], "Y", "a fiducial")}
        for _, r in fiducials.iterrows()
    ]

    panel_offsets = [
        {
            "label": (str(r["Designator"]).strip()
                      if not pd.isna(r.get("Designator")) else None),
            "dx": _to_float(r["X"], "X", "a panel offset"),
            "dy": _to_float(r["Y"], "Y", "a panel offset"),
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
        "notes": notes,                          # what the parser had to assume
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
        print("Usage: python program_parser.py <xy_file.xlsx|.csv> <program_name>")
        sys.exit(1)

    program = parse_program(sys.argv[1], sys.argv[2])
    out = save_program(program, "programs")

    print(f"Parsed program saved to: {out}")
    print(f"  Panel: {program['is_panel']}")
    print(f"  Components: {len(program['components'])}")
    print(f"  Fiducials: {len(program['fiducials'])}")
    print(f"  Panel offsets: {len(program['panel_offsets'])}")
    print(f"  Unique part numbers needing size lookup: {len(program['unknown_parts'])}")
    for note in program["notes"]:
        print(f"  NOTE: {note}")
