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
  - dropping what cannot be inspected: the export's own section
    banners and rules, placements carrying neither a part number nor a
    library (they could never be given an ROI size), and rows the
    mounter is told to skip. Every drop is counted in `notes`.

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
OPTIONAL_COLUMNS = {"Type", "Rotation", "Part", "Library", "Skip"}

# Values a Skip column may hold. Anything outside this vocabulary means
# the column is not the yes/no flag we take it for (a "Skip Number", say),
# and it is then ignored rather than guessed at.
SKIP_TRUE = {"1", "y", "yes", "true", "skip", "x", "dnp"}
SKIP_FALSE = {"0", "n", "no", "false", "place", "", "nan", "none"}

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
    "skip": "Skip", "skipno": "Skip", "skipped": "Skip", "noplace": "Skip",
    "donotplace": "Skip", "dnp": "Skip",
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
    """Flags marker rows that aren't components, by designator alone.

    Mounter exports interleave section markers with the placements --
    a rule of dashes, or a bracketed banner such as "[PLACEMENT ITEM
    ...]" that the export wrapped across several rows. A real reference
    designator never opens with a bracket or is made only of rules.
    """
    designator = str(row.get("Designator", "")).strip()
    if designator == "" or designator.lower() == "nan":
        return True
    if all(ch in "-_=*. " for ch in designator):
        return True
    if designator[0] in "[<{#":
        return True
    return False


def _is_unidentifiable(row: pd.Series) -> bool:
    """A placement carrying neither a part number nor a library.

    ROI sizes are keyed on the part number, so such a row can never be
    sized and therefore never inspected: it would sit in "unchecked" for
    the life of the program and hold every verdict at INCOMPLETE. In
    practice these are the export's own leftovers -- the second line of a
    wrapped banner, or the board's F1/F2/F3 marks listed among the
    placements. Only applied when the file does record part numbers;
    see _records_parts.
    """
    for field in ("Part", "Library"):
        value = row.get(field)
        if not pd.isna(value) and str(value).strip():
            return False
    return True


def _records_parts(placements: pd.DataFrame, threshold: float = 0.5) -> bool:
    """Whether this file identifies its placements at all.

    Plenty of exports are a bare X/Y/Designator list with no part
    number anywhere; dropping unidentifiable rows there would empty the
    program. So the rule only applies to a file where most placements do
    carry an identity and the few that don't stand out as leftovers.
    """
    columns = [c for c in ("Part", "Library") if c in placements.columns]
    if not columns or placements.empty:
        return False
    identified = ~placements.apply(_is_unidentifiable, axis=1)
    return bool(identified.mean() >= threshold)


def _skip_token(value) -> str:
    """Canonical form of a Skip cell. Numbers are normalised first: a
    spreadsheet hands back 0.0 where the file says 0, and "00" is not a
    value any yes/no vocabulary contains."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        if float(value) == int(value):
            return str(int(value))
    return _canonical(value)


def _skip_flags(placements: pd.DataFrame):
    """(mask of rows the mounter is told not to place, note or None).

    A part that was never placed is missing by design; inspecting it
    fails every board and teaches the operator to ignore the verdict.
    """
    if "Skip" not in placements.columns or placements.empty:
        return None, None
    values = placements["Skip"].map(_skip_token)
    unknown = sorted({v for v in values.unique() if v not in SKIP_TRUE | SKIP_FALSE})
    if unknown:
        return None, ("A Skip column was found but holds values this app does not "
                      f"recognise as yes/no ({', '.join(unknown[:4])}); it was ignored, "
                      "so any not-placed parts will be reported missing.")
    return values.isin(SKIP_TRUE), None


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

    # Filter placements only; fiducials/offsets don't carry a meaningful
    # Designator anyway. Each drop is counted and reported in notes --
    # a component that quietly vanishes from a program is worse than one
    # that is wrongly present, because nothing downstream can show it.
    if not placements.empty:
        before = len(placements)
        placements = placements[~placements.apply(_is_junk_row, axis=1)]
        markers = before - len(placements)
        if markers:
            notes.append(f"Skipped {markers} marker row(s) from the export "
                         "(section banners and rules, not components).")

    if not placements.empty and _records_parts(placements):
        before = len(placements)
        dropped = placements[placements.apply(_is_unidentifiable, axis=1)]
        placements = placements.drop(dropped.index)
        if before - len(placements):
            examples = ", ".join(sorted({str(d).strip() for d in dropped["Designator"]})[:5])
            notes.append(f"Skipped {before - len(placements)} row(s) with no part number "
                         f"or library ({examples}) - they cannot be sized, so they could "
                         "never be inspected.")

    if not placements.empty:
        skip_mask, skip_note = _skip_flags(placements)
        if skip_note:
            notes.append(skip_note)
        elif skip_mask is not None and skip_mask.any():
            notes.append(f"Skipped {int(skip_mask.sum())} placement(s) the mounter is "
                         "told not to place (Skip column).")
            placements = placements[~skip_mask]

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
