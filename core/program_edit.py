"""
program_edit.py

Editing a parsed program: removing components that should not be
inspected, and saving the result back.

Not everything a mounter places is worth checking -- test points,
mechanical hardware, parts that vary by build option, or a designator
whose ROI simply cannot be made reliable. Leaving those in means a
board that is fine keeps coming back FAIL, which trains the operator to
ignore the verdict. Removing them is the honest fix.

Removal is by designator or by part number, and always reports what it
took out, so the caller can tell the operator rather than silently
shrinking their program.
"""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def part_summary(program: dict) -> Dict[str, int]:
    """Component count per part number, including an UNSPECIFIED bucket
    for rows that carry no part."""
    counts: Dict[str, int] = {}
    for c in program.get("components") or []:
        key = c.get("part") or "UNSPECIFIED"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _refresh_unknown_parts(program: dict) -> None:
    """Keep the parts-needing-a-size list in step with what remains."""
    remaining = sorted({c["part"] for c in (program.get("components") or []) if c.get("part")})
    program["unknown_parts"] = remaining


def delete_designators(program: dict, designators: Iterable[str]) -> List[dict]:
    """Remove components by designator. Returns the removed rows.

    On a panel in replicate mode a designator names one placement in the
    repeating unit, so removing it removes that component from every
    unit -- which is what the operator means by "don't inspect R47".
    """
    targets = {str(d).strip() for d in designators if str(d).strip()}
    if not targets:
        return []
    keep, removed = [], []
    for c in program.get("components") or []:
        (removed if str(c.get("designator", "")).strip() in targets else keep).append(c)
    program["components"] = keep
    _refresh_unknown_parts(program)
    return removed


def delete_part(program: dict, part: Optional[str]) -> List[dict]:
    """Remove every component of one part number. `part` may be
    "UNSPECIFIED" to clear rows that carry no part number at all."""
    keep, removed = [], []
    for c in program.get("components") or []:
        key = c.get("part") or "UNSPECIFIED"
        (removed if key == part else keep).append(c)
    program["components"] = keep
    _refresh_unknown_parts(program)
    return removed


def save_program_json(program: dict, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(program, indent=2))
    return str(path)


def describe_removal(removed: List[dict]) -> Tuple[int, str]:
    """(count, short human summary) for a confirmation or status line."""
    if not removed:
        return 0, "nothing removed"
    names = [str(c.get("designator", "?")) for c in removed]
    shown = ", ".join(names[:6])
    if len(names) > 6:
        shown += f", +{len(names) - 6} more"
    return len(removed), shown
