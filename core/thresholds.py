"""
thresholds.py

Presence-decision tuning: the global sensitivity the operator sets on
the Live tab, and per-part-number overrides learned from false calls.

The heuristic (variance + intensity range inside the ROI) has no single
right threshold -- it depends on lighting, solder mask colour and how a
particular part reflects. So the numbers are treated as tunable data,
stored beside part_sizes.json and shared across programs the same way,
since the same part number reappears on different boards.

Two knobs, deliberately distinct:

  sensitivity  a single global multiplier, for "the whole board is
               calling parts missing that are there". Higher demands
               more texture before calling a part present.
  per-part     an override for one part number, for "this ONE part
               keeps false-calling". Set by pointing at a false call
               rather than by typing numbers.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# A false call is fixed by moving the threshold just below what that
# part actually measured, with margin so the next board is not borderline.
FALSE_CALL_MARGIN = 0.80
# Wide, because the defaults are placeholders: until they are tuned on
# real boards the right threshold may be far from 1x in either
# direction, and an operator chasing false calls needs to be able to
# get there with the slider rather than by editing JSON.
MIN_SENSITIVITY, MAX_SENSITIVITY = 0.1, 5.0


def load_part_thresholds(path: str) -> Dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_part_thresholds(path: str, thresholds: Dict[str, dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(thresholds, indent=2, sort_keys=True))


def effective_thresholds(part: Optional[str], base_std: float, base_range: float,
                         part_thresholds: Optional[Dict[str, dict]] = None,
                         sensitivity: float = 1.0) -> Tuple[float, float]:
    """The thresholds actually applied to one component.

    A per-part override replaces the base value; sensitivity then scales
    whatever is in force, so the global control still works on parts
    that have been individually tuned.
    """
    std_min, range_min = base_std, base_range
    override = (part_thresholds or {}).get(part) if part else None
    if override:
        std_min = float(override.get("std_min", std_min))
        range_min = float(override.get("range_min", range_min))
    return std_min * sensitivity, range_min * sensitivity


def thresholds_for_false_call(std: float, intensity_range: float,
                              margin: float = FALSE_CALL_MARGIN,
                              sensitivity: float = 1.0) -> dict:
    """Thresholds that would have accepted a component the operator says
    is actually present. Kept a little under the measured values so an
    identical part next time is not decided on the knife edge.

    Divided by the sensitivity currently in force, because sensitivity
    is applied on top of per-part overrides: without this, marking a
    false call while the slider sits above 1x would store a value that
    the slider then scales straight back past the measurement, and the
    part would keep failing however often the operator accepted it.
    """
    scale = max(float(sensitivity), 1e-6)
    return {
        "std_min": round(max(float(std) * margin / scale, 0.01), 3),
        "range_min": round(max(float(intensity_range) * margin / scale, 0.1), 3),
    }


def clamp_sensitivity(value: float) -> float:
    return max(MIN_SENSITIVITY, min(MAX_SENSITIVITY, float(value)))
