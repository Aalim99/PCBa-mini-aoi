# PCB Inspection Station

A standalone visual inspection station: place a PCB under a camera,
trigger a single-frame capture, and get a PASS/FAIL verdict on missing
components. Not an AOI machine — a small, self-contained bench station.

## Running

```bash
pip install -r requirements.txt
python main.py
```

## Session flow

1. **Program Manager** — *Import XY File…* parses a mounter pick-and-place
   export into a program (components, fiducials, panel offsets). Select
   each part number and set its ROI box size by dragging a corner or
   typing exact mm; sizes live in a shared `programs/part_sizes.json`
   that carries across boards, since the same part number reappears.
   Press *Set Active for Inspection*.

   *Load Reference Image…* takes a photo of a known-good board and aligns
   it to the program's fiducials, then draws the real component behind
   the ROI box at true millimetre scale — so a box is sized against the
   actual part instead of guessed. The part is shown de-rotated, so one
   ROI size fits every placement of it, and *Prev/Next* steps through the
   part's other instances (every panel unit included) to check the box
   holds up across the board. The reference is stored with the program
   and reloads with it.
2. **Live Inspection** — start the camera (or load a still image), press
   *Calibrate* to align the board, then *INSPECT* (or Space) for each
   board. The view holds the annotated result until *Resume Live*.
3. **Logs / History** — every pass is appended to `logs/results.csv`,
   filterable by verdict and free text.

## Layout

| Path | What it does |
| --- | --- |
| `core/program_parser.py` | Mounter XY (Excel) export → program JSON |
| `core/calibration.py` | Fiducial detection, correspondence matching, mm→px homography |
| `core/inspection.py` | ROI projection, presence check, panel handling, verdicts |
| `core/barcode_reader.py` | Traceability code from the inspection frame |
| `core/reference_image.py` | Fiducial-aligned reference board photo, mm-accurate component patches |
| `core/result_log.py` | Append/read the results CSV |
| `core/camera.py` | Camera, with a still-image stand-in |
| `core/testutils.py` | Synthetic board frames (dev/test only) |
| `ui/` | The three tabs plus the calibration widget |

Each UI module runs standalone for testing, e.g. `python ui/live_tab.py`.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python tests/run_all.py
```

Everything is verified against synthetic board images, so the pipeline is
testable without a camera or a physical board.

## Known limitations

These are real and worth knowing before trusting a verdict:

- **Presence thresholds are placeholders.** `PresenceThresholds` defaults
  (grayscale std dev and intensity range) have never been tuned against a
  real board. They must be re-tuned from captures of a known-good and a
  known-missing board before the PASS/FAIL means anything. The heuristic
  itself is deliberately simple — variance and intensity range within the
  ROI — and is not a trained model.
- **Fiducial auto-detect can be ambiguous on panels.** A panel repeating
  the same fiducial pattern at a regular pitch can align equally well one
  pitch over. This is detected, not ignored: the calibration reports the
  ambiguity and falls back to manual click-to-calibrate rather than
  returning a confident wrong answer.
- **Panel offset semantics are auto-detected.** A mounter file may list
  every unit's components absolutely, or list one unit for the Pattern
  Offsets to repeat. Which one is inferred from the data; set
  `"panel_mode": "expanded"` or `"replicate"` in the program JSON to pin
  it once the real board's file settles the question.
- **Dust/dirt detection is not built** — planned as anomaly detection
  outside the known component ROIs.
