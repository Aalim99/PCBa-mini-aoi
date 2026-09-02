# PCB Inspection Station

A standalone visual inspection station: place a PCB under a camera,
trigger a single-frame capture, and get a PASS/FAIL verdict on missing
components. Not an AOI machine — a small, self-contained bench station.

## Running

```bash
pip install -r requirements.txt
python main.py
```

## Setting up a board (once per board model)

In **Program Manager**:

1. **Import XY…** — parse the mounter's pick-and-place export into a
   program: components, fiducials, and panel offsets.
2. **Load Reference…** — a photo of a known-good board, aligned to the
   program's fiducials. Everything below depends on it.
3. **Alignment fiducials** — define **F1/F2/F3**, the three points the
   station aligns every board from. *Auto-suggest* picks a well-spread
   trio from the XY file; the dropdown chooses a specific one; **Pick**
   places a point anywhere on the reference photo. Then **Teach from
   Reference** stores what each mark looks like. The board overview
   draws the marks and the triangle they form — a long thin triangle
   pins rotation poorly, which a list of coordinates won't show you.
4. **ROI size** — select a part number and size its box by dragging a
   corner or typing millimetres. With a reference loaded, the real
   component sits behind the box at true scale, de-rotated, and
   *Prev/Next* steps through its other placements (all panel units
   included). Sizes live in a shared `programs/part_sizes.json` that
   carries across boards, since the same part number reappears.
5. **Delete** what shouldn't be inspected — a whole part number, or
   individual designators (filter box for finding them). Test points,
   mechanical parts and build-option parts left in the program mean a
   good board keeps failing, which teaches the operator to ignore the
   verdict.
6. **Set Active for Inspection**.

## Inspecting

In **Live Inspection**: pick the camera (the dropdown lists what's
actually attached) or **Open Image…**, press **Align Board**, then
**INSPECT** (or Space) for each board. The view holds the annotated
result until *Resume Live*.

Alignment tries the taught fiducials first, falls back to blob
detection, then to clicking each fiducial by name. The panel says which
path ran, because that changes how much to trust a marginal call.

### When it calls a good part missing

Two controls, for two different problems:

- **Sensitivity slider** — the whole board is over- or under-calling.
- **"This part IS present"** — one part number keeps false-calling.
  Select the finding and press it; the threshold for that part number
  moves to just under what it actually measured.

Both re-decide the capture already on screen, from measurements already
taken — so you see the effect on the board that prompted the change, with
no re-shoot. Each finding shows how close the call was (`0.62x` means it
reached 62% of what was required); hover for the numbers. **Save Tuning**
persists it to `programs/part_thresholds.json`.

**Logs / History** shows every pass with a yield summary, filterable by
verdict and free text, from `logs/results.csv`.

## Layout

| Path | What it does |
| --- | --- |
| `core/program_parser.py` | Mounter XY (Excel) export → program JSON |
| `core/fiducials.py` | Named F1/F2/F3, template teaching, geometry-verified alignment |
| `core/calibration.py` | Blob fiducial detection, correspondence matching, mm→px homography |
| `core/inspection.py` | ROI projection, presence check, panel handling, verdicts |
| `core/thresholds.py` | Sensitivity and per-part thresholds for false-call tuning |
| `core/program_edit.py` | Delete designators / part numbers |
| `core/reference_image.py` | Aligned reference photo, mm-accurate component patches |
| `core/barcode_reader.py` | Traceability code from the inspection frame |
| `core/result_log.py` | Append/read the results CSV |
| `core/camera.py` | Camera with backend fallback, plus a still-image stand-in |
| `core/testutils.py` | Synthetic board frames (dev/test only) |
| `ui/theme.py` | Dark theme and shared widgets |
| `ui/` | The three tabs, the fiducial panel, the alignment widget |

Each UI module runs standalone for testing, e.g. `python ui/live_tab.py`.

## Tests

```bash
QT_QPA_PLATFORM=offscreen python tests/run_all.py     # 12 suites
QT_QPA_PLATFORM=offscreen python scripts/shoot_ui.py  # screenshots of every tab
```

Everything is verified against synthetic board images, so the pipeline is
testable without a camera or a physical board.

## Known limitations

Worth knowing before trusting a verdict:

- **Presence thresholds are placeholders.** The defaults (grayscale std
  dev and intensity range) have never been tuned against a real board.
  The sensitivity slider and per-part overrides exist to do that tuning
  from real captures — until then, PASS/FAIL means little. The heuristic
  itself is deliberately simple and is not a trained model.
- **Alignment quality with 3 fiducials.** A 3-point fit passes through
  every point by construction, so its RMS is always zero and is not
  reported as accuracy; the template match score is shown instead. Four
  or more points give a meaningful RMS.
- **Taught templates assume the station doesn't change.** They are
  matched across a range of scales, but re-teach after moving the
  camera, changing the lens, or changing the lighting.
- **Panel offset semantics are auto-detected.** A mounter file may list
  every unit's components absolutely, or list one unit for the Pattern
  Offsets to repeat. Which one is inferred from the data; set
  `"panel_mode": "expanded"` or `"replicate"` in the program JSON to pin
  it once the real board's file settles the question.
- **Dust/dirt detection is not built** — planned as anomaly detection
  outside the known component ROIs.
