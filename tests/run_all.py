"""Run every test suite. The GUI suites need an offscreen Qt platform
when there's no display:

    QT_QPA_PLATFORM=offscreen python tests/run_all.py
"""
import subprocess
import sys
from pathlib import Path

TESTS = [
    "test_calibration.py",
    "test_inspection.py",
    "test_barcode_and_log.py",
    "test_fiducials.py",
    "test_tuning_and_editing.py",
    "test_reference_image.py",
    "test_program_tab_smoke.py",
    "test_program_tab_reference_smoke.py",
    "test_calibration_widget_smoke.py",
    "test_live_tab_smoke.py",
    "test_ui_features_smoke.py",
    "test_main_integration.py",
]

here = Path(__file__).resolve().parent
failed = []

for name in TESTS:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    proc = subprocess.run([sys.executable, str(here / name)], capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stdout.write(proc.stderr)
        failed.append(name)

print(f"\n{'=' * 60}")
if failed:
    print(f"FAILED ({len(failed)}/{len(TESTS)}): {', '.join(failed)}")
    sys.exit(1)
print(f"ALL {len(TESTS)} SUITES PASSED")
