"""Report what each attached camera can actually deliver.

Run this when the live view is blank or will not start:

    python scripts/camera_probe.py

For every camera index it tries each platform backend and a range of
pixel formats and resolutions, and says whether the frames that come
back contain a picture or just a flat colour. Sample frames are saved
next to the report so they can be looked at.

A "blank" result means the camera opened and returned data, but the data
decodes to a single colour -- almost always a format the driver could not
actually supply (an uncompressed stream beyond the USB link's bandwidth
is the usual cause, which is why MJPG at a lower resolution often works).
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core.camera import (
    WARMUP_FRAMES, backend_candidates, backend_name, fourcc_of,
    frame_is_blank, normalise_frame, _quiet_opencv,
)

MODES = [
    (None, None, None),          # whatever the camera defaults to
    ("MJPG", None, None),
    ("MJPG", 1920, 1080),
    ("MJPG", 1280, 720),
    ("MJPG", 640, 480),
    ("YUY2", 1280, 720),
    (None, 1280, 720),
    (None, 640, 480),
]
MAX_INDEX = 6


def describe(fourcc, width, height):
    fmt = fourcc or "default"
    size = f"{width}x{height}" if width else "native"
    return f"{fmt:>7} @ {size:>9}"


def try_mode(index, backend, fourcc, width, height, out_dir):
    with _quiet_opencv():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        frame = None
        for _ in range(WARMUP_FRAMES):
            ok, candidate = cap.read()
            if ok and candidate is not None:
                frame = normalise_frame(candidate)
                if not frame_is_blank(frame):
                    break

        actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        got_fourcc = fourcc_of(cap)
        cap.release()

    if frame is None:
        return {"ok": False, "reason": "no frame returned"}

    blank = frame_is_blank(frame)
    result = {
        "ok": True,
        "blank": blank,
        "actual": actual,
        "fourcc": got_fourcc,
        "shape": frame.shape,
        "std": float(frame.std()),
        "mean_bgr": [round(float(v), 1) for v in frame.reshape(-1, 3).mean(axis=0)],
    }
    if not blank:
        name = f"cam{index}_{backend_name(backend)}_{fourcc or 'default'}_{actual[0]}x{actual[1]}.png"
        path = out_dir / name.replace(" ", "")
        cv2.imwrite(str(path), frame)
        result["saved"] = str(path)
    return result


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="camprobe-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"OpenCV {cv2.__version__}")
    print(f"Backends tried: {', '.join(backend_name(b) for b in backend_candidates())}")
    print(f"Sample frames -> {out_dir}\n")

    any_camera = False
    any_working = False

    for index in range(MAX_INDEX):
        header_shown = False
        for backend in backend_candidates():
            for fourcc, width, height in MODES:
                result = try_mode(index, backend, fourcc, width, height, out_dir)
                if result is None:
                    break          # this backend cannot open this index at all
                if not header_shown:
                    print(f"=== Camera {index} " + "=" * 52)
                    header_shown = True
                    any_camera = True

                label = f"  {backend_name(backend):>16} | {describe(fourcc, width, height)} | "
                if not result["ok"]:
                    print(label + result["reason"])
                    continue
                actual = f"{result['actual'][0]}x{result['actual'][1]}"
                got = result["fourcc"] or "?"
                if result["blank"]:
                    print(f"{label}BLANK  got {actual} {got}  "
                          f"std={result['std']:.2f} mean={result['mean_bgr']}")
                else:
                    any_working = True
                    print(f"{label}OK     got {actual} {got}  "
                          f"std={result['std']:.1f}  saved {Path(result['saved']).name}")
        if header_shown:
            print()

    if not any_camera:
        print("No cameras could be opened on indices 0-5.")
        print("Check the camera is plugged in and not in use by another application.")
    elif not any_working:
        print("Every camera opened but only returned blank frames.")
        print("The camera is not supplying decodable video in any format tried.")
        print("Worth checking: another application holding the camera, a USB 2 port")
        print("for a high-resolution camera, or a vendor driver that needs its own app.")
    else:
        print(f"Working configurations found. Sample frames are in {out_dir}")


if __name__ == "__main__":
    main()
