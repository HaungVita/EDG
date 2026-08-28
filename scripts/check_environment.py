#!/usr/bin/env python3
"""Validate the local path configuration without loading models or features."""

import os
import shlex
import sys
from pathlib import Path


FILE_KEYS = (
    "TVR_VAL_JSONL",
    "QUERY_H5",
    "SUB_H5",
    "SUB_INFO_JSON",
    "DURATION_JSON",
    "SLOWFAST_H5",
    "FACE_H5",
    "PORTRAIT_H5",
    "VCMR_VR_INPUT",
)
MODEL_KEYS = ("VR_MODEL_DIR", "SVMR_MODEL_DIR", "VCMR_MODEL_DIR")
TRAIN_FILE_KEYS = (
    "TVR_VR_TRAIN_JSONL",
    "TVR_MOMENT_TRAIN_JSONL",
    "TRAIN_VR_INPUT",
    "IDX2VIDEO_JSON",
)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        tokens = shlex.split(value, comments=True)
        values[key.strip()] = tokens[0] if tokens else ""
    return values


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    config = Path(os.environ.get("EDG_PATHS_FILE", root / "configs/paths.env"))
    if not config.is_file():
        print(f"MISSING config: {config}")
        return 2

    try:
        values = load_env(config)
    except ValueError as error:
        print(error)
        return 2

    failed = False
    for key in FILE_KEYS:
        path = Path(values.get(key, ""))
        ok = bool(str(path)) and path.is_file()
        print(f"{'OK' if ok else 'MISSING':7s} {key:16s} {path}")
        failed |= not ok

    for key in MODEL_KEYS:
        path = Path(values.get(key, ""))
        ok = path.is_dir() and all((path / name).is_file() for name in ("model.ckpt", "opt.json"))
        print(f"{'OK' if ok else 'MISSING':7s} {key:16s} {path}")
        failed |= not ok

    for key in TRAIN_FILE_KEYS:
        path = Path(values.get(key, ""))
        ok = path.is_file()
        print(f"{'OK' if ok else 'MISSING':7s} {key:22s} {path}")
        failed |= not ok

    results_root = Path(values.get("TRAIN_RESULTS_ROOT", ""))
    ok = results_root.is_dir() or (results_root.parent.is_dir() and os.access(results_root.parent, os.W_OK))
    print(f"{'OK' if ok else 'MISSING':7s} {'TRAIN_RESULTS_ROOT':22s} {results_root}")
    failed |= not ok

    python_bin = Path(values.get("PYTHON_BIN", sys.executable))
    ok = python_bin.is_file()
    print(f"{'OK' if ok else 'MISSING':7s} {'PYTHON_BIN':16s} {python_bin}")
    return 1 if failed or not ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
