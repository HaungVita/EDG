#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Compare EDG TVR metrics with the reproduced targets.")
    parser.add_argument("metrics", type=Path, help="Generated *_metrics.json file")
    parser.add_argument("--task", required=True, choices=("VR", "SVMR", "VCMR"))
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "expected_metrics" / "tvr.json").read_text())[args.task]
    actual = json.loads(args.metrics.read_text())[args.task]

    failed = False
    for key, target in expected.items():
        value = actual.get(key)
        ok = value is not None and abs(value - target) <= args.atol
        print(f"{key:10s} actual={value!s:>6s} expected={target:6.2f}  {'OK' if ok else 'FAIL'}")
        failed |= not ok
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

