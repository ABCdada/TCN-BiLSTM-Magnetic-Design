#!/usr/bin/env python3
"""Create a deterministic, clearly labelled CSV for the browser upload demo."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


SAMPLE_RATE_HZ = 16_000_000
WINDOW_LENGTH = 1_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/data/synthetic_window_A_1000.csv"),
    )
    args = parser.parse_args()

    time_s = np.arange(WINDOW_LENGTH, dtype=np.float64) / SAMPLE_RATE_HZ
    phase = 2.0 * np.pi * 100_000.0 * time_s
    b = 0.092 * np.sin(phase) + 0.010 * np.sin(3.0 * phase + 0.25)
    db = np.diff(b, prepend=b[:1])
    direction = np.where(db >= 0.0, 1.0, -1.0)
    h = 225.0 * b + 2.75 * direction

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["B", "H"])
        writer.writerows(zip(b, h))
    print(f"wrote {WINDOW_LENGTH} synthetic rows to {args.output}")


if __name__ == "__main__":
    main()

