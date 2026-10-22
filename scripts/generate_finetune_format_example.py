#!/usr/bin/env python3
"""Create a deterministic ZIP that documents a leakage-auditable data format."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np


SAMPLE_RATE_HZ = 16_000_000
SEQUENCE_LENGTH = 1_000
RECORDS = 12


def csv_text(rows) -> str:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/data/synthetic_finetune_format.zip"),
    )
    args = parser.parse_args()

    time_s = np.arange(SEQUENCE_LENGTH, dtype=np.float64) / SAMPLE_RATE_HZ
    b_rows = []
    h_rows = []
    temperatures = []
    acquisition_ids = []
    for index in range(RECORDS):
        frequency = 70_000.0 + 5_000.0 * index
        phase = 2.0 * np.pi * frequency * time_s + 0.08 * index
        amplitude = 0.065 + 0.002 * index
        b = amplitude * np.sin(phase) + 0.007 * np.sin(3.0 * phase + 0.2)
        db = np.diff(b, prepend=b[:1])
        h = 220.0 * b + (2.2 + 0.05 * index) * np.where(db >= 0.0, 1.0, -1.0)
        b_rows.append(b)
        h_rows.append(h)
        temperatures.append([25.0 + 5.0 * (index % 4)])
        acquisition_ids.append([f"synthetic_acquisition_{index:03d}"])

    readme = """Synthetic fine-tuning data-format example

Files:
- B.csv: 12 rows x 1000 flux-density values in tesla
- H.csv: 12 rows x 1000 synthetic field-strength values in A/m
- T.csv: one temperature in degrees Celsius per row
- acquisition_id.csv: one independent-record identifier per row

This ZIP is a deterministic format-only fixture. It contains no measured data,
does not reproduce a manuscript experiment, and is not accepted by a Web
fine-tuning endpoint in this inference-focused release. A future trainer must
split acquisition IDs before any overlapping window extraction.
"""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("B.csv", csv_text(b_rows))
        archive.writestr("H.csv", csv_text(h_rows))
        archive.writestr("T.csv", csv_text(temperatures))
        archive.writestr("acquisition_id.csv", csv_text(acquisition_ids))
        archive.writestr("README.txt", readme)
    print(f"wrote format-only synthetic ZIP to {args.output}")


if __name__ == "__main__":
    main()

