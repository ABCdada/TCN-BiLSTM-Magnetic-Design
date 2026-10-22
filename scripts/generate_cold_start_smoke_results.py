#!/usr/bin/env python3
"""Generate deterministic browser cold-start smoke outputs with real checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def write_trace(path: Path, b_values: np.ndarray, h_values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("index", "B_T", "H_A_per_m"))
        writer.writerows(
            (i, float(b), float(h))
            for i, (b, h) in enumerate(zip(b_values, h_values))
        )


def predict(client, material_id: int, b_values: np.ndarray) -> dict:
    response = client.post(
        "/api/predict-cold-start",
        json={
            "materialId": material_id,
            "sample": {"B": b_values.tolist(), "T": 25.0},
        },
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"materialId={material_id}: HTTP {response.status_code}: "
            f"{response.get_data(as_text=True)}"
        )
    return response.get_json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cold_start_smoke",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source = np.loadtxt(
        PROJECT_ROOT / "examples" / "data" / "synthetic_window_A_1000.csv",
        delimiter=",",
        skiprows=1,
    )[:, 0]
    source = np.asarray(source[:1000], dtype=np.float64)
    grades = ("3C92", "3C95", "FEC014", "T37")
    summary: dict[str, object] = {
        "report_version": 1,
        "fixture": "examples/data/synthetic_window_A_1000.csv:B",
        "temperature_C": 25.0,
        "protocol": "cold_start_1_999",
        "models": {},
    }

    app = create_app({"TESTING": True})
    with app.test_client() as client:
        for material_id, grade in enumerate(grades, start=1):
            result = predict(client, material_id, source)
            h_values = np.asarray(result["H_pred"], dtype=np.float64)
            filename = f"{grade}_window_1000.csv"
            write_trace(output_dir / filename, source, h_values)
            summary["models"][grade] = {
                "materialId": material_id,
                "points": int(h_values.size),
                "H_min_A_per_m": float(h_values.min()),
                "H_max_A_per_m": float(h_values.max()),
                "H_float64_sha256": array_sha256(h_values),
                "energy_quantity": result["energy_quantity"],
                "energy_value_W_per_m3": float(result["core_loss"]),
                "stitch": result["stitch"],
                "trace": filename,
            }

        long_b = np.concatenate((source, source, source[:200]))
        long_result = predict(client, 1, long_b)
        long_h = np.asarray(long_result["H_pred"], dtype=np.float64)
        write_trace(output_dir / "3C92_recursive_2200.csv", long_b, long_h)
        summary["recursive_2200"] = {
            "grade": "3C92",
            "points": int(long_h.size),
            "H_float64_sha256": array_sha256(long_h),
            "stitch": long_result["stitch"],
            "trace": "3C92_recursive_2200.csv",
        }

    (output_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote cold-start smoke evidence to {output_dir}")


if __name__ == "__main__":
    main()
