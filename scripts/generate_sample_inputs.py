#!/usr/bin/env python3
"""Generate deterministic API smoke-test inputs without private data.

The synthetic waveforms exercise serialization, validation, preprocessing, and
model inference. They are not experimental measurements and must not be used to
reproduce or support accuracy claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


SAMPLE_RATE_HZ = 16_000_000
SWITCHING_FREQUENCY_HZ = 100_000
WINDOW_SIZE = 1_000


def synthetic_b(length: int) -> list[float]:
    """Return a bounded multi-harmonic flux-density waveform in tesla."""
    values: list[float] = []
    for index in range(length):
        time_s = index / SAMPLE_RATE_HZ
        phase = 2.0 * math.pi * SWITCHING_FREQUENCY_HZ * time_s
        value = 0.105 * math.sin(phase) + 0.012 * math.sin(3.0 * phase + 0.25)
        values.append(round(value, 10))
    return values


def synthetic_h(b_values: list[float]) -> list[float]:
    """Create a simple history-dependent trajectory solely for API examples."""
    result: list[float] = []
    previous = b_values[0]
    for value in b_values:
        direction = 1.0 if value - previous >= 0.0 else -1.0
        result.append(round(260.0 * value + 3.5 * direction, 8))
        previous = value
    return result


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_payloads() -> dict[str, dict]:
    b_window = synthetic_b(WINDOW_SIZE)
    h_window = synthetic_h(b_window)

    long_length = 2_500
    b_long = synthetic_b(long_length)
    h_long = synthetic_h(b_long)

    turns = 20
    area_m2 = 1.0e-4
    path_length_m = 0.08
    dt = 1.0 / SAMPLE_RATE_HZ
    voltage = [0.0]
    for index in range(1, WINDOW_SIZE):
        db_dt = (b_window[index] - b_window[index - 1]) / dt
        voltage.append(round(turns * area_m2 * db_dt, 8))
    current = [round(value * path_length_m / turns, 8) for value in h_window]

    return {
        "predict_window_A.json": {
            "material": "A",
            "B": b_window,
            "temperature": 25.0,
            "known_H": h_window[:100],
            "mode": "100-900",
            "sampling_rate_hz": SAMPLE_RATE_HZ,
        },
        "predict_sequence_A.json": {
            "material": "A",
            "B": b_long,
            "temperature": 25.0,
            "known_H": h_long[:500],
            "history_length": 500,
            "hop_length": 500,
            "sampling_rate_hz": SAMPLE_RATE_HZ,
        },
        "circuit_demo_A.json": {
            "material": "A",
            "induced_voltage": voltage,
            "current": current[:100],
            "temperature": 25.0,
            "turns": turns,
            "area_m2": area_m2,
            "path_length_m": path_length_m,
            "sampling_rate_hz": SAMPLE_RATE_HZ,
            "initial_B": b_window[0],
            "history_length": 100,
        },
        "converter_demo_A.json": {
            "material": "A",
            "topology": "single_phase_full_bridge",
            "mode": "100-900",
            "temperature": 25.0,
            "dc_voltage": 50.0,
            "grid_voltage_rms": 20.0,
            "grid_frequency_hz": 50.0,
            "switching_frequency_hz": SWITCHING_FREQUENCY_HZ,
            "filter_inductance_h": 145.0e-6,
            "series_resistance_ohm": 0.05,
            "initial_current_A": 0.0,
            "grid_phase_deg": 0.0,
            "turns": turns,
            "area_m2": 25.0e-6,
            "path_length_m": 0.06,
            "sampling_rate_hz": SAMPLE_RATE_HZ,
            "initial_B": 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/generated"),
        help="directory for generated JSON requests",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, payload in build_payloads().items():
        output = args.output_dir / filename
        write_json(output, payload)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
