#!/usr/bin/env python3
"""Generate honest synthetic examples for software-chain verification.

The script deliberately does not calculate accuracy metrics.  Its known-H
prefixes are synthetic initial conditions, not reference continuations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


SAMPLE_RATE_HZ = 16_000_000
WINDOW_LENGTH = 1_000
DISCLAIMER = (
    "Deterministic synthetic software-verification artifact; contains no "
    "measured target and supports no prediction-accuracy claim."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def synthetic_b(length: int = WINDOW_LENGTH) -> np.ndarray:
    """Bounded multi-harmonic B waveform used only to exercise inference."""

    time_s = np.arange(length, dtype=np.float64) / SAMPLE_RATE_HZ
    phase = 2.0 * np.pi * 100_000.0 * time_s
    return (
        0.092 * np.sin(phase)
        + 0.010 * np.sin(3.0 * phase + 0.25)
        + 0.004 * np.sin(5.0 * phase - 0.15)
    )


def synthetic_known_h(b: np.ndarray, history: int) -> np.ndarray:
    """Synthetic state prefix; never interpreted as a measured target."""

    db = np.diff(b, prepend=b[:1])
    direction = np.where(db >= 0.0, 1.0, -1.0)
    return (225.0 * b + 2.75 * direction)[:history]


def all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def polyline(values: Iterable[float], x: float, y: float, width: float, height: float) -> str:
    array = np.asarray(list(values), dtype=np.float64)
    indices = np.linspace(0, array.size - 1, min(array.size, 500), dtype=int)
    array = array[indices]
    lower = float(np.min(array))
    upper = float(np.max(array))
    span = max(upper - lower, 1e-12)
    points = []
    for position, value in enumerate(array):
        px = x + width * position / max(array.size - 1, 1)
        py = y + height * (1.0 - (float(value) - lower) / span)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def write_window_svg(path: Path, payload: dict, result: dict) -> None:
    b = payload["B"]
    h = result["H"]
    split = int(result["history_length"])
    duration_us = 1e6 * WINDOW_LENGTH / SAMPLE_RATE_HZ
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720" viewBox="0 0 1200 720">
<rect width="1200" height="720" fill="white"/>
<style>text{{font-family:Arial,sans-serif;fill:#202124}} .axis{{stroke:#333;stroke-width:1}} .grid{{stroke:#ddd;stroke-width:1}} .b{{fill:none;stroke:#1261a0;stroke-width:2}} .h{{fill:none;stroke:#c73535;stroke-width:2}}</style>
<text x="60" y="42" font-size="25">Synthetic window response: Material {result['material']} ({result['grade']}), {result['mode']}</text>
<text x="60" y="70" font-size="15">{DISCLAIMER}</text>
<rect x="75" y="105" width="1050" height="225" fill="none" class="axis"/>
<polyline points="{polyline(b, 75, 105, 1050, 225)}" class="b"/>
<text x="20" y="225" font-size="18" transform="rotate(-90 20 225)">B (T), independently scaled</text>
<rect x="75" y="390" width="1050" height="225" fill="none" class="axis"/>
<polyline points="{polyline(h, 75, 390, 1050, 225)}" class="h"/>
<line x1="{75 + 1050 * split / WINDOW_LENGTH:.2f}" y1="390" x2="{75 + 1050 * split / WINDOW_LENGTH:.2f}" y2="615" stroke="#555" stroke-dasharray="7,5"/>
<text x="20" y="510" font-size="18" transform="rotate(-90 20 510)">Reconstructed H (A/m)</text>
<text x="520" y="655" font-size="18">Time within 1000-sample window (0 to {duration_us:.1f} us)</text>
<text x="80" y="690" font-size="14">Dashed line: known/predicted boundary. No measured continuation is present.</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def write_converter_svg(path: Path, result: dict) -> None:
    sim = result["simulation"]
    h = result["H"]
    b = result["B"]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" viewBox="0 0 1200 820">
<rect width="1200" height="820" fill="white"/>
<style>text{{font-family:Arial,sans-serif;fill:#202124}} .axis{{stroke:#333;stroke-width:1}} .v{{fill:none;stroke:#1261a0;stroke-width:1.7}} .i{{fill:none;stroke:#5b8c35;stroke-width:1.7}} .b{{fill:none;stroke:#6a4c93;stroke-width:1.7}} .h{{fill:none;stroke:#c73535;stroke-width:1.7}}</style>
<text x="60" y="40" font-size="25">Synthetic converter pipeline: {sim['topology']}</text>
<text x="60" y="67" font-size="15">{DISCLAIMER}</text>
<rect x="75" y="100" width="1050" height="140" fill="none" class="axis"/><polyline points="{polyline(sim['induced_voltage_V'], 75, 100, 1050, 140)}" class="v"/><text x="15" y="185" font-size="16">vL</text>
<rect x="75" y="280" width="1050" height="140" fill="none" class="axis"/><polyline points="{polyline(sim['nominal_current_A'], 75, 280, 1050, 140)}" class="i"/><text x="15" y="365" font-size="16">i nominal</text>
<rect x="75" y="460" width="1050" height="140" fill="none" class="axis"/><polyline points="{polyline(b, 75, 460, 1050, 140)}" class="b"/><text x="15" y="545" font-size="16">B</text>
<rect x="75" y="640" width="1050" height="140" fill="none" class="axis"/><polyline points="{polyline(h, 75, 640, 1050, 140)}" class="h"/><text x="15" y="725" font-size="16">H model</text>
<text x="760" y="808" font-size="14">Nominal current seeds only the known prefix; it is not a measured validation target.</text>
</svg>'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def window_cases(service, materials, mode_history, output: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    summaries: list[dict] = []
    b = synthetic_b().tolist()
    for material_id, spec in materials.items():
        for mode in spec.supported_modes:
            history = mode_history[mode]
            known_h = [0.0] if spec.known_h_placeholder_zero else synthetic_known_h(np.asarray(b), history).tolist()
            case_id = f"window_{material_id}_{mode.replace('-', '_')}"
            payload = {
                "artifact_class": "synthetic_software_verification_input",
                "scope_note": DISCLAIMER,
                "material": material_id,
                "mode": mode,
                "B": b,
                "temperature": 25.0,
                "known_H": known_h,
                "sampling_rate_hz": SAMPLE_RATE_HZ,
            }
            service_payload = {key: value for key, value in payload.items() if key not in {"artifact_class", "scope_note"}}
            result = service.predict_window_payload(service_payload)
            result["artifact_metadata"] = {
                "artifact_class": "synthetic_software_verification_output",
                "scope_note": DISCLAIMER,
                "accuracy_metrics_present": False,
            }
            input_path = output / "inputs" / f"{case_id}.json"
            output_path = output / "outputs" / f"{case_id}.json"
            timeseries_path = output / "timeseries" / f"{case_id}.csv"
            write_json(input_path, payload)
            write_json(output_path, result)
            write_window_timeseries(timeseries_path, payload, result)
            if mode == spec.default_mode:
                write_window_svg(output / "figures" / f"{case_id}.svg", payload, result)
            finite = all_finite(result)
            records.append({
                "case_id": case_id,
                "kind": "window_prediction",
                "input": input_path.relative_to(output).as_posix(),
                "output": output_path.relative_to(output).as_posix(),
                "input_sha256": sha256_file(input_path),
                "output_sha256": sha256_file(output_path),
                "timeseries_csv": timeseries_path.relative_to(output).as_posix(),
                "timeseries_csv_sha256": sha256_file(timeseries_path),
                "inference_completed": True,
                "finite_output": finite,
            })
            summaries.append({
                "case_id": case_id,
                "material": material_id,
                "grade": spec.grade,
                "mode": mode,
                "known_samples": history,
                "predicted_samples": WINDOW_LENGTH - history,
                "output_samples": len(result["H"]),
                "finite_output": finite,
                "validation_reference_present": False,
                "accuracy_metrics_reported": False,
                "signed_interval_h_db_j_per_m3": result["energy"]["h_db_integral_j_per_m3"],
                "closed_path_energy_j_per_m3": result["energy"]["closed_path_energy_density_j_per_m3"],
            })
    return records, summaries


def converter_cases(service, output: Path) -> tuple[list[dict], list[dict]]:
    records: list[dict] = []
    summaries: list[dict] = []
    topologies = (
        "single_phase_full_bridge",
        "three_phase_full_bridge",
        "three_phase_t_type",
    )
    for topology in topologies:
        case_id = f"converter_A_{topology}"
        payload = {
            "artifact_class": "synthetic_software_verification_input",
            "scope_note": DISCLAIMER,
            "material": "A",
            "topology": topology,
            "mode": "100-900",
            "temperature": 25.0,
            "dc_voltage": 50.0,
            "grid_voltage_rms": 20.0,
            "grid_frequency_hz": 50.0,
            "switching_frequency_hz": 100_000.0,
            "filter_inductance_h": 145.0e-6,
            "series_resistance_ohm": 0.05,
            "initial_current_A": 0.0,
            "grid_phase_deg": 0.0,
            "turns": 20,
            "area_m2": 25.0e-6,
            "path_length_m": 0.06,
            "sampling_rate_hz": SAMPLE_RATE_HZ,
            "initial_B": 0.0,
        }
        service_payload = {key: value for key, value in payload.items() if key not in {"artifact_class", "scope_note"}}
        result = service.converter_demo_payload(service_payload)
        result["artifact_metadata"] = {
            "artifact_class": "synthetic_software_verification_output",
            "scope_note": DISCLAIMER,
            "accuracy_metrics_present": False,
        }
        input_path = output / "inputs" / f"{case_id}.json"
        output_path = output / "outputs" / f"{case_id}.json"
        timeseries_path = output / "timeseries" / f"{case_id}.csv"
        write_json(input_path, payload)
        write_json(output_path, result)
        write_converter_timeseries(timeseries_path, result)
        write_converter_svg(output / "figures" / f"{case_id}.svg", result)
        finite = all_finite(result)
        records.append({
            "case_id": case_id,
            "kind": "converter_pipeline",
            "input": input_path.relative_to(output).as_posix(),
            "output": output_path.relative_to(output).as_posix(),
            "input_sha256": sha256_file(input_path),
            "output_sha256": sha256_file(output_path),
            "timeseries_csv": timeseries_path.relative_to(output).as_posix(),
            "timeseries_csv_sha256": sha256_file(timeseries_path),
            "inference_completed": True,
            "finite_output": finite,
        })
        summaries.append({
            "case_id": case_id,
            "topology": topology,
            "sample_count": result["simulation"]["sample_count"],
            "finite_output": finite,
            "validation_reference_present": result["metrics"]["validation_available"],
            "accuracy_metrics_reported": False,
            "scope": "deterministic_SPWM_RL_plus_neural_window_postprocessing",
        })
    return records, summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_window_timeseries(path: Path, payload: dict, result: dict) -> None:
    history = int(result["history_length"])
    known = list(result["known_H"])
    rows = []
    for index, (b_value, h_value) in enumerate(zip(payload["B"], result["H"])):
        rows.append({
            "sample_index": index,
            "time_us": index / SAMPLE_RATE_HZ * 1e6,
            "B_T": b_value,
            "known_H_A_per_m": known[index] if index < history else "",
            "reconstructed_H_A_per_m": h_value,
            "region": "known" if index < history else "predicted",
        })
    write_csv(path, rows)


def write_converter_timeseries(path: Path, result: dict) -> None:
    simulation = result["simulation"]
    history = int(result["prediction"]["history_length"])
    rows = []
    for index in range(int(simulation["sample_count"])):
        rows.append({
            "sample_index": index,
            "time_us": float(simulation["time_s"][index]) * 1e6,
            "inverter_voltage_V": simulation["inverter_voltage_V"][index],
            "grid_voltage_V": simulation["grid_voltage_V"][index],
            "induced_voltage_V": simulation["induced_voltage_V"][index],
            "nominal_current_A": simulation["nominal_current_A"][index],
            "B_T": result["B"][index],
            "reconstructed_H_A_per_m": result["H"][index],
            "region": "known" if index < history else "predicted",
        })
    write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("generated"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output = args.output.resolve()
    if not (project_root / "magnetic_model").is_dir():
        raise SystemExit(f"invalid project root: {project_root}")
    sys.path.insert(0, str(project_root))
    os.environ["MAGNET_MODEL_DIR"] = str(project_root / "weights")
    os.environ["MAGNET_DEVICE"] = args.device

    from magnetic_model.config import MATERIALS, MODE_HISTORY  # noqa: PLC0415
    from magnetic_model.inference import ModelManager  # noqa: PLC0415
    from magnetic_model.service import InferenceService  # noqa: PLC0415

    manager = ModelManager(device=args.device)
    service = InferenceService(manager)
    records, window_summary = window_cases(service, MATERIALS, MODE_HISTORY, output)
    converter_records, converter_summary = converter_cases(service, output)
    records.extend(converter_records)
    write_csv(output / "tables" / "window_smoke_summary.csv", window_summary)
    write_csv(output / "tables" / "converter_smoke_summary.csv", converter_summary)

    checkpoint_manifest = project_root / "weights" / "manifest.json"
    manifest = {
        "format_version": 1,
        "artifact_class": "synthetic_software_verification",
        "claim_scope": {
            "measured_data": False,
            "accuracy_claims_permitted": False,
            "statement": DISCLAIMER,
        },
        "generator": {
            "script": "scripts/generate_reproducible_bundle.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_root_name": project_root.name,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(manager.device),
            "cuda_runtime": torch.version.cuda,
        },
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "cases": records,
    }
    write_json(output / "manifest.json", manifest)
    print(f"generated {len(records)} cases under {output}")


if __name__ == "__main__":
    main()
