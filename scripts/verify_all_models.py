#!/usr/bin/env python3
"""Strictly verify hashes, checkpoints, parameter counts, and A-E forwards.

The report is a software-integrity artifact, not an accuracy benchmark. It uses
a deterministic synthetic tensor and never reads private training/test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magnetic_model import MATERIALS, ModelManager, count_parameters  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value.astype("<f4", copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def read_and_verify_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or payload.get("algorithm") != "sha256":
        raise ValueError("unsupported weight-manifest format")
    records = payload.get("models")
    if not isinstance(records, dict) or set(records) != set(MATERIALS):
        raise ValueError("manifest must contain exactly A, B, C, D, and E")

    root = path.parent.resolve()
    for model_id, spec in MATERIALS.items():
        record = records[model_id]
        checkpoint = (root / record["path"]).resolve()
        if root not in checkpoint.parents:
            raise ValueError(f"manifest path for {model_id} escapes the weights directory")
        if checkpoint.name != spec.checkpoint_filename:
            raise ValueError(f"manifest filename mismatch for {model_id}")
        if int(record["parameter_count"]) != spec.parameter_count:
            raise ValueError(f"manifest parameter count mismatch for {model_id}")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint.stat().st_size != int(record["bytes"]):
            raise ValueError(f"checkpoint byte-count mismatch for {model_id}")
        if sha256_file(checkpoint) != record["sha256"]:
            raise ValueError(f"checkpoint SHA-256 mismatch for {model_id}")
    return payload


def deterministic_input(device: torch.device, input_features: tuple[str, ...]) -> torch.Tensor:
    index = torch.linspace(0.0, 1.0, 1_000, dtype=torch.float32)
    values = {
        "B": 0.1 * torch.sin(12.5 * torch.pi * index),
        "dB": 0.01 * torch.cos(12.5 * torch.pi * index),
        "ddB": 0.001 * torch.sin(25.0 * torch.pi * index),
        "T": torch.full_like(index, 0.25),
        "H_known": torch.where(index < 0.1, 0.05 * index, torch.zeros_like(index)),
        "mask": (index < 0.1).to(torch.float32),
    }
    channels = torch.stack(tuple(values[name] for name in input_features), dim=-1)
    return channels.unsqueeze(0).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("weights/manifest.json"))
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, cuda, cuda:0, or auto (default: cpu for reproducibility)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_and_verify_manifest(manifest_path)
    torch.manual_seed(2026)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026)

    manager = ModelManager(model_dir=manifest_path.parent, device=args.device)
    try:
        displayed_manifest = manifest_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        displayed_manifest = manifest_path.name
    report = {
        "report_version": 1,
        "status": "ok",
        "manifest": displayed_manifest,
        "manifest_format_version": manifest["format_version"],
        "device": str(manager.device),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name(manager.device)
            if manager.device.type == "cuda"
            else None
        ),
        "models": {},
    }

    for model_id, spec in MATERIALS.items():
        started = time.perf_counter()
        loaded = manager.load(model_id)  # strict state-dict and norm validation
        actual_parameters = count_parameters(loaded.model)
        if actual_parameters != spec.parameter_count:
            raise RuntimeError(
                f"{model_id} has {actual_parameters} parameters; "
                f"expected {spec.parameter_count}"
            )
        sample = deterministic_input(manager.device, spec.input_features)
        with loaded.lock, torch.inference_mode():
            output = loaded.model(sample).detach().float().cpu().numpy()
        if output.shape != (1, 1_000) or not np.isfinite(output).all():
            raise RuntimeError(f"{model_id} produced invalid output {output.shape}")
        record = manifest["models"][model_id]
        report["models"][model_id] = {
            "material": spec.grade,
            "checkpoint": record["path"],
            "checkpoint_sha256": record["sha256"],
            "parameter_count": actual_parameters,
            "output_shape": list(output.shape),
            "output_min": float(output.min()),
            "output_max": float(output.max()),
            "output_mean": float(output.mean()),
            "output_float32_sha256": sha256_array(output),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }

    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
