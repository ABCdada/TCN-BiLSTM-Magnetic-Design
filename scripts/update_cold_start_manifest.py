#!/usr/bin/env python3
"""Atomically register completed cold-start checkpoints in their manifest.

Missing files remain explicitly ``pending``.  A file is marked ``ready`` only
after its state dictionary loads strictly into the declared architecture and
its normalization metadata is validated.  Hashes and byte counts are always
computed from the actual file; this script never invents integrity metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magnetic_model import COLD_START_MATERIALS, build_model, count_parameters  # noqa: E402
from magnetic_model.inference import (  # noqa: E402
    NormalizationStats,
    normalize_cold_start_state_dict_keys,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(path: Path, material_id: str) -> None:
    spec = COLD_START_MATERIALS[material_id]
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "norm" not in checkpoint:
        raise ValueError(f"{path.name} must contain 'model' and 'norm'")
    state = checkpoint["model"]
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}
    state = normalize_cold_start_state_dict_keys(state)
    model = build_model(spec.model)
    model.load_state_dict(state, strict=True)
    if count_parameters(model) != spec.parameter_count:
        raise ValueError(f"parameter-count mismatch for Material {material_id}")
    NormalizationStats.from_mapping(checkpoint["norm"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights-dir", type=Path, default=PROJECT_ROOT / "weights"
    )
    parser.add_argument(
        "--material",
        nargs="+",
        choices=sorted(COLD_START_MATERIALS),
        default=sorted(COLD_START_MATERIALS),
    )
    args = parser.parse_args()
    weights_dir = args.weights_dir.expanduser().resolve()
    manifest_path = weights_dir / "cold_start_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("profile") != "cold_start_1_999":
        raise ValueError("refusing to update a non-cold-start manifest")

    # The manifest is an exact release registry, not an append-only archive.
    # Drop records for models that are no longer exposed by the browser-facing
    # cold-start service while leaving their checkpoint files untouched.
    existing_models = payload.get("models", {})
    payload["models"] = {
        material_id: existing_models.get(material_id, {})
        for material_id in COLD_START_MATERIALS
    }

    for material_id in args.material:
        spec = COLD_START_MATERIALS[material_id]
        path = weights_dir / spec.checkpoint_filename
        record = {
            "status": "pending",
            "material": spec.grade,
            "parameter_count": spec.parameter_count,
            "path": spec.checkpoint_filename,
        }
        if path.is_file():
            validate_checkpoint(path, material_id)
            stat = path.stat()
            record.update(
                status="ready",
                bytes=stat.st_size,
                sha256=sha256_file(path),
            )
        payload["models"][material_id] = record

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
