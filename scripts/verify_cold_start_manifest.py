#!/usr/bin/env python3
"""Verify the four ready browser cold-start checkpoints without modifying them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from magnetic_model import COLD_START_MATERIALS, ModelManager, count_parameters  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("weights/cold_start_manifest.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = tuple(COLD_START_MATERIALS)
    if payload.get("format_version") != 1 or payload.get("algorithm") != "sha256":
        raise SystemExit("unsupported manifest format or algorithm")
    if payload.get("profile") != "cold_start_1_999":
        raise SystemExit("manifest is not the cold_start_1_999 profile")
    if payload.get("window_length") != 1000 or payload.get("history_length") != 1:
        raise SystemExit("manifest does not declare the native 1-999 contract")
    records = payload.get("models")
    if not isinstance(records, dict) or tuple(records) != expected_ids:
        raise SystemExit("manifest must contain exactly A, B, D, and E in release order")

    manager = ModelManager(
        model_dir=manifest_path.parent,
        device=args.device,
        registry=COLD_START_MATERIALS,
        manifest_filename=manifest_path.name,
        checkpoint_env_prefix="MAGNET_COLD_START_CHECKPOINT_",
        expected_profile="cold_start_1_999",
    )
    failures: list[str] = []
    for material_id, spec in COLD_START_MATERIALS.items():
        record = records[material_id]
        if record.get("status") != "ready":
            failures.append(f"{material_id}: checkpoint is not marked ready")
            continue
        try:
            digest = manager.verify_checkpoint(material_id)
            loaded = manager.load(material_id)
            actual = count_parameters(loaded.model)
            if actual != spec.parameter_count:
                failures.append(
                    f"{material_id}: parameter count {actual} != {spec.parameter_count}"
                )
            if digest != record.get("sha256"):
                failures.append(f"{material_id}: verified digest differs from manifest")
        except Exception as exc:  # report all four records in one invocation
            failures.append(f"{material_id}: {exc}")

    if failures:
        raise SystemExit("cold-start manifest verification failed:\n" + "\n".join(failures))
    print("verified checkpoint structure, sizes, SHA-256, and parameters for A/B/D/E")


if __name__ == "__main__":
    main()
