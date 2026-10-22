#!/usr/bin/env python3
"""Create a versioned SHA-256 manifest for the five model checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODELS = {
    "A": ("3C92", 497_793),
    "B": ("3C95", 2_092_545),
    "C": ("FEC007", 2_092_545),
    "D": ("FEC014", 2_512_593),
    "E": ("T37", 2_092_865),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--output", type=Path, default=Path("weights/manifest.json")
    )
    args = parser.parse_args()

    root = args.weights_dir.resolve()
    records: dict[str, dict] = {}
    missing: list[Path] = []
    for model_id, (material, parameter_count) in MODELS.items():
        checkpoint = root / f"{model_id}_best_model.pth"
        if not checkpoint.is_file():
            missing.append(checkpoint)
            continue
        records[model_id] = {
            "material": material,
            "parameter_count": parameter_count,
            "path": checkpoint.relative_to(root).as_posix(),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256_file(checkpoint),
        }

    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"missing required checkpoints:\n{joined}")

    payload = {
        "format_version": 1,
        "algorithm": "sha256",
        "models": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} with {len(records)} checkpoint records")


if __name__ == "__main__":
    main()
