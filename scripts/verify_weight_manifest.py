#!/usr/bin/env python3
"""Verify checkpoint paths, sizes, and hashes in a weight manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_IDS = {"A", "B", "C", "D", "E"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("weights/manifest.json"))
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format_version") != 1 or payload.get("algorithm") != "sha256":
        raise SystemExit("unsupported manifest format or algorithm")

    records = payload.get("models")
    if not isinstance(records, dict) or set(records) != EXPECTED_IDS:
        raise SystemExit("manifest must contain exactly model IDs A, B, C, D, and E")

    weights_root = manifest_path.parent.resolve()
    failures: list[str] = []
    for model_id in sorted(EXPECTED_IDS):
        record = records[model_id]
        relative = Path(record["path"])
        checkpoint = (weights_root / relative).resolve()
        if weights_root not in checkpoint.parents:
            failures.append(f"{model_id}: path escapes the weights directory")
            continue
        expected_hash = str(record.get("sha256", ""))
        if not SHA256_PATTERN.fullmatch(expected_hash):
            failures.append(f"{model_id}: malformed SHA-256 value")
            continue
        if not checkpoint.is_file():
            failures.append(f"{model_id}: missing {checkpoint}")
            continue
        if checkpoint.stat().st_size != int(record["bytes"]):
            failures.append(f"{model_id}: byte count differs")
        if sha256_file(checkpoint) != expected_hash:
            failures.append(f"{model_id}: SHA-256 differs")

    if failures:
        raise SystemExit("manifest verification failed:\n" + "\n".join(failures))
    print("verified checkpoint sizes and SHA-256 hashes for A-E")


if __name__ == "__main__":
    main()
