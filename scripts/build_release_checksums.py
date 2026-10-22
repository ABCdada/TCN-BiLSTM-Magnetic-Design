#!/usr/bin/env python3
"""Write SHA-256 entries for stable release evidence and distributions."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


DEFAULT_ROOTS = (
    ".github",
    "docs",
    "examples",
    "licenses",
    "magnetic_model",
    "results",
    "schemas",
    "scripts",
    "static",
    "templates",
    "tests",
    "weights/manifest.json",
    "weights/cold_start_manifest.json",
    "weights/README.md",
    ".dockerignore",
    ".gitattributes",
    ".gitignore",
    "API.md",
    "app.py",
    "CHANGELOG.md",
    "CITATION.cff",
    "cli.py",
    "CONTRIBUTING.md",
    "DATA_CARD.md",
    "DEPLOYMENT.md",
    "Dockerfile",
    "GITHUB_UPLOAD_CHECKLIST.md",
    "LICENSE",
    "MANIFEST.in",
    "MODEL_CARD.md",
    "MODEL_WEIGHTS_NOTICE.md",
    "pyproject.toml",
    "pytest.ini",
    "README.md",
    "README_zh-CN.md",
    "RELEASE_NOTES.md",
    "REPRODUCIBILITY.md",
    "requirements.txt",
    "requirements-validated.txt",
    "SECURITY.md",
    "start_linux.sh",
    "start_windows.bat",
    "TESTING.md",
    "THIRD_PARTY_NOTICES.md",
    "TRAINING.md",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def files_under(project_root: Path, entries: tuple[str, ...]):
    files: list[Path] = []
    for entry in entries:
        path = project_root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(candidate for candidate in path.rglob("*") if candidate.is_file())
    return sorted(set(files), key=lambda item: item.relative_to(project_root).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("SHA256SUMS.txt"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    records = []
    for path in files_under(root, DEFAULT_ROOTS):
        if path.resolve() == output.resolve():
            continue
        records.append(f"{digest(path)}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(records) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} checksums to {output}")


if __name__ == "__main__":
    main()
