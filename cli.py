"""Command-line interface for offline and scripted model inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from magnetic_model import InferenceService, ModelManager


def _read_json(path: str) -> dict:
    text = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("input JSON must be an object")
    return value


def _write_json(value: dict, path: str | None) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if path:
        Path(path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    """Build the public CLI argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="trusted directory containing A--E checkpoints")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("models", help="list configured material models")
    for command in (
        "predict-window",
        "predict-sequence",
        "circuit-demo",
        "converter-demo",
    ):
        child = subparsers.add_parser(command)
        child.add_argument("input", help="JSON request file, or - for standard input")
        child.add_argument("--output", help="write JSON to this path instead of stdout")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one CLI request and return a conventional process status."""

    args = build_parser().parse_args(argv)
    try:
        service = InferenceService(ModelManager(args.model_dir, args.device))
        if args.command == "models":
            result = service.list_models()
            output = None
        else:
            payload = _read_json(args.input)
            function = {
                "predict-window": service.predict_window_payload,
                "predict-sequence": service.predict_sequence_payload,
                "circuit-demo": service.circuit_demo_payload,
                "converter-demo": service.converter_demo_payload,
            }[args.command]
            result = function(payload)
            output = args.output
        _write_json(result, output)
        return 0
    except Exception as exc:  # concise CLI error; Flask has structured HTTP errors.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
