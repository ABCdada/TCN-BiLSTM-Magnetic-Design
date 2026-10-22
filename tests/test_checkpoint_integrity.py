from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from magnetic_model import ModelManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_rejects_architecture_compatible_checkpoint_swap(tmp_path) -> None:
    """B and C share a shape, so the release hash must distinguish them."""

    shutil.copy2(PROJECT_ROOT / "weights" / "manifest.json", tmp_path / "manifest.json")
    shutil.copy2(
        PROJECT_ROOT / "weights" / "B_best_model.pth",
        tmp_path / "C_best_model.pth",
    )
    manager = ModelManager(model_dir=tmp_path, device="cpu")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        manager.load("C")
