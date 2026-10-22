from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from app import app
from magnetic_model import (
    COLD_START_MATERIALS,
    MATERIALS,
    ModelManager,
    Predictor,
    build_model,
)
from magnetic_model.inference import normalize_cold_start_state_dict_keys


def test_cold_start_registry_is_isolated_from_scientific_contracts() -> None:
    assert set(MATERIALS) == {"A", "B", "C", "D", "E"}
    assert list(COLD_START_MATERIALS) == ["A", "B", "D", "E"]
    for material_id, cold in COLD_START_MATERIALS.items():
        scientific = MATERIALS[material_id]
        assert cold is not scientific
        assert cold.model == scientific.model
        assert cold.parameter_count == scientific.parameter_count
        assert cold.supported_modes == ("1-999",)
        assert cold.default_mode == "1-999"
        assert cold.known_h_placeholder_zero is True
        assert cold.checkpoint_filename == f"{material_id}_cold_start_best_model.pth"

    # The old API/model family remains unchanged.
    assert MATERIALS["A"].supported_modes == ("100-900", "500-500", "900-100")
    assert MATERIALS["A"].checkpoint_filename == "A_best_model.pth"
    assert MATERIALS["C"].checkpoint_filename == "C_best_model.pth"


def test_cold_start_manifest_availability_matches_integrity_records() -> None:
    manager = ModelManager(
        device="cpu",
        registry=COLD_START_MATERIALS,
        manifest_filename="cold_start_manifest.json",
        checkpoint_env_prefix="MAGNET_COLD_START_CHECKPOINT_",
        expected_profile="cold_start_1_999",
    )
    manifest_path = Path(__file__).resolve().parents[1] / "weights" / "cold_start_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert list(manifest["models"]) == ["A", "B", "D", "E"]
    availability = {item["id"]: item["available"] for item in manager.public_models()}
    expected = {
        material_id: manifest["models"][material_id]["status"] == "ready"
        for material_id in COLD_START_MATERIALS
    }
    assert availability == expected
    for material_id, is_ready in expected.items():
        if is_ready:
            assert len(manager.verify_checkpoint(material_id)) == 64
        else:
            with pytest.raises(FileNotFoundError, match="pending"):
                manager.verify_checkpoint(material_id)


@pytest.mark.parametrize("material_id", ["A", "B", "D", "E"])
def test_training_aliases_normalize_and_load_strictly(material_id: str) -> None:
    model = build_model(COLD_START_MATERIALS[material_id].model)
    canonical = model.state_dict()
    training_aliases = {}
    for key, value in canonical.items():
        parts = key.split(".")
        if len(parts) == 4 and parts[0] == "tcn":
            if parts[2] == "pw":
                parts[2] = "pointwise"
            elif parts[2] == "proj":
                parts[2] = "residual"
        training_aliases[".".join(parts)] = value

    assert any(".pointwise." in key for key in training_aliases)
    if material_id in {"B", "D", "E"}:
        assert any(".residual." in key for key in training_aliases)
    normalized = normalize_cold_start_state_dict_keys(training_aliases)
    assert set(normalized) == set(canonical)
    build_model(COLD_START_MATERIALS[material_id].model).load_state_dict(
        normalized, strict=True
    )


def test_canonical_cold_start_keys_are_unchanged() -> None:
    canonical = build_model(COLD_START_MATERIALS["A"].model).state_dict()
    normalized = normalize_cold_start_state_dict_keys(canonical)
    assert list(normalized) == list(canonical)
    assert all(normalized[key] is value for key, value in canonical.items())


def test_cold_start_key_alias_collision_is_rejected() -> None:
    state = {
        "tcn.0.pw.weight": torch.zeros(1),
        "tcn.0.pointwise.weight": torch.ones(1),
    }
    with pytest.raises(ValueError, match="alias collision"):
        normalize_cold_start_state_dict_keys(state)


def test_recursive_carry_over_supports_any_cold_start_material(monkeypatch) -> None:
    manager = SimpleNamespace(
        get_material=lambda material: COLD_START_MATERIALS[str(material).upper()]
    )
    predictor = Predictor(manager)
    prefixes: list[float] = []

    def fake_predict_window(material, B, temperature, known_H, **kwargs):
        prefixes.append(float(known_H[0]))
        tail = float(known_H[0]) + np.arange(1, 1_000, dtype=np.float64)
        return SimpleNamespace(predicted_H=tail)

    monkeypatch.setattr(predictor, "predict_window", fake_predict_window)
    result = predictor.predict_recursive_sequence(
        "A", np.linspace(-0.1, 0.1, 1_200), 25.0, [0.0]
    )

    assert result["mode"] == "1-999"
    assert result["window_count"] == 2
    assert prefixes == [0.0, 999.0]
    assert len(result["H"]) == 1_200


@pytest.mark.api
def test_cold_start_route_uses_separate_service(monkeypatch) -> None:
    scientific = app.extensions["magnetic_service"]
    cold = app.extensions["cold_start_magnetic_service"]
    monkeypatch.setattr(
        scientific,
        "predict_legacy_payload",
        lambda payload: {"family": "scientific", "payload": payload},
    )
    monkeypatch.setattr(
        cold,
        "predict_legacy_payload",
        lambda payload: {"family": "cold-start", "payload": payload},
    )

    request = {"materialId": 1, "sample": {"B": [0.0, 0.1], "T": 25.0}}
    app.config.update(TESTING=True)
    with app.test_client() as client:
        old_response = client.post("/api/predict", json=request)
        cold_response = client.post("/api/predict-cold-start", json=request)

    assert old_response.get_json()["family"] == "scientific"
    assert cold_response.get_json()["family"] == "cold-start"
