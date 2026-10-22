from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from magnetic_model import InferenceService


class _FakePredictor:
    """Fast deterministic stand-in that also records the supplied H prefix."""

    def __init__(self):
        self.last_known_h = None

    def predict_window(self, material, B, temperature, known_H, **kwargs):
        self.last_known_h = np.asarray(known_H, dtype=float)
        output = np.ones(1_000, dtype=float)
        output[: len(self.last_known_h)] = self.last_known_h
        return SimpleNamespace(H=output)


def _service() -> tuple[InferenceService, _FakePredictor]:
    service = InferenceService()
    predictor = _FakePredictor()
    service.predictor = predictor
    return service, predictor


def _base_payload() -> dict:
    return {
        "material": "A",
        "induced_voltage": [0.0] * 1_000,
        "temperature": 25.0,
        "turns": 20,
        "area_m2": 1.0e-4,
        "path_length_m": 0.08,
        "sampling_rate_hz": 16_000_000,
        "initial_B": 0.0,
        "history_length": 100,
    }


def test_circuit_solver_accepts_current_prefix_without_target_current() -> None:
    service, _ = _service()
    payload = _base_payload()
    payload["current"] = [0.0] * 100

    result = service.circuit_demo_payload(payload)

    assert len(result["H"]) == 1_000
    assert len(result["predicted_H"]) == 900
    assert len(result["predicted_magnetizing_current_A"]) == 1_000
    assert result["reference_H"] is None
    assert result["metrics"]["validation_available"] is False
    assert result["metrics"]["sequence_relative_rmse_percent"] is None
    assert result["metrics"]["energy_relative_error_percent"] is None


def test_all_zero_full_reference_returns_null_metrics_not_an_error() -> None:
    service, _ = _service()
    payload = _base_payload()
    # Backward-compatible form: a full current supplies both prefix and reference.
    payload["current"] = [0.0] * 1_000

    result = service.circuit_demo_payload(payload)

    assert result["metrics"]["validation_available"] is True
    assert result["metrics"]["sequence_relative_rmse_percent"] is None
    assert "all-zero" in result["metrics"]["sequence_metric_reason"]
    assert result["metrics"]["energy_relative_error_percent"] is None
    assert result["metrics"]["energy_metric_reason"]


def test_explicit_full_reference_enables_validation_metrics() -> None:
    service, _ = _service()
    payload = _base_payload()
    payload["known_H"] = [0.0] * 100
    payload["reference_current"] = np.linspace(0.1, 0.2, 1_000).tolist()
    payload["reference_loss_density"] = 1.0

    result = service.circuit_demo_payload(payload)

    assert result["metrics"]["validation_available"] is True
    assert result["metrics"]["sequence_relative_rmse_percent"] is not None
    assert result["metrics"]["energy_relative_error_percent"] is not None
    assert len(result["reference_H"]) == 1_000


def test_material_c_always_uses_zero_placeholder_for_solver_input() -> None:
    service, predictor = _service()
    payload = _base_payload()
    payload.update({"material": "C", "history_length": 1, "current": [1.0]})

    result = service.circuit_demo_payload(payload)

    assert predictor.last_known_h.tolist() == [0.0]
    assert result["prediction"]["known_H_source"] == "material_C_zero_placeholder"
