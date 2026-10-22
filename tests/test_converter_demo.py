from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app import create_app
from magnetic_model import InferenceService, simulate_converter_window


class _FakePredictor:
    """Deterministic predictor that preserves the supplied history."""

    def predict_window(self, material, B, temperature, known_H, **kwargs):
        output = np.linspace(-2.0, 2.0, 1_000, dtype=np.float64)
        output[: len(known_H)] = np.asarray(known_H, dtype=np.float64)
        return SimpleNamespace(H=output)


def _payload(topology: str = "single_phase_full_bridge") -> dict:
    return {
        "material": "A",
        "mode": "100-900",
        "topology": topology,
        "temperature": 25.0,
        "turns": 20,
        "area_m2": 1.0e-4,
        "path_length_m": 0.08,
        "sampling_rate_hz": 16_000_000,
        "initial_B": 0.0,
        "dc_voltage": 800.0,
        "grid_voltage_rms": 220.0 if topology.startswith("single") else 380.0,
        "grid_frequency_hz": 50.0,
        "switching_frequency_hz": 100_000.0,
        "filter_inductance_h": 1.45e-4,
        "series_resistance_ohm": 0.2,
        "initial_current_A": 0.5,
        "grid_phase_deg": 17.0,
    }


@pytest.mark.parametrize(
    "topology",
    [
        "single_phase_full_bridge",
        "three_phase_full_bridge",
        "three_phase_t_type",
    ],
)
def test_converter_window_is_deterministic_and_physically_consistent(topology) -> None:
    payload = _payload(topology)
    kwargs = {
        "topology": topology,
        "dc_voltage": payload["dc_voltage"],
        "grid_voltage_rms": payload["grid_voltage_rms"],
        "grid_frequency_hz": payload["grid_frequency_hz"],
        "switching_frequency_hz": payload["switching_frequency_hz"],
        "filter_inductance_h": payload["filter_inductance_h"],
        "series_resistance_ohm": payload["series_resistance_ohm"],
        "initial_current_a": payload["initial_current_A"],
        "grid_phase_deg": payload["grid_phase_deg"],
    }
    first = simulate_converter_window(**kwargs)
    second = simulate_converter_window(**kwargs)

    assert len(first.time_s) == 1_000
    assert np.array_equal(first.inverter_voltage_v, second.inverter_voltage_v)
    assert np.array_equal(first.nominal_current_a, second.nominal_current_a)
    expected = (
        first.inverter_voltage_v
        - first.grid_voltage_v
        - payload["series_resistance_ohm"] * first.nominal_current_a
    )
    assert np.allclose(first.induced_voltage_v, expected, rtol=0.0, atol=1e-12)


def test_supported_topologies_produce_distinct_switching_waveforms() -> None:
    two_level = simulate_converter_window(
        **{
            "topology": "three_phase_full_bridge",
            "dc_voltage": 800.0,
            "grid_voltage_rms": 380.0,
            "grid_frequency_hz": 50.0,
            "switching_frequency_hz": 100_000.0,
            "filter_inductance_h": 1.45e-4,
            "series_resistance_ohm": 0.2,
        }
    )
    three_level = simulate_converter_window(
        **{
            "topology": "three_phase_t_type",
            "dc_voltage": 800.0,
            "grid_voltage_rms": 380.0,
            "grid_frequency_hz": 50.0,
            "switching_frequency_hz": 100_000.0,
            "filter_inductance_h": 1.45e-4,
            "series_resistance_ohm": 0.2,
        }
    )

    assert not np.array_equal(
        two_level.inverter_voltage_v, three_level.inverter_voltage_v
    )
    assert 0.0 in np.unique(three_level.inverter_voltage_v)


def test_converter_service_does_not_treat_nominal_current_as_reference() -> None:
    service = InferenceService()
    service.predictor = _FakePredictor()

    result = service.converter_demo_payload(_payload())

    assert len(result["simulation"]["induced_voltage_V"]) == 1_000
    assert len(result["simulation"]["nominal_current_A"]) == 1_000
    assert len(result["H"]) == 1_000
    assert len(result["predicted_H"]) == 900
    assert len(result["predicted_magnetizing_current_A"]) == 1_000
    assert result["reference_H"] is None
    assert result["metrics"]["validation_available"] is False
    assert result["metrics"]["sequence_relative_rmse_percent"] is None
    assert result["metrics"]["energy_relative_error_percent"] is None
    assert "not measured data" in result["scope_note"]


def test_converter_endpoint_uses_same_service_contract() -> None:
    application = create_app({"TESTING": True})
    application.extensions["magnetic_service"].predictor = _FakePredictor()

    with application.test_client() as client:
        response = client.post("/api/converter-demo", json=_payload("three_phase_t_type"))

    assert response.status_code == 200
    result = response.get_json()
    assert result["simulation"]["topology"] == "three_phase_t_type"
    assert result["metrics"]["validation_available"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("dc_voltage", float("nan")),
        ("filter_inductance_h", 0.0),
        ("series_resistance_ohm", -0.1),
    ],
)
def test_converter_service_rejects_invalid_numerical_parameters(field, value) -> None:
    service = InferenceService()
    service.predictor = _FakePredictor()
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError):
        service.converter_demo_payload(payload)


def test_converter_service_rejects_unknown_topology() -> None:
    service = InferenceService()
    service.predictor = _FakePredictor()
    payload = _payload()
    payload["topology"] = "unknown"

    with pytest.raises(ValueError, match="topology"):
        service.converter_demo_payload(payload)


def test_converter_service_allows_ideal_zero_series_resistance() -> None:
    service = InferenceService()
    service.predictor = _FakePredictor()
    payload = _payload()
    payload["series_resistance_ohm"] = 0.0

    result = service.converter_demo_payload(payload)

    assert len(result["simulation"]["nominal_current_A"]) == 1_000
