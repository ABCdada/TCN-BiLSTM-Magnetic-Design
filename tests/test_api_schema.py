from __future__ import annotations

import pytest

from app import app


@pytest.fixture()
def client():
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


def assert_error_envelope(response, expected_status: int = 400) -> None:
    assert response.status_code == expected_status
    payload = response.get_json()
    assert isinstance(payload, dict)
    assert isinstance(payload.get("error"), dict)
    assert isinstance(payload["error"].get("code"), str)
    assert isinstance(payload["error"].get("message"), str)


@pytest.mark.api
def test_health_schema(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert "version" in payload


@pytest.mark.api
def test_model_discovery_schema(client) -> None:
    response = client.get("/api/models")
    assert response.status_code == 200
    payload = response.get_json()
    models = payload["models"]
    assert {record["id"] for record in models} == {"A", "B", "C", "D", "E"}
    assert {record["grade"] for record in models} == {
        "3C92",
        "3C95",
        "FEC007",
        "FEC014",
        "T37",
    }
    assert all(isinstance(record["parameter_count"], int) for record in models)


@pytest.mark.api
def test_cold_start_model_discovery_is_limited_to_browser_materials(client) -> None:
    response = client.get("/api/cold-start-models")
    assert response.status_code == 200
    models = response.get_json()["models"]
    assert [record["id"] for record in models] == ["A", "B", "D", "E"]
    assert [record["grade"] for record in models] == [
        "3C92",
        "3C95",
        "FEC014",
        "T37",
    ]
    assert [record["label"] for record in models] == [
        "3C92",
        "3C95",
        "FEC014",
        "T37",
    ]
    assert all(record["supported_modes"] == ["1-999"] for record in models)
    assert all(record["known_h_placeholder_zero"] is True for record in models)


@pytest.mark.api
def test_cold_start_predict_rejects_removed_material_id_five(client) -> None:
    response = client.post(
        "/api/predict-cold-start",
        json={"materialId": 5, "sample": {"B": [0.0, 0.1], "T": 25.0}},
    )

    assert_error_envelope(response)
    assert "1 to 4" in response.get_json()["error"]["message"]


@pytest.mark.api
def test_predict_window_rejects_wrong_length_before_inference(client) -> None:
    response = client.post(
        "/api/predict-window",
        json={
            "material": "A",
            "B": [0.0] * 999,
            "temperature": 25.0,
            "known_H": [0.0] * 100,
            "mode": "100-900",
            "sampling_rate_hz": 16_000_000,
        },
    )
    assert_error_envelope(response)


@pytest.mark.api
def test_predict_window_rejects_inconsistent_mode_and_history(client) -> None:
    response = client.post(
        "/api/predict-window",
        json={
            "material": "A",
            "B": [0.0] * 1_000,
            "temperature": 25.0,
            "known_H": [0.0] * 500,
            "mode": "100-900",
            "sampling_rate_hz": 16_000_000,
        },
    )
    assert_error_envelope(response)


@pytest.mark.api
def test_circuit_demo_rejects_nonphysical_geometry(client) -> None:
    response = client.post(
        "/api/circuit-demo",
        json={
            "material": "A",
            "induced_voltage": [0.0] * 1_000,
            "current": [0.0] * 1_000,
            "temperature": 25.0,
            "turns": 0,
            "area_m2": 1.0e-4,
            "path_length_m": 0.08,
            "sampling_rate_hz": 16_000_000,
            "initial_B": 0.0,
            "history_length": 100,
        },
    )
    assert_error_envelope(response)


@pytest.mark.api
def test_material_c_rejects_nonzero_cold_start_placeholder(client) -> None:
    response = client.post(
        "/api/predict-window",
        json={
            "material": "C",
            "B": [0.0] * 1_000,
            "temperature": 25.0,
            "known_H": [1.0],
            "mode": "1-999",
            "sampling_rate_hz": 16_000_000,
        },
    )
    assert_error_envelope(response)


@pytest.mark.api
def test_predict_sequence_rejects_excessive_window_count(client) -> None:
    response = client.post(
        "/api/predict-sequence",
        json={
            "material": "A",
            "B": [0.0] * 1_300,
            "temperature": 25.0,
            "known_H": [0.0] * 100,
            "mode": "100-900",
            "hop_length": 1,
            "sampling_rate_hz": 16_000_000,
        },
    )
    assert_error_envelope(response)
