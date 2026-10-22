from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np
import pytest

from app import app
from magnetic_model.service import InferenceService


@pytest.mark.api
@pytest.mark.parametrize("path", ["/", "/design", "/fine_tune", "/test", "/document"])
def test_reference_ui_routes_render(path: str) -> None:
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get(path)
    assert response.status_code == 200


@pytest.mark.api
def test_model_libraries_navigation_baselines_and_hyperparameters_are_usable() -> None:
    app.config.update(TESTING=True)
    with app.test_client() as client:
        index_response = client.get("/")
        library_response = client.get("/fine_tune")

    index_html = index_response.get_data(as_text=True)
    library_html = library_response.get_data(as_text=True)
    model_link = re.search(
        r'<a(?=[^>]*data-page="model-libs")(?=[^>]*href="/fine_tune")[^>]*>',
        index_html,
    )
    base_model_select = re.search(
        r'<select[^>]*id="base-model"[^>]*>(.*?)</select>',
        library_html,
        re.DOTALL,
    )

    assert index_response.status_code == 200
    assert library_response.status_code == 200
    assert model_link is not None
    assert base_model_select is not None
    assert re.findall(
        r'<option\s+value="([^"]+)"[^>]*>([^<]+)</option>',
        base_model_select.group(1),
    ) == [
        ("A", "3C92"),
        ("B", "3C95"),
        ("D", "FEC014"),
        ("E", "T37"),
    ]
    assert "FEC007" not in library_html
    assert "3C90" not in library_html
    assert "Material A - 3C92" not in library_html
    assert "Material ${materialId}" not in library_html
    assert "C: { grade:" not in library_html
    for control_id in (
        "epochs",
        "learning-rate",
        "batch-size",
        "validation-split",
        "window-size",
        "loss-type",
        "frozen-layers",
    ):
        opening_tag = re.search(
            rf'<(?:input|select)(?=[^>]*\bid="{re.escape(control_id)}")[^>]*>',
            library_html,
        )
        assert opening_tag is not None
        assert re.search(r"\sdisabled(?:\s|=|>)", opening_tag.group(0)) is None
        assert "aria-disabled" not in opening_tag.group(0)
        assert re.search(r"\sreadonly(?:\s|=|>)", opening_tag.group(0)) is None


@pytest.mark.api
def test_design_center_exposes_four_named_cold_start_materials() -> None:
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get("/design")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "const DESIGN_DEFAULT_MATERIAL_ID = 1" in html
    assert "name: '3C92'" in html
    assert "name: '3C95'" in html
    assert "name: 'FEC014'" in html
    assert "name: 'T37'" in html
    assert "name: 'FEC007'" not in html
    assert "name: '3C90'" not in html
    assert "model: 'C'" not in html
    assert "model: 'F'" not in html
    assert "{ id: 3, model: 'D', name: 'FEC014'" in html
    assert "{ id: 4, model: 'E', name: 'T37'" in html
    assert "${mat.name}</span>" in html
    assert "${mat.model} - ${mat.name}" not in html
    assert "${mat.model}" not in html
    assert "let selectedMaterialIds = [DESIGN_DEFAULT_MATERIAL_ID]" in html
    assert "let activeMatIds = [DESIGN_DEFAULT_MATERIAL_ID]" in html
    assert "fetch('/api/predict-cold-start'" in html
    assert "materialId: material.id" in html
    assert "for (const material of selectedMaterials)" in html
    assert "Promise.all(predictionPromises)" not in html
    assert "duration_semantics: 'endpoint_span'" in html
    assert "sampling_mode: 'original_design_raw'" in html
    assert "const DESIGN_SAMPLES_PER_SWITCHING_CYCLE = 10" in html
    assert "const DESIGN_PREDICTION_WINDOW_SIZE = 1_000" in html
    assert "const DESIGN_PREDICTION_WINDOW_STRIDE = DESIGN_PREDICTION_WINDOW_SIZE" in html
    assert "const DESIGN_REFERENCE_PATH_LENGTH_MM = 85.5" in html
    assert "DESIGN_REFERENCE_PATH_LENGTH_MM / material.le" in html
    assert "H_pred: H_pred_scaled" in html
    assert "H_pred_raw: H_pred_raw" in html
    assert "scale_factor: scaleFactor" in html
    assert "normalizePredictionSelection" in html
    assert "estimatePredictionWindowCount" in html
    assert "original-compatible 1,000-sample windows" in html
    assert "displayed samples used directly" in html
    assert "DESIGN_MAX_PREDICTION_SAMPLES" not in html
    assert "automatically limited to 3.125 ms" not in html
    assert "The selected interval is too long" not in html
    assert "errorPayload?.error?.message" in html
    assert "Prediction failed, please try again" not in html
    assert ".flex.flex-wrap.gap-3..text-xs" not in html
    assert "Compatibility mode:" not in html
    assert "Core Loss Comparison (W/m³, Total W)" in html
    assert "calculateCoreLossByIntegration(B_seq, H_seq, time_seq, volume_m3, switchingFreq)" in html
    assert "energy_density = Math.abs(energy_density)" in html
    assert "var T_period = 1.0 / switchingFreq" in html
    assert "var loss_per_volume = energy_density / T_period" in html
    assert "var loss_total = loss_per_volume * volume_m3" in html
    assert "conditions.frequency" in html
    assert "Loss per volume: ${lossPerVol.toFixed(2)} W/m³" in html
    assert "Total loss: ${lossTotal.toFixed(2)} W" in html
    assert "calculateIntervalEnergyExchangeRate" not in html
    assert "Open path: total power includes reversible energy exchange" not in html


@pytest.mark.api
def test_design_center_is_single_phase_with_independent_core_geometry() -> None:
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get("/design")
        image_response = client.get("/static/images/inductor-configurable.png")

    html = response.get_data(as_text=True)
    topology_select = re.search(
        r'<select[^>]*id="topology-select"[^>]*>(.*?)</select>',
        html,
        re.DOTALL,
    )

    assert response.status_code == 200
    assert image_response.status_code == 200
    assert topology_select is not None
    assert re.findall(r'<option\s+value="([^"]+)"', topology_select.group(1)) == [
        "Single-Phase-full-bridge"
    ]
    assert "Grid Voltage (RMS)" in html
    assert "R: 200, r: 80, d: 60" in html
    assert 'min="1" max="500"' in html
    assert 'min="0" max="400"' in html
    assert 'min="1" max="2000"' in html
    assert "autoCalcMatParams" not in html
    assert "0.4 * numValue" not in html
    assert "0.3 * numValue" not in html
    assert "/static/images/inductor-configurable.png" in html


@pytest.mark.api
def test_sidebar_team_contact_block_is_removed_and_documents_match_design_scope() -> None:
    app.config.update(TESTING=True)
    with app.test_client() as client:
        index_response = client.get("/")
        document_response = client.get("/document")

    index_html = index_response.get_data(as_text=True)
    document_html = document_response.get_data(as_text=True)
    assert index_response.status_code == 200
    assert document_response.status_code == 200
    assert 'id="team-info"' not in index_html
    assert "mailto:" not in index_html.lower()
    assert "mailto:" not in document_html.lower()
    assert "The Design Center uses the single-phase converter topology" in document_html
    assert "3Phase Full Bridge" not in document_html
    assert "3Phase T-Type Three-Level" not in document_html


@pytest.mark.api
def test_test_center_exposes_four_named_cold_start_materials() -> None:
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get("/test")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "const TEST_DEFAULT_MATERIAL_ID = 1" in html
    assert "name: '3C92'" in html
    assert "name: '3C95'" in html
    assert "name: 'FEC014'" in html
    assert "name: 'T37'" in html
    assert "name: 'FEC007'" not in html
    assert "name: '3C90'" not in html
    assert "model: 'C'" not in html
    assert "model: 'F'" not in html
    assert "{ id: 3, model: 'D', name: 'FEC014'" in html
    assert "{ id: 4, model: 'E', name: 'T37'" in html
    assert "${mat.name}</div>" in html
    assert "${mat.model} - ${mat.name}" not in html
    assert "${mat.model}" not in html
    assert "let selectedMaterialIds = [TEST_DEFAULT_MATERIAL_ID]" in html
    assert "let activeMatIds = [TEST_DEFAULT_MATERIAL_ID]" in html
    assert "fetch('/api/predict-cold-start'" in html
    assert "materialId: material.id" in html
    assert "for (const material of selectedMaterials)" in html
    assert "Promise.all(predictionPromises)" not in html
    assert "Native cold-start mode:" in html
    assert "Select Materials for Comparison" in html
    assert "button.disabled = true" not in html
    assert "toggleMaterialSelection(mat.id)" in html
    assert "duration_semantics: 'endpoint_span'" in html
    assert "Compatibility mode:" not in html
    assert "errorPayload?.error?.message" in html
    assert "new Array(bDataFromFile.length).fill(0)" not in html
    assert "DESIGN_REFERENCE_PATH_LENGTH_MM" not in html
    assert "H_pred_scaled" not in html


@pytest.mark.api
def test_reference_predict_route_keeps_flat_response_contract(monkeypatch) -> None:
    service = app.extensions["magnetic_service"]
    monkeypatch.setattr(
        service,
        "predict_legacy_payload",
        lambda payload: {"H_pred": [0.0, 1.0], "mu": [0.0, 0.1], "core_loss": 2.0},
    )
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.post(
            "/api/predict",
            json={"materialId": 1, "sample": {"B": [0.0, 0.1], "T": 25}},
        )
    assert response.status_code == 200
    assert response.get_json() == {
        "H_pred": [0.0, 1.0],
        "mu": [0.0, 0.1],
        "core_loss": 2.0,
    }


@pytest.mark.api
@pytest.mark.parametrize(
    ("material_id", "expected_material", "expected_history"),
    [(1, "A", 100), (2, "B", 100), (3, "C", 1), (4, "D", 100), (5, "E", 100)],
)
def test_reference_predict_material_mapping_and_lengths(
    monkeypatch, material_id: int, expected_material: str, expected_history: int
) -> None:
    service = InferenceService()
    captured = {}

    def fake_window(material, B, temperature, known_H, **kwargs):
        captured.update(
            material=material,
            B=np.asarray(B),
            known_H=np.asarray(known_H),
            mode=kwargs["mode"],
        )
        h = np.arange(len(B), dtype=np.float64)
        h[: len(known_H)] = 0.0
        return SimpleNamespace(H=h)

    monkeypatch.setattr(service.predictor, "predict_window", fake_window)
    payload = {
        "materialId": material_id,
        "sample": {
            "B": np.linspace(-0.05, 0.05, 200).tolist(),
            "T": 25.0,
            "real_time": 200 / 16_000_000.0,
        },
    }
    result = service.predict_legacy_payload(payload)

    assert captured["material"] == expected_material
    assert len(captured["known_H"]) == expected_history
    assert np.all(captured["known_H"] == 0.0)
    assert len(captured["B"]) == 1_000
    assert len(result["H_pred"]) == 200
    assert len(result["mu"]) == 200
    assert np.isfinite(result["mu"]).all()
    assert np.isfinite(result["core_loss"])
    assert result["H_pred"][0] == 0.0


@pytest.mark.api
def test_material_c_legacy_request_uses_exact_native_cold_start(monkeypatch) -> None:
    service = InferenceService()
    captured = {}
    original_b = np.linspace(-0.1, 0.1, 1_000)

    def fake_window(material, B, temperature, known_H, **kwargs):
        captured.update(
            material=material,
            B=np.asarray(B),
            known_H=np.asarray(known_H),
            mode=kwargs["mode"],
        )
        h = np.linspace(0.0, 1.0, len(B), dtype=np.float64)
        h[0] = known_H[0]
        return SimpleNamespace(H=h)

    monkeypatch.setattr(service.predictor, "predict_window", fake_window)
    result = service.predict_legacy_payload(
        {
            "materialId": 3,
            "sample": {
                "B": original_b.tolist(),
                "T": 25.0,
                "real_time": 999 / 16_000_000.0,
                "duration_semantics": "endpoint_span",
            },
        }
    )

    assert captured["material"] == "C"
    assert captured["mode"] == "1-999"
    assert captured["B"].shape == (1_000,)
    np.testing.assert_allclose(captured["B"], original_b)
    np.testing.assert_array_equal(captured["known_H"], np.array([0.0]))
    assert len(result["H_pred"]) == 1_000
    assert result["H_pred"][0] == 0.0
    assert result["initial_state"] == "material_C_native_cold_start_H0_zero"


@pytest.mark.api
def test_material_c_design_interval_boundary_is_inclusive(monkeypatch) -> None:
    service = InferenceService()
    captured = {}

    def fake_sequence(material, B, temperature, known_H, **kwargs):
        captured["length"] = len(B)
        captured["max_windows"] = kwargs["max_windows"]
        return {
            "H": np.zeros(len(B), dtype=np.float64),
            "window_count": int(np.ceil((len(B) - 1) / 999)),
            "hop_length": 999,
            "stitching": "carry_last_h_overlap_one",
        }

    monkeypatch.setattr(
        service.predictor, "predict_recursive_sequence", fake_sequence
    )
    payload = {
        "materialId": 3,
        "sample": {
            "B": np.linspace(-0.1, 0.1, 10).tolist(),
            "T": 25.0,
            "real_time": (50_000 - 1) / 16_000_000.0,
            "duration_semantics": "endpoint_span",
        },
    }

    result = service.predict_legacy_payload(payload)
    assert captured["length"] == 50_000
    assert captured["max_windows"] == 2_048
    assert len(result["H_pred"]) == 10
    assert result["stitch"] == {
        "strategy": "carry_last_h_overlap_one",
        "window_length": 1_000,
        "hop_length": 999,
        "window_count": 51,
        "initial_H_A_per_m": 0.0,
        "model_sample_count": 50_000,
        "returned_sample_count": 10,
    }

    payload["sample"]["real_time"] = 50_000 / 16_000_000.0
    result = service.predict_legacy_payload(payload)
    assert captured["length"] == 50_001
    assert len(result["H_pred"]) == 10


@pytest.mark.api
def test_original_design_sampling_uses_raw_nonoverlapping_windows(monkeypatch) -> None:
    service = InferenceService()
    original_b = np.linspace(-0.2, 0.2, 2_200)
    calls = []

    def fake_window(material, B, temperature, known_H, **kwargs):
        b = np.asarray(B, dtype=np.float64)
        known = float(known_H[0])
        h = np.linspace(known, known + 1.0, len(b), dtype=np.float64)
        h[0] = known
        calls.append({"B": b.copy(), "known": known})
        return SimpleNamespace(H=h)

    monkeypatch.setattr(service.predictor, "predict_window", fake_window)
    result = service.predict_legacy_payload(
        {
            "materialId": 3,
            "sample": {
                "B": original_b.tolist(),
                "T": 25.0,
                "real_time": (len(original_b) - 1) / 750_000.0,
                "duration_semantics": "endpoint_span",
                "sampling_mode": "original_design_raw",
            },
        }
    )

    assert len(calls) == 3
    np.testing.assert_allclose(calls[0]["B"], original_b[:1_000])
    np.testing.assert_allclose(calls[1]["B"], original_b[1_000:2_000])
    np.testing.assert_allclose(calls[2]["B"], original_b[1_200:2_200])
    assert calls[0]["known"] == 0.0
    assert calls[1]["known"] == pytest.approx(1.0)
    assert calls[2]["known"] == pytest.approx(2.0)
    assert len(result["H_pred"]) == len(original_b)
    assert result["stitch"] == {
        "strategy": "original_design_nonoverlap_carry_last_h",
        "window_length": 1_000,
        "hop_length": 1_000,
        "window_count": 3,
        "initial_H_A_per_m": 0.0,
        "model_sample_count": len(original_b),
        "returned_sample_count": len(original_b),
    }
    assert result["sampling_mode"] == "original_design_raw"


@pytest.mark.api
def test_material_c_legacy_accepts_more_than_100000_raw_samples(monkeypatch) -> None:
    service = InferenceService()
    raw_length = 100_001
    captured = {}

    def fake_sequence(material, B, temperature, known_H, **kwargs):
        captured.update(
            material=material,
            B_length=len(B),
            temperature_length=len(temperature),
            known_H=list(known_H),
        )
        return {
            "H": np.linspace(0.0, 1.0, len(B), dtype=np.float64),
            "window_count": int(np.ceil((len(B) - 1) / 999)),
            "hop_length": 999,
            "stitching": "carry_last_h_overlap_one",
        }

    monkeypatch.setattr(
        service.predictor, "predict_recursive_sequence", fake_sequence
    )
    b = np.linspace(-0.1, 0.1, raw_length, dtype=np.float64)
    result = service.predict_legacy_payload(
        {
            "materialId": 3,
            "sample": {"B": b.tolist(), "T": 25.0},
        }
    )

    assert captured == {
        "material": "C",
        "B_length": raw_length,
        "temperature_length": raw_length,
        "known_H": [0.0],
    }
    assert len(result["H_pred"]) == raw_length
    assert result["stitch"]["window_count"] == 101
    assert result["stitch"]["returned_sample_count"] == raw_length


@pytest.mark.api
def test_material_c_legacy_rejects_more_than_two_million_model_samples() -> None:
    service = InferenceService()
    with pytest.raises(ValueError, match="16 MHz Design Center limit"):
        service.predict_legacy_payload(
            {
                "materialId": 3,
                "sample": {
                    "B": [0.0, 0.1],
                    "T": 25.0,
                    "real_time": 2_000_000 / 16_000_000.0,
                    "duration_semantics": "endpoint_span",
                },
            }
        )


@pytest.mark.api
def test_reference_predict_rejects_invalid_material_before_inference() -> None:
    service = InferenceService()
    with pytest.raises(ValueError, match="materialId"):
        service.predict_legacy_payload(
            {"materialId": 6, "sample": {"B": [0.0, 0.1], "T": 25.0}}
        )


@pytest.mark.api
@pytest.mark.parametrize(
    ("material_id", "expected_material"),
    [(1, "A"), (2, "B"), (3, "D"), (4, "E")],
)
def test_cold_start_material_ids_map_to_browser_registry(
    monkeypatch, material_id: int, expected_material: str
) -> None:
    service = app.extensions["cold_start_magnetic_service"]
    captured = {}

    def fake_window(material, B, temperature, known_H, **kwargs):
        captured.update(
            material=material,
            known_H=np.asarray(known_H),
            mode=kwargs["mode"],
        )
        return SimpleNamespace(H=np.zeros(len(B), dtype=np.float64))

    monkeypatch.setattr(service.predictor, "predict_window", fake_window)
    result = service.predict_legacy_payload(
        {"materialId": material_id, "sample": {"B": [0.0, 0.1], "T": 25.0}}
    )

    assert captured["material"] == expected_material
    assert captured["mode"] == "1-999"
    np.testing.assert_array_equal(captured["known_H"], np.array([0.0]))
    assert result["initial_state"] == (
        f"material_{expected_material}_native_cold_start_H0_zero"
    )


@pytest.mark.api
def test_cold_start_material_id_five_is_rejected() -> None:
    service = app.extensions["cold_start_magnetic_service"]

    with pytest.raises(ValueError, match="1 to 4"):
        service.predict_legacy_payload(
            {"materialId": 5, "sample": {"B": [0.0, 0.1], "T": 25.0}}
        )


@pytest.mark.api
def test_reference_predict_material_id_takes_priority_over_model_id(monkeypatch) -> None:
    service = InferenceService()
    captured = {}

    def fake_window(material, B, temperature, known_H, **kwargs):
        captured["material"] = material
        return SimpleNamespace(H=np.zeros(len(B), dtype=np.float64))

    monkeypatch.setattr(service.predictor, "predict_window", fake_window)
    service.predict_legacy_payload(
        {
            "materialId": 1,
            "modelId": "E",
            "sample": {"B": [0.0, 0.1], "T": 25.0},
        }
    )
    assert captured["material"] == "A"


@pytest.mark.api
def test_reference_predict_reports_h_db_interval_energy_rate(monkeypatch) -> None:
    service = InferenceService()

    def fake_window(material, B, temperature, known_H, **kwargs):
        h = 2.0 * np.asarray(B, dtype=np.float64)
        h[: len(known_H)] = 0.0
        return SimpleNamespace(H=h)

    monkeypatch.setattr(service.predictor, "predict_window", fake_window)
    result = service.predict_legacy_payload(
        {"materialId": 1, "sample": {"B": [0.0, 1.0, 2.0], "T": 25.0}}
    )

    # Cropped compatibility output is H=[0, 2, 4], hence the trapezoidal
    # integral sum(H_mid * dB)=1*1+3*1=4 J/m^3 over 3/16e6 seconds.
    assert result["core_loss"] == pytest.approx(4.0 / (3.0 / 16_000_000.0))
    assert result["energy_quantity"] == "interval_magnetic_energy_exchange_rate"
    assert result["energy_unit"] == "W/m^3"
