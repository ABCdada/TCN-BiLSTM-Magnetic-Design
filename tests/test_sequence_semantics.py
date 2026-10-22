from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from magnetic_model.inference import Predictor


def test_material_c_reuses_zero_placeholder_in_every_stitched_window(monkeypatch) -> None:
    predictor = Predictor(manager=SimpleNamespace())
    supplied_prefixes: list[np.ndarray] = []

    def fake_predict_window(material, B, temperature, known_H, **kwargs):
        supplied_prefixes.append(np.asarray(known_H, dtype=np.float64).copy())
        return SimpleNamespace(predicted_H=np.ones(999, dtype=np.float64))

    monkeypatch.setattr(predictor, "predict_window", fake_predict_window)
    result = predictor.predict_sequence(
        "C",
        np.linspace(-0.1, 0.1, 1_500),
        25.0,
        [0.0],
        mode="1-999",
        hop_length=500,
    )

    assert len(supplied_prefixes) == result["window_count"]
    assert all(prefix.tolist() == [0.0] for prefix in supplied_prefixes)
    assert len(result["predicted_H"]) == 1_499


def test_recursive_sequence_carries_last_state_and_trims_padding(monkeypatch) -> None:
    predictor = Predictor(manager=SimpleNamespace())
    supplied_prefixes: list[np.ndarray] = []
    supplied_b: list[np.ndarray] = []

    def fake_predict_window(material, B, temperature, known_H, **kwargs):
        prefix = np.asarray(known_H, dtype=np.float64).copy()
        supplied_prefixes.append(prefix)
        supplied_b.append(np.asarray(B, dtype=np.float64).copy())
        predicted = prefix[0] + np.arange(1, 1_000, dtype=np.float64)
        return SimpleNamespace(predicted_H=predicted)

    monkeypatch.setattr(predictor, "predict_window", fake_predict_window)
    b = np.linspace(-0.2, 0.3, 2_100)
    result = predictor.predict_recursive_sequence(
        "C", b, 25.0, [3.0], mode="1-999"
    )

    assert result["stitching"] == "carry_last_h_overlap_one"
    assert result["hop_length"] == 999
    assert result["window_count"] == 3
    assert [prefix.tolist() for prefix in supplied_prefixes] == [
        [3.0],
        [1_002.0],
        [2_001.0],
    ]
    np.testing.assert_allclose(result["H"], 3.0 + np.arange(b.size))
    assert supplied_b[1][0] == b[999]
    assert supplied_b[2][0] == b[1_998]
    assert np.all(supplied_b[2][101:] == b[-1])
    assert result["windows"][-1] == {
        "start": 1_998,
        "stop": 2_100,
        "model_stop": 2_998,
        "valid_length": 102,
        "padded_samples": 898,
    }


def test_recursive_sequence_has_optional_explicit_window_limit(monkeypatch) -> None:
    predictor = Predictor(manager=SimpleNamespace())
    calls = 0

    def fake_predict_window(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(predicted_H=np.zeros(999, dtype=np.float64))

    monkeypatch.setattr(predictor, "predict_window", fake_predict_window)

    with np.testing.assert_raises_regex(ValueError, "requires 3 recursive model windows"):
        predictor.predict_recursive_sequence(
            "C", np.zeros(2_100), 25.0, [0.0], max_windows=2
        )

    assert calls == 0
