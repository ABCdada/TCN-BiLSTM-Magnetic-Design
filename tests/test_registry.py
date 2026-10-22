from __future__ import annotations

import pytest

from magnetic_model import MATERIALS, MODEL_REGISTRY, build_model, count_parameters


EXPECTED = {
    "A": ("3C92", 497_793),
    "B": ("3C95", 2_092_545),
    "C": ("FEC007", 2_092_545),
    "D": ("FEC014", 2_512_593),
    "E": ("T37", 2_092_865),
}


def test_material_registry_is_complete_and_unambiguous() -> None:
    assert set(MATERIALS) == set(EXPECTED)
    assert set(MODEL_REGISTRY) == set(EXPECTED)
    for model_id, (grade, parameter_count) in EXPECTED.items():
        spec = MATERIALS[model_id]
        assert spec.grade == grade
        assert spec.parameter_count == parameter_count


@pytest.mark.slow
@pytest.mark.parametrize("model_id", sorted(EXPECTED))
def test_constructed_model_parameter_counts(model_id: str) -> None:
    model = build_model(model_id)
    assert count_parameters(model) == EXPECTED[model_id][1]


def test_unknown_material_is_rejected() -> None:
    with pytest.raises((KeyError, ValueError), match="material|model|unknown|Unsupported"):
        build_model("not-a-material")
