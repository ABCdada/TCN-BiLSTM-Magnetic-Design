from __future__ import annotations

import numpy as np
import pytest

from magnetic_model import closed_cycle_loss_density, integrate_h_db


def test_h_db_integral_matches_hand_calculated_square_loop() -> None:
    # Path: lower branch from B=-1 to 1 at H=-1, then upper branch back to
    # B=-1 at H=1. Vertical segments have dB=0. Trapezoidal integration gives
    # (-1)*(+2) + (+1)*(-2) = -4 J/m3 for this orientation.
    b = np.asarray([-1.0, 1.0, 1.0, -1.0, -1.0])
    h = np.asarray([-1.0, -1.0, 1.0, 1.0, -1.0])

    assert integrate_h_db(b, h) == pytest.approx(-4.0, abs=1e-12)
    assert closed_cycle_loss_density(b, h) == pytest.approx(4.0, abs=1e-12)


def test_open_path_is_not_silently_reported_as_closed_cycle_loss() -> None:
    b = np.asarray([0.0, 0.5, 1.0])
    h = np.asarray([0.0, 1.0, 2.0])
    assert integrate_h_db(b, h) == pytest.approx(1.0, abs=1e-12)

    with pytest.raises(ValueError, match="closed|closure|cycle"):
        closed_cycle_loss_density(b, h)


@pytest.mark.parametrize(
    "b,h",
    [
        ([0.0, 1.0], [0.0]),
        ([0.0, float("nan")], [0.0, 1.0]),
        ([0.0], [0.0]),
    ],
)
def test_h_db_integral_rejects_invalid_inputs(b: list[float], h: list[float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        integrate_h_db(np.asarray(b), np.asarray(h))
