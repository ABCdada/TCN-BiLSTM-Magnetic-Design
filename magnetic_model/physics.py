"""Numerically consistent magnetic and circuit post-processing helpers."""

from __future__ import annotations

import numpy as np


def _finite_vector(values, name: str, *, minimum_length: int = 2) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if array.size < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} samples")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def integrate_h_db(B, H, *, close_path: bool = False) -> float:
    """Integrate H dB with one trapezoidal convention.

    The returned unit is J/m^3 when ``B`` is in tesla and ``H`` is in A/m.
    With ``close_path=False`` this is signed transient magnetic-energy
    exchange over the supplied interval.  ``close_path=True`` adds the final
    segment from the last sample back to the first and is appropriate only
    when the samples represent a complete cycle.
    """

    b = _finite_vector(B, "B")
    h = _finite_vector(H, "H")
    if b.size != h.size:
        raise ValueError("B and H must have the same number of samples")
    if close_path:
        b = np.concatenate((b, b[:1]))
        h = np.concatenate((h, h[:1]))
    return float(np.sum(0.5 * (h[:-1] + h[1:]) * np.diff(b), dtype=np.float64))


def closed_path_energy_density(B, H) -> float:
    """Return the non-negative H-dB integral of an explicitly closed path.

    Open transient paths are rejected rather than silently connected, because
    their integrals include reversible energy exchange and must not be labeled
    as irreversible cycle loss.  If the supplied path contains exactly one
    magnetic period, this quantity equals loss energy density per cycle; for a
    multi-period record it is the cumulative energy of the entire path.
    """

    b = _finite_vector(B, "B")
    h = _finite_vector(H, "H")
    if b.size != h.size:
        raise ValueError("B and H must have the same number of samples")
    if not (
        np.isclose(b[0], b[-1], rtol=1e-5, atol=1e-9)
        and np.isclose(h[0], h[-1], rtol=1e-5, atol=1e-7)
    ):
        raise ValueError(
            "closed-cycle loss requires a closed B-H path with matching endpoints"
        )
    return abs(integrate_h_db(b, h, close_path=False))


def closed_cycle_loss_density(B, H) -> float:
    """Backward-compatible alias for :func:`closed_path_energy_density`."""

    return closed_path_energy_density(B, H)


def voltage_to_flux_density(
    voltage,
    sampling_rate_hz: float,
    turns: float,
    area_m2: float,
    *,
    initial_B: float = 0.0,
) -> np.ndarray:
    """Reconstruct B from magnetic-core induced voltage.

    The caller must remove winding-resistance, leakage-inductance, and relevant
    parasitic voltage drops before using the terminal voltage in Faraday's law.
    """

    v = _finite_vector(voltage, "voltage")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not np.isfinite(turns) or turns <= 0:
        raise ValueError("turns must be positive")
    if not np.isfinite(area_m2) or area_m2 <= 0:
        raise ValueError("area_m2 must be positive")
    if not np.isfinite(initial_B):
        raise ValueError("initial_B must be finite")
    dt = 1.0 / sampling_rate_hz
    increments = 0.5 * (v[:-1] + v[1:]) * dt / (turns * area_m2)
    b = np.empty_like(v)
    b[0] = initial_B
    b[1:] = initial_B + np.cumsum(increments, dtype=np.float64)
    return b


def current_to_field_strength(current, turns: float, path_length_m: float) -> np.ndarray:
    """Convert winding current to H through Ampere's law for a toroidal core."""

    i = _finite_vector(current, "current", minimum_length=1)
    if not np.isfinite(turns) or turns <= 0:
        raise ValueError("turns must be positive")
    if not np.isfinite(path_length_m) or path_length_m <= 0:
        raise ValueError("path_length_m must be positive")
    return i * turns / path_length_m


def field_strength_to_magnetizing_current(H, turns: float, path_length_m: float) -> np.ndarray:
    """Convert reconstructed H back to magnetic-branch current for a solver."""

    h = _finite_vector(H, "H", minimum_length=1)
    if not np.isfinite(turns) or turns <= 0:
        raise ValueError("turns must be positive")
    if not np.isfinite(path_length_m) or path_length_m <= 0:
        raise ValueError("path_length_m must be positive")
    return h * path_length_m / turns


def relative_rmse_percent(predicted, measured) -> float:
    """Return the sequence relative RMSE used by the evaluation scripts."""

    pred = _finite_vector(predicted, "predicted", minimum_length=1)
    meas = _finite_vector(measured, "measured", minimum_length=1)
    if pred.size != meas.size:
        raise ValueError("predicted and measured must have the same length")
    denominator = float(np.sqrt(np.mean(meas * meas)))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("relative RMSE is undefined for an all-zero reference")
    return float(np.sqrt(np.mean((pred - meas) ** 2)) / denominator * 100.0)


def relative_energy_error_percent(
    B,
    predicted_H,
    measured_H,
    *,
    reference_loss_density: float | None = None,
) -> float:
    """Compare two H dB integrals using a declared energy denominator.

    If ``reference_loss_density`` is supplied, it is used exactly as the
    denominator (matching the challenge metric).  Otherwise the magnitude of
    the measured interval integral is used and the caller should describe the
    result as an interval-energy error, not a calorimetric loss error.
    """

    pred_energy = integrate_h_db(B, predicted_H)
    measured_energy = integrate_h_db(B, measured_H)
    denominator = (
        abs(float(reference_loss_density))
        if reference_loss_density is not None
        else abs(measured_energy)
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).eps:
        raise ValueError("energy-error denominator must be finite and non-zero")
    return abs(pred_energy - measured_energy) / denominator * 100.0
