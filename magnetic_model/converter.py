"""Deterministic one-window SPWM converter excitation.

This module intentionally implements only the electrical front end needed to
exercise the released magnetic model: an ideal voltage-source inverter, a
sinusoidal grid, and a nominal series ``R-L`` filter.  It is not a switching-
device, controller, thermal, or grid-stability model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import SAMPLING_RATE_HZ, WINDOW_LENGTH


SUPPORTED_TOPOLOGIES = (
    "single_phase_full_bridge",
    "three_phase_full_bridge",
    "three_phase_t_type",
)


def _finite_number(
    value: float,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    """Validate direct Python callers as strictly as the HTTP service."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


@dataclass(frozen=True)
class ConverterWindow:
    """Arrays and metadata produced by one deterministic converter window."""

    topology: str
    sampling_rate_hz: float
    modulation_index: float
    grid_voltage_convention: str
    time_s: np.ndarray
    carrier: np.ndarray
    modulation_reference: np.ndarray
    inverter_voltage_v: np.ndarray
    grid_voltage_v: np.ndarray
    induced_voltage_v: np.ndarray
    nominal_current_a: np.ndarray

    def to_dict(self) -> dict:
        return {
            "topology": self.topology,
            "sampling_rate_hz": self.sampling_rate_hz,
            "sample_count": int(self.time_s.size),
            "window_duration_s": float(self.time_s.size / self.sampling_rate_hz),
            "modulation_index": self.modulation_index,
            "grid_voltage_convention": self.grid_voltage_convention,
            "time_s": self.time_s.tolist(),
            "carrier": self.carrier.tolist(),
            "modulation_reference": self.modulation_reference.tolist(),
            "inverter_voltage_V": self.inverter_voltage_v.tolist(),
            "grid_voltage_V": self.grid_voltage_v.tolist(),
            "induced_voltage_V": self.induced_voltage_v.tolist(),
            "nominal_current_A": self.nominal_current_a.tolist(),
        }


def _triangle_carrier(time_s: np.ndarray, switching_frequency_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return synchronized triangular carriers in ``[0, 1]`` and ``[-1, 1]``."""

    phase = np.mod(time_s * switching_frequency_hz, 1.0)
    carrier_01 = 1.0 - np.abs(2.0 * phase - 1.0)
    return carrier_01, 2.0 * carrier_01 - 1.0


def _three_phase_pole_voltages(
    references: np.ndarray,
    carrier_01: np.ndarray,
    carrier_bipolar: np.ndarray,
    dc_voltage: float,
    *,
    three_level: bool,
) -> np.ndarray:
    """Generate three pole voltages using two-level or PD three-level SPWM."""

    if three_level:
        upper = carrier_01[:, None]
        lower = upper - 1.0
        half_bus = 0.5 * dc_voltage
        return np.where(
            references >= upper,
            half_bus,
            np.where(references <= lower, -half_bus, 0.0),
        )
    return np.where(
        references >= carrier_bipolar[:, None],
        0.5 * dc_voltage,
        -0.5 * dc_voltage,
    )


def simulate_converter_window(
    *,
    topology: str,
    dc_voltage: float,
    grid_voltage_rms: float,
    grid_frequency_hz: float,
    switching_frequency_hz: float,
    filter_inductance_h: float,
    series_resistance_ohm: float,
    initial_current_a: float = 0.0,
    grid_phase_deg: float = 0.0,
    sampling_rate_hz: float = SAMPLING_RATE_HZ,
) -> ConverterWindow:
    """Simulate exactly one 1,000-sample converter/filter window.

    ``grid_voltage_rms`` is a single-phase RMS value for the single-phase
    topology and a line-to-line RMS value for either three-phase topology.
    The returned three-phase inverter voltage is phase A relative to the
    floating converter neutral, rather than an unreferenced pole voltage.

    The nominal filter state uses forward Euler with
    ``L di/dt = v_inv - v_grid - R i``.  Consequently, the returned induced
    voltage is exactly ``v_inv - v_grid - R*i`` at every sample and can be
    integrated by the magnetic inference pipeline.
    """

    if not isinstance(topology, str) or topology not in SUPPORTED_TOPOLOGIES:
        allowed = ", ".join(SUPPORTED_TOPOLOGIES)
        raise ValueError(f"topology must be one of: {allowed}")

    dc_voltage = _finite_number(dc_voltage, "dc_voltage", positive=True)
    grid_voltage_rms = _finite_number(
        grid_voltage_rms, "grid_voltage_rms", positive=True
    )
    grid_frequency_hz = _finite_number(
        grid_frequency_hz, "grid_frequency_hz", positive=True
    )
    switching_frequency_hz = _finite_number(
        switching_frequency_hz, "switching_frequency_hz", positive=True
    )
    filter_inductance_h = _finite_number(
        filter_inductance_h, "filter_inductance_h", positive=True
    )
    series_resistance_ohm = _finite_number(
        series_resistance_ohm, "series_resistance_ohm", nonnegative=True
    )
    initial_current_a = _finite_number(initial_current_a, "initial_current_A")
    grid_phase_deg = _finite_number(grid_phase_deg, "grid_phase_deg")
    sampling_rate_hz = _finite_number(
        sampling_rate_hz, "sampling_rate_hz", positive=True
    )
    if not math.isclose(
        sampling_rate_hz, SAMPLING_RATE_HZ, rel_tol=0.0, abs_tol=1.0
    ):
        raise ValueError("converter windows require a sampling_rate_hz of 16000000")
    nyquist_hz = 0.5 * sampling_rate_hz
    if grid_frequency_hz >= nyquist_hz:
        raise ValueError("grid_frequency_hz must be below the Nyquist frequency")
    if switching_frequency_hz >= nyquist_hz:
        raise ValueError("switching_frequency_hz must be below the Nyquist frequency")

    time_s = np.arange(WINDOW_LENGTH, dtype=np.float64) / sampling_rate_hz
    carrier_01, carrier_bipolar = _triangle_carrier(
        time_s, switching_frequency_hz
    )
    theta = (
        2.0 * np.pi * grid_frequency_hz * time_s
        + np.deg2rad(grid_phase_deg)
    )

    if topology == "single_phase_full_bridge":
        grid_peak = math.sqrt(2.0) * grid_voltage_rms
        modulation_index = grid_peak / dc_voltage
        convention = "single_phase_rms"
        if modulation_index > 1.0 + 1e-12:
            raise ValueError(
                "dc_voltage is too low for linear single-phase SPWM at the "
                "requested grid_voltage_rms"
            )
        reference_a = modulation_index * np.sin(theta)
        inverter_voltage = np.where(
            reference_a >= carrier_bipolar, dc_voltage, -dc_voltage
        )
    else:
        grid_peak = math.sqrt(2.0 / 3.0) * grid_voltage_rms
        modulation_index = 2.0 * grid_peak / dc_voltage
        convention = "three_phase_line_to_line_rms"
        if modulation_index > 1.0 + 1e-12:
            raise ValueError(
                "dc_voltage is too low for linear three-phase SPWM at the "
                "requested line-to-line grid_voltage_rms"
            )
        phase_offsets = np.array(
            [0.0, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0],
            dtype=np.float64,
        )
        references = modulation_index * np.sin(theta[:, None] + phase_offsets)
        pole_voltages = _three_phase_pole_voltages(
            references,
            carrier_01,
            carrier_bipolar,
            dc_voltage,
            three_level=topology == "three_phase_t_type",
        )
        inverter_voltage = pole_voltages[:, 0] - np.mean(
            pole_voltages, axis=1
        )
        reference_a = references[:, 0]

    grid_voltage = grid_peak * np.sin(theta)
    nominal_current = np.empty(WINDOW_LENGTH, dtype=np.float64)
    induced_voltage = np.empty(WINDOW_LENGTH, dtype=np.float64)
    current_state = initial_current_a
    dt = 1.0 / sampling_rate_hz
    for index in range(WINDOW_LENGTH):
        nominal_current[index] = current_state
        induced = (
            inverter_voltage[index]
            - grid_voltage[index]
            - series_resistance_ohm * current_state
        )
        induced_voltage[index] = induced
        current_state += dt * induced / filter_inductance_h
        if not math.isfinite(current_state):
            raise ValueError(
                "the nominal Euler state became non-finite; use physical R-L parameters"
            )

    return ConverterWindow(
        topology=topology,
        sampling_rate_hz=sampling_rate_hz,
        modulation_index=float(modulation_index),
        grid_voltage_convention=convention,
        time_s=time_s,
        carrier=carrier_bipolar,
        modulation_reference=reference_a,
        inverter_voltage_v=np.asarray(inverter_voltage, dtype=np.float64),
        grid_voltage_v=grid_voltage,
        induced_voltage_v=induced_voltage,
        nominal_current_a=nominal_current,
    )
