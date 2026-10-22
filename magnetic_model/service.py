"""Application service shared by Flask routes and the command-line client."""

from __future__ import annotations

import numpy as np

from .config import (
    LEGACY_MAX_RAW_SAMPLES,
    LEGACY_MAX_RECURSIVE_WINDOWS,
    LEGACY_MAX_RESAMPLED_SAMPLES,
    MAX_SEQUENCE_LENGTH,
    MODE_HISTORY,
    WINDOW_LENGTH,
    resolve_mode,
)
from .converter import simulate_converter_window
from .inference import ModelManager, Predictor
from .physics import (
    closed_path_energy_density,
    current_to_field_strength,
    field_strength_to_magnetizing_current,
    integrate_h_db,
    relative_energy_error_percent,
    relative_rmse_percent,
    voltage_to_flux_density,
)
from .validation import (
    RequestValidationError,
    extract_known_h,
    finite_number,
    finite_vector,
    require_object,
    sampling_rate,
    temperature_value,
)


def _closed_loss_or_none(B, H):
    try:
        return closed_path_energy_density(B, H)
    except ValueError:
        return None


class InferenceService:
    """Validate untrusted payloads before invoking numerical inference."""

    def __init__(self, manager: ModelManager | None = None):
        self.manager = manager or ModelManager()
        self.predictor = Predictor(self.manager)

    def _material_and_mode(self, payload: dict):
        material = payload.get("material")
        if not isinstance(material, str):
            choices = ", ".join(self.manager.registry)
            raise RequestValidationError(f"material must be one of {choices}")
        try:
            spec = self.manager.get_material(material)
            requested_mode = payload.get("mode")
            requested_history = payload.get("history_length")
            if requested_history is not None:
                if isinstance(requested_history, bool) or not isinstance(
                    requested_history, int
                ):
                    raise ValueError("history_length must be an integer")
                modes = [
                    candidate
                    for candidate in spec.supported_modes
                    if MODE_HISTORY[candidate] == requested_history
                ]
                if not modes:
                    allowed = ", ".join(
                        str(MODE_HISTORY[candidate])
                        for candidate in spec.supported_modes
                    )
                    raise ValueError(
                        f"history_length {requested_history} is not supported for "
                        f"Material {spec.material_id}; choose {allowed}"
                    )
                history_mode = modes[0]
                if requested_mode is not None and requested_mode != history_mode:
                    raise ValueError("mode and history_length describe different intervals")
                requested_mode = history_mode
            mode, history = resolve_mode(spec, requested_mode)
        except (KeyError, ValueError) as exc:
            raise RequestValidationError(str(exc)) from exc
        return spec, mode, history

    def list_models(self) -> dict:
        models = self.manager.public_models()
        if self.manager.expected_profile == "cold_start_1_999":
            # Browser-facing metadata uses the ferrite grade as its public
            # name. Internal A/B/D/E identifiers remain stable API keys but
            # are intentionally not presented as user-visible labels.
            for model in models:
                model["label"] = model["grade"]
        return {"models": models}

    def _predict_original_design_sequence(
        self,
        spec,
        b: np.ndarray,
        temperature: np.ndarray,
        known: np.ndarray,
        mode: str,
    ) -> tuple[np.ndarray, int]:
        """Reproduce the upstream Design Center's raw-grid window stitching.

        The upstream website sends the decimated points shown in the browser
        directly to the model.  It uses non-overlapping 1,000-point windows and
        writes the preceding final H into the first position of the next window.
        Its short final window is left-padded with preceding B/T samples and the
        first ``actual_length`` outputs are retained.  Although that tail rule is
        not the scientific sequence protocol, preserving it here makes the
        isolated Design Center compatibility path numerically comparable.
        """

        if not spec.known_h_placeholder_zero or mode != "1-999" or len(known) != 1:
            raise RequestValidationError(
                "original Design Center sampling is available only for a native "
                "1-999 cold-start model"
            )

        sample_count = len(b)
        predicted_parts: list[np.ndarray] = []
        last_h = float(known[0])

        if sample_count <= WINDOW_LENGTH:
            pad = WINDOW_LENGTH - sample_count
            b_window = np.pad(b, (0, pad), mode="edge")
            t_window = np.pad(temperature, (0, pad), mode="edge")
            result = self.predictor.predict_window(
                spec.material_id,
                b_window,
                t_window,
                [last_h],
                mode=mode,
            )
            return np.asarray(result.H[:sample_count], dtype=np.float64), 1

        for start in range(0, sample_count, WINDOW_LENGTH):
            stop = min(start + WINDOW_LENGTH, sample_count)
            actual_length = stop - start
            b_window = b[start:stop]
            t_window = temperature[start:stop]

            if actual_length < WINDOW_LENGTH:
                padding_needed = WINDOW_LENGTH - actual_length
                if start >= padding_needed:
                    b_prefix = b[start - padding_needed : start]
                    t_prefix = temperature[start - padding_needed : start]
                else:
                    b_prefix = b[:start] if start > 0 else b[:1]
                    t_prefix = temperature[:start] if start > 0 else temperature[:1]
                    if len(b_prefix) < padding_needed:
                        b_prefix = np.pad(
                            b_prefix,
                            (padding_needed - len(b_prefix), 0),
                            mode="constant",
                            constant_values=float(b[0]),
                        )
                        t_prefix = np.pad(
                            t_prefix,
                            (padding_needed - len(t_prefix), 0),
                            mode="constant",
                            constant_values=float(temperature[0]),
                        )
                b_window = np.concatenate((b_prefix[-padding_needed:], b_window))
                t_window = np.concatenate((t_prefix[-padding_needed:], t_window))

            result = self.predictor.predict_window(
                spec.material_id,
                b_window,
                t_window,
                [last_h],
                mode=mode,
            )
            window_h = np.asarray(result.H, dtype=np.float64)
            if actual_length < WINDOW_LENGTH:
                window_h = window_h[:actual_length]
            predicted_parts.append(window_h)
            last_h = float(window_h[-1])

        prediction = np.concatenate(predicted_parts)[:sample_count]
        return prediction, len(predicted_parts)

    def predict_legacy_payload(self, raw_payload) -> dict:
        """Serve the upstream web UI while using the released TCN--BiLSTM models.

        The reference UI submits only ``B`` and temperature.  The released
        models additionally require a known-H prefix.  The scientific service
        keeps its released per-material history contracts; a service backed by
        ``COLD_START_MATERIALS`` instead uses the native one-known/999-predicted
        contract for every registered browser material.  For any native cold-start
        model, a long trace starts
        from H=0 and each later 1,000-point window carries the preceding final H
        as its first state.  The bounded scientific ``predict-sequence``
        endpoint remains unchanged.
        """

        payload = require_object(raw_payload)
        registered_ids = tuple(self.manager.registry)
        legacy_id_map = {
            numeric_id: model_id
            for numeric_id, model_id in enumerate(registered_ids, start=1)
        }
        numeric_range = f"1 to {len(registered_ids)}"
        material_id = payload.get("materialId")
        if material_id is not None:
            if isinstance(material_id, bool) or not isinstance(material_id, int):
                raise RequestValidationError(
                    f"materialId must be an integer from {numeric_range}"
                )
            model_id = legacy_id_map.get(material_id)
            if model_id is None:
                raise RequestValidationError(
                    f"materialId must be an integer from {numeric_range}"
                )
        else:
            model_id = payload.get("modelId")
        if not isinstance(model_id, str):
            choices = ", ".join(registered_ids)
            raise RequestValidationError(f"modelId must be one of {choices}")
        try:
            spec = self.manager.get_material(model_id)
            mode, history_length = resolve_mode(spec, None)
        except (KeyError, ValueError) as exc:
            raise RequestValidationError(str(exc)) from exc

        if not isinstance(payload.get("sample"), dict):
            raise RequestValidationError("sample must be a JSON object")
        sample = payload["sample"]
        original_b = np.asarray(finite_vector(
            sample.get("B"),
            "sample.B",
            min_length=2,
            max_length=LEGACY_MAX_RAW_SAMPLES,
        ), dtype=np.float64)
        raw_temperature = sample.get("T")
        if isinstance(raw_temperature, list):
            original_temperature = finite_vector(
                raw_temperature,
                "sample.T",
                exact_length=len(original_b),
                max_length=LEGACY_MAX_RAW_SAMPLES,
            )
        else:
            original_temperature = finite_number(raw_temperature, "sample.T")
        if np.isscalar(original_temperature):
            original_temperature = np.full(
                len(original_b), float(original_temperature), dtype=np.float64
            )
        else:
            original_temperature = np.asarray(original_temperature, dtype=np.float64)

        sampling_mode = sample.get("sampling_mode", "model_16mhz")
        if sampling_mode not in {"model_16mhz", "original_design_raw"}:
            raise RequestValidationError(
                "sample.sampling_mode must be model_16mhz or original_design_raw"
            )
        raw_duration = sample.get("real_time")
        duration_semantics = sample.get("duration_semantics", "sample_count")
        if duration_semantics not in {"sample_count", "endpoint_span"}:
            raise RequestValidationError(
                "sample.duration_semantics must be sample_count or endpoint_span"
            )
        if raw_duration is None:
            duration = len(original_b) / 16_000_000.0
            model_b = original_b
            model_temperature = original_temperature
        elif sampling_mode == "original_design_raw":
            duration = finite_number(raw_duration, "sample.real_time", positive=True)
            model_b = original_b
            model_temperature = original_temperature
        else:
            duration = finite_number(raw_duration, "sample.real_time", positive=True)
            resampled_length = int(round(duration * 16_000_000.0))
            if duration_semantics == "endpoint_span":
                # N samples contain N-1 sampling intervals.  The Design Center
                # submits elapsed time from its first selected point to its
                # last, so the inclusive 16 MHz grid has one additional point.
                resampled_length += 1
            if resampled_length < 2:
                raise RequestValidationError(
                    "sample.real_time is too short for two samples at 16 MHz"
                )
            if resampled_length > LEGACY_MAX_RESAMPLED_SAMPLES:
                raise RequestValidationError(
                    "sample.real_time exceeds the 16 MHz Design Center limit"
                )
            if resampled_length == len(original_b):
                model_b = original_b
                model_temperature = original_temperature
            else:
                source_axis = np.linspace(0.0, duration, len(original_b))
                model_axis = np.linspace(0.0, duration, resampled_length)
                model_b = np.interp(model_axis, source_axis, original_b)
                model_temperature = np.interp(
                    model_axis, source_axis, original_temperature
                )

        # Preserve the reference UI's single visible H=0 initial state without
        # exposing an artificial 100-sample flat segment.  The extra constant-B
        # samples exist only before the user's first sample and are removed from
        # the response.  This is a compatibility initialization, not a substitute
        # for the measured history used by the manuscript evaluation protocol.
        prefix_padding = history_length - 1
        if prefix_padding:
            b = np.pad(model_b, (prefix_padding, 0), mode="edge")
            temperature = np.pad(
                model_temperature, (prefix_padding, 0), mode="edge"
            )
        else:
            b = model_b
            temperature = model_temperature
        known = np.zeros(history_length, dtype=np.float64)

        if not spec.known_h_placeholder_zero and len(b) > MAX_SEQUENCE_LENGTH:
            raise RequestValidationError(
                "long Design Center reconstruction requires a native cold-start model; "
                f"history-conditioned models remain limited to {MAX_SEQUENCE_LENGTH} "
                "model samples"
            )

        if sampling_mode == "original_design_raw":
            extended_prediction, stitch_window_count = (
                self._predict_original_design_sequence(
                    spec,
                    b,
                    temperature,
                    known,
                    mode,
                )
            )
            stitch_strategy = "original_design_nonoverlap_carry_last_h"
            stitch_hop_length = WINDOW_LENGTH
        elif len(b) < WINDOW_LENGTH:
            pad = WINDOW_LENGTH - len(b)
            padded_b = np.pad(b, (0, pad), mode="edge")
            padded_temperature = np.pad(temperature, (0, pad), mode="edge")
            extended_prediction = self.predictor.predict_window(
                spec.material_id,
                padded_b,
                padded_temperature,
                known,
                mode=mode,
            ).H[: len(b)]
            stitch_strategy = "single_window_cold_start"
            stitch_window_count = 1
            stitch_hop_length = None
        elif len(b) == WINDOW_LENGTH:
            extended_prediction = self.predictor.predict_window(
                spec.material_id,
                b,
                temperature,
                known,
                mode=mode,
            ).H
            stitch_strategy = "single_window_cold_start"
            stitch_window_count = 1
            stitch_hop_length = None
        elif spec.known_h_placeholder_zero:
            result = self.predictor.predict_recursive_sequence(
                spec.material_id,
                b,
                temperature,
                known,
                mode=mode,
                max_windows=LEGACY_MAX_RECURSIVE_WINDOWS,
            )
            extended_prediction = np.asarray(result["H"], dtype=np.float64)
            stitch_strategy = result.get(
                "stitching", "carry_last_h_overlap_one"
            )
            stitch_window_count = int(result["window_count"])
            stitch_hop_length = int(result["hop_length"])
        else:
            result = self.predictor.predict_sequence(
                spec.material_id,
                b,
                temperature,
                known,
                mode=mode,
            )
            extended_prediction = np.asarray(result["H"], dtype=np.float64)
            stitch_strategy = result.get("stitching", "overlap_average")
            stitch_window_count = int(result["window_count"])
            stitch_hop_length = int(result["hop_length"])

        model_prediction = extended_prediction[
            prefix_padding : prefix_padding + len(model_b)
        ]
        if len(model_prediction) == len(original_b):
            prediction = model_prediction
        else:
            model_axis = np.linspace(0.0, duration, len(model_prediction))
            original_axis = np.linspace(0.0, duration, len(original_b))
            prediction = np.interp(original_axis, model_axis, model_prediction)

        b = original_b
        d_b = np.diff(b)
        d_h = np.diff(prediction)
        permeability_tail = np.divide(
            d_b,
            d_h,
            out=np.zeros_like(d_b),
            where=np.abs(d_h) > 1e-12,
        )
        permeability = np.concatenate((permeability_tail[:1], permeability_tail))

        # Keep the legacy response key expected by the adapted upstream UI, but
        # compute the physically relevant line integral H dB.  For an open
        # transient interval this is an *interval magnetic-energy exchange
        # rate*; it is core-loss density only when the submitted path is a
        # complete closed magnetization cycle.
        midpoint_h = 0.5 * (prediction[:-1] + prediction[1:])
        interval_energy_rate = float(
            abs(np.sum(midpoint_h * d_b, dtype=np.float64)) / duration
        )

        return {
            "H_pred": prediction.tolist(),
            "mu": permeability.tolist(),
            # ``core_loss`` is retained solely for the original web contract.
            # See ``energy_quantity`` before interpreting this value.
            "core_loss": interval_energy_rate,
            "energy_quantity": "interval_magnetic_energy_exchange_rate",
            "energy_unit": "W/m^3",
            "material": spec.material_id,
            "grade": spec.grade,
            "mode": mode,
            "history_length": history_length,
            "stitch": {
                "strategy": stitch_strategy,
                "window_length": WINDOW_LENGTH,
                "hop_length": stitch_hop_length,
                "window_count": stitch_window_count,
                "initial_H_A_per_m": float(known[0]),
                "model_sample_count": int(len(model_b)),
                "returned_sample_count": int(len(prediction)),
            },
            "sampling_mode": sampling_mode,
            "initial_state": (
                f"material_{spec.material_id}_native_cold_start_H0_zero"
                if spec.known_h_placeholder_zero
                else "zero_H_prefix_compatibility_mode"
            ),
        }

    @staticmethod
    def _validate_known_h_for_material(spec, known):
        """Keep a cold-start model's zero placeholder identical to training."""

        if spec.known_h_placeholder_zero and not np.allclose(
            np.asarray(known, dtype=np.float64), 0.0, rtol=0.0, atol=1e-12
        ):
            raise RequestValidationError(
                f"Material {spec.material_id} uses one zero H placeholder; "
                "known_H must be [0.0]"
            )
        return known

    def predict_window_payload(self, raw_payload) -> dict:
        payload = require_object(raw_payload)
        spec, mode, history = self._material_and_mode(payload)
        rate = sampling_rate(payload)
        b = finite_vector(payload.get("B"), "B", exact_length=WINDOW_LENGTH)
        temperature = temperature_value(payload.get("temperature"), WINDOW_LENGTH)
        known = self._validate_known_h_for_material(
            spec,
            extract_known_h(
                payload.get("known_H"),
                history,
                allow_default_zero=spec.known_h_placeholder_zero,
            ),
        )
        try:
            return self.predictor.predict_window(
                spec.material_id,
                b,
                temperature,
                known,
                mode=mode,
                sampling_rate_hz=rate,
            ).to_dict()
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise exc

    def predict_sequence_payload(self, raw_payload) -> dict:
        payload = require_object(raw_payload)
        spec, mode, history = self._material_and_mode(payload)
        rate = sampling_rate(payload)
        b = finite_vector(
            payload.get("B"), "B", min_length=WINDOW_LENGTH, max_length=MAX_SEQUENCE_LENGTH
        )
        temperature = temperature_value(payload.get("temperature"), len(b))
        known = self._validate_known_h_for_material(
            spec,
            extract_known_h(
                payload.get("known_H"),
                history,
                allow_default_zero=spec.known_h_placeholder_zero,
            ),
        )
        hop = payload.get("hop_length")
        if hop is not None:
            if isinstance(hop, bool) or not isinstance(hop, int):
                raise RequestValidationError("hop_length must be an integer")
        return self.predictor.predict_sequence(
            spec.material_id,
            b,
            temperature,
            known,
            mode=mode,
            hop_length=hop,
            sampling_rate_hz=rate,
        )

    def circuit_demo_payload(self, raw_payload) -> dict:
        """Run one solver window without requiring the unknown target current.

        A solver can supply the known magnetic state either as an H prefix or
        as an equally short current prefix.  A full ``current`` remains
        backward compatible and doubles as a validation reference; the more
        explicit ``reference_current`` field is preferred for validation.
        """

        payload = require_object(raw_payload)
        spec, mode, history = self._material_and_mode(payload)
        rate = sampling_rate(payload)
        if payload.get("induced_voltage") is not None and payload.get("voltage") is not None:
            raise RequestValidationError(
                "provide induced_voltage only; voltage is a deprecated alias"
            )
        voltage_field = (
            "induced_voltage" if payload.get("induced_voltage") is not None else "voltage"
        )
        voltage = finite_vector(
            payload.get(voltage_field),
            voltage_field,
            min_length=WINDOW_LENGTH,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        temperature = temperature_value(payload.get("temperature"), len(voltage))
        turns = finite_number(payload.get("turns"), "turns", positive=True)
        area = finite_number(payload.get("area_m2"), "area_m2", positive=True)
        path = finite_number(payload.get("path_length_m"), "path_length_m", positive=True)
        initial_b = finite_number(payload.get("initial_B", 0.0), "initial_B")
        reference_loss = payload.get("reference_loss_density")
        if reference_loss is not None:
            reference_loss = finite_number(
                reference_loss, "reference_loss_density", positive=True
            )
        hop = payload.get("hop_length")
        if hop is not None and (isinstance(hop, bool) or not isinstance(hop, int)):
            raise RequestValidationError("hop_length must be an integer")

        b = voltage_to_flux_density(voltage, rate, turns, area, initial_B=initial_b)

        # ``current`` may be only the known prefix.  A full legacy current is
        # also treated as the optional validation reference.
        current_raw = payload.get("current")
        current_prefix = None
        legacy_reference_current = None
        if current_raw is not None:
            if not isinstance(current_raw, list):
                raise RequestValidationError("current must be a JSON array")
            if len(current_raw) == history:
                current_prefix = np.asarray(
                    finite_vector(current_raw, "current", exact_length=history),
                    dtype=np.float64,
                )
            elif len(current_raw) == len(voltage):
                legacy_reference_current = np.asarray(
                    finite_vector(
                        current_raw,
                        "current",
                        exact_length=len(voltage),
                        max_length=MAX_SEQUENCE_LENGTH,
                    ),
                    dtype=np.float64,
                )
                current_prefix = legacy_reference_current[:history]
            else:
                raise RequestValidationError(
                    f"current must contain either the {history}-sample known prefix "
                    f"or all {len(voltage)} reference samples"
                )

        reference_current_raw = payload.get("reference_current")
        if reference_current_raw is not None:
            reference_current = np.asarray(
                finite_vector(
                    reference_current_raw,
                    "reference_current",
                    exact_length=len(voltage),
                    max_length=MAX_SEQUENCE_LENGTH,
                ),
                dtype=np.float64,
            )
        else:
            reference_current = legacy_reference_current

        known_h_raw = payload.get("known_H")
        if spec.known_h_placeholder_zero:
            if known_h_raw is not None:
                self._validate_known_h_for_material(
                    spec,
                    extract_known_h(
                        known_h_raw,
                        history,
                        allow_default_zero=True,
                    ),
                )
            known_prefix = np.zeros(history, dtype=np.float64)
            known_source = "material_C_zero_placeholder"
        elif known_h_raw is not None:
            known_prefix = np.asarray(
                extract_known_h(
                    known_h_raw,
                    history,
                    allow_default_zero=False,
                ),
                dtype=np.float64,
            )
            known_source = "known_H"
        elif current_prefix is not None:
            known_prefix = current_to_field_strength(current_prefix, turns, path)
            known_source = "current_prefix"
        else:
            raise RequestValidationError(
                "provide known_H or a current prefix; target current is not required"
            )

        reference_h = (
            current_to_field_strength(reference_current, turns, path)
            if reference_current is not None
            else None
        )
        if len(b) == WINDOW_LENGTH:
            result = self.predictor.predict_window(
                spec.material_id,
                b,
                temperature,
                known_prefix,
                mode=mode,
                sampling_rate_hz=rate,
            )
            predicted_h = result.H
            prediction_meta = {
                "mode": mode,
                "history_length": history,
                "window_length": WINDOW_LENGTH,
                "window_count": 1,
            }
        else:
            sequence = self.predictor.predict_sequence(
                spec.material_id,
                b,
                temperature,
                known_prefix,
                mode=mode,
                hop_length=hop,
                sampling_rate_hz=rate,
            )
            predicted_h = np.asarray(sequence["H"], dtype=np.float64)
            prediction_meta = {
                key: sequence[key]
                for key in ("mode", "history_length", "window_length", "window_count", "hop_length")
            }

        target = slice(history, len(b))
        predicted_magnetizing_current = field_strength_to_magnetizing_current(
            predicted_h, turns, path
        )
        known_magnetizing_current = field_strength_to_magnetizing_current(
            known_prefix, turns, path
        )
        metrics = {
            "validation_available": reference_h is not None,
            "validation_reason": (
                None
                if reference_h is not None
                else "No full reference_current was supplied; prediction-only solver mode."
            ),
            "sequence_relative_rmse_percent": None,
            "sequence_metric_reason": (
                None if reference_h is not None else "full reference current not supplied"
            ),
            "energy_relative_error_percent": None,
            "energy_metric_reason": (
                None if reference_h is not None else "full reference current not supplied"
            ),
            "measured_h_db_integral_j_per_m3": (
                integrate_h_db(b, reference_h) if reference_h is not None else None
            ),
            "predicted_h_db_integral_j_per_m3": integrate_h_db(b, predicted_h),
            "predicted_target_h_db_integral_j_per_m3": integrate_h_db(
                b[target], predicted_h[target]
            ),
            "measured_closed_path_energy_density_j_per_m3": (
                _closed_loss_or_none(b, reference_h) if reference_h is not None else None
            ),
            "predicted_closed_path_energy_density_j_per_m3": _closed_loss_or_none(
                b, predicted_h
            ),
        }
        if reference_h is not None:
            try:
                metrics["sequence_relative_rmse_percent"] = relative_rmse_percent(
                    predicted_h[target], reference_h[target]
                )
                metrics["sequence_metric_reason"] = None
            except ValueError as exc:
                metrics["sequence_metric_reason"] = str(exc)
            try:
                metrics["energy_relative_error_percent"] = relative_energy_error_percent(
                    b[target],
                    predicted_h[target],
                    reference_h[target],
                    reference_loss_density=reference_loss,
                )
                metrics["energy_metric_reason"] = None
                metrics["energy_error_denominator"] = (
                    "reference_loss_density"
                    if reference_loss is not None
                    else "measured_interval_integral"
                )
            except ValueError as exc:
                metrics["energy_metric_reason"] = str(exc)
                metrics["energy_error_denominator"] = "undefined"
        else:
            metrics["energy_error_denominator"] = None

        reference_magnetizing_current = (
            field_strength_to_magnetizing_current(reference_h, turns, path)
            if reference_h is not None
            else None
        )

        return {
            "material": spec.material_id,
            "grade": spec.grade,
            "sampling_rate_hz": rate,
            "circuit": {
                "turns": turns,
                "area_m2": area,
                "path_length_m": path,
                "initial_B": initial_b,
                "voltage_input": voltage_field,
            },
            "prediction": {**prediction_meta, "known_H_source": known_source},
            "B": b.tolist(),
            "known_H": known_prefix.tolist(),
            "known_magnetizing_current_A": known_magnetizing_current.tolist(),
            "measured_H": reference_h.tolist() if reference_h is not None else None,
            "reference_H": reference_h.tolist() if reference_h is not None else None,
            "H": predicted_h.tolist(),
            "predicted_H": predicted_h[target].tolist(),
            "measured_magnetizing_current_A": (
                reference_magnetizing_current.tolist()
                if reference_magnetizing_current is not None
                else None
            ),
            "reference_magnetizing_current_A": (
                reference_magnetizing_current.tolist()
                if reference_magnetizing_current is not None
                else None
            ),
            "predicted_magnetizing_current_A": predicted_magnetizing_current.tolist(),
            "metrics": metrics,
            "interpretation_note": (
                "A closed-path H dB integral equals per-cycle core-loss energy "
                "only when the supplied record contains exactly one magnetic period. "
                "The complete B window must "
                "be available before inference because the BiLSTM is bidirectional "
                "and therefore noncausal within each model window."
            ),
            "voltage_note": (
                "induced_voltage must be the magnetic-core induced voltage after "
                "removing winding-resistance, leakage-inductance, and relevant "
                "parasitic voltage drops; raw terminal voltage is not interchangeable."
            ),
        }

    def converter_demo_payload(self, raw_payload) -> dict:
        """Generate a deterministic converter window and infer its H interval.

        The idealized ``R-L`` current is used only to initialize the model's
        known-history prefix.  It is deliberately not passed as a full
        reference current and therefore cannot create validation metrics.
        """

        payload = require_object(raw_payload)
        spec, mode, history = self._material_and_mode(payload)
        rate = sampling_rate(payload)
        topology = payload.get("topology")
        if not isinstance(topology, str):
            raise RequestValidationError("topology must be a string")

        temperature = temperature_value(payload.get("temperature"), WINDOW_LENGTH)
        turns = finite_number(payload.get("turns"), "turns", positive=True)
        area = finite_number(payload.get("area_m2"), "area_m2", positive=True)
        path = finite_number(
            payload.get("path_length_m"), "path_length_m", positive=True
        )
        initial_b = finite_number(payload.get("initial_B", 0.0), "initial_B")

        dc_voltage = finite_number(
            payload.get("dc_voltage"), "dc_voltage", positive=True
        )
        grid_voltage_rms = finite_number(
            payload.get("grid_voltage_rms"), "grid_voltage_rms", positive=True
        )
        grid_frequency_hz = finite_number(
            payload.get("grid_frequency_hz"), "grid_frequency_hz", positive=True
        )
        switching_frequency_hz = finite_number(
            payload.get("switching_frequency_hz"),
            "switching_frequency_hz",
            positive=True,
        )
        filter_inductance_h = finite_number(
            payload.get("filter_inductance_h"),
            "filter_inductance_h",
            positive=True,
        )
        series_resistance_ohm = finite_number(
            payload.get("series_resistance_ohm"),
            "series_resistance_ohm",
        )
        if series_resistance_ohm < 0:
            raise RequestValidationError("series_resistance_ohm must be nonnegative")
        initial_current_a = finite_number(
            payload.get("initial_current_A", 0.0), "initial_current_A"
        )
        grid_phase_deg = finite_number(
            payload.get("grid_phase_deg", 0.0), "grid_phase_deg"
        )

        try:
            simulation = simulate_converter_window(
                topology=topology,
                dc_voltage=dc_voltage,
                grid_voltage_rms=grid_voltage_rms,
                grid_frequency_hz=grid_frequency_hz,
                switching_frequency_hz=switching_frequency_hz,
                filter_inductance_h=filter_inductance_h,
                series_resistance_ohm=series_resistance_ohm,
                initial_current_a=initial_current_a,
                grid_phase_deg=grid_phase_deg,
                sampling_rate_hz=rate,
            )
        except ValueError as exc:
            raise RequestValidationError(str(exc)) from exc

        solver_payload = {
            "material": spec.material_id,
            "mode": mode,
            "sampling_rate_hz": rate,
            "induced_voltage": simulation.induced_voltage_v.tolist(),
            "current": simulation.nominal_current_a[:history].tolist(),
            "temperature": temperature,
            "turns": turns,
            "area_m2": area,
            "path_length_m": path,
            "initial_B": initial_b,
        }
        result = self.circuit_demo_payload(solver_payload)
        result["simulation"] = simulation.to_dict()
        result["scope_note"] = (
            "This deterministic SPWM and nominal R-L calculation is a "
            "window-level integration example, not a closed-loop converter "
            "or switching-device validation. nominal_current_A is used only "
            "to supply the known history; it is not measured data and is not "
            "used as a full validation reference. H and "
            "predicted_magnetizing_current_A are outputs of the magnetic model."
        )
        return result
