"""Small dependency-free validators for public JSON and CLI inputs."""

from __future__ import annotations

import math
from typing import Any

from .config import MAX_SEQUENCE_LENGTH, SAMPLING_RATE_HZ


class RequestValidationError(ValueError):
    """A client-supplied value failed schema or physical validation."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


def require_object(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise RequestValidationError("request body must be a JSON object")
    return payload


def finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise RequestValidationError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{name} must be a number") from exc
    if not math.isfinite(number):
        raise RequestValidationError(f"{name} must be finite")
    if positive and number <= 0:
        raise RequestValidationError(f"{name} must be positive")
    return number


def finite_vector(
    value: Any,
    name: str,
    *,
    exact_length: int | None = None,
    min_length: int | None = None,
    max_length: int = MAX_SEQUENCE_LENGTH,
) -> list[float]:
    if not isinstance(value, list):
        raise RequestValidationError(f"{name} must be a JSON array")
    size = len(value)
    if exact_length is not None and size != exact_length:
        raise RequestValidationError(f"{name} must contain exactly {exact_length} samples")
    if min_length is not None and size < min_length:
        raise RequestValidationError(f"{name} must contain at least {min_length} samples")
    if size > max_length:
        raise RequestValidationError(f"{name} exceeds the {max_length}-sample limit")
    return [finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def sampling_rate(payload: dict) -> float:
    value = finite_number(
        payload.get("sampling_rate_hz", SAMPLING_RATE_HZ),
        "sampling_rate_hz",
        positive=True,
    )
    if not math.isclose(value, SAMPLING_RATE_HZ, rel_tol=0.0, abs_tol=1.0):
        raise RequestValidationError(
            "the released models require a sampling_rate_hz of 16000000"
        )
    return value


def temperature_value(value: Any, length: int):
    if isinstance(value, list):
        return finite_vector(value, "temperature", exact_length=length)
    return finite_number(value, "temperature")


def extract_known_h(value: Any, history_length: int, *, allow_default_zero: bool) -> list[float]:
    if value is None:
        if allow_default_zero and history_length == 1:
            return [0.0]
        raise RequestValidationError("known_H is required")
    if not isinstance(value, list):
        raise RequestValidationError("known_H must be a JSON array")
    if len(value) == history_length:
        return finite_vector(value, "known_H", exact_length=history_length)
    if len(value) == 1_000:
        prefix = value[:history_length]
        return finite_vector(prefix, "known_H prefix", exact_length=history_length)
    raise RequestValidationError(
        f"known_H must contain the {history_length}-sample prefix or a 1000-sample masked array"
    )
