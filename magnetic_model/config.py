"""Immutable model and material metadata used by inference.

The registry mirrors the architectures recorded in the training reports for
Materials A--E.  Checkpoint paths are intentionally not embedded here: an
installation selects its checkpoint directory through ``MAGNET_MODEL_DIR`` or
passes it explicitly to :class:`magnetic_model.inference.ModelManager`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Tuple


WINDOW_LENGTH = 1_000
SAMPLING_RATE_HZ = 16_000_000.0
MAX_SEQUENCE_LENGTH = 50_000
MAX_SEQUENCE_WINDOWS = 256

# The scientific ``/api/predict-sequence`` endpoint intentionally keeps the
# conservative limits above.  The legacy Design Center, however, submits an
# entire selected waveform and expects the model to advance through it using
# consecutive 1,000-sample windows.  These separate bounds allow a complete
# Design trace without silently relaxing the public scientific API contract.
LEGACY_MAX_RAW_SAMPLES = 200_000
LEGACY_MAX_RESAMPLED_SAMPLES = 2_000_000
LEGACY_MAX_RECURSIVE_WINDOWS = 2_048


@dataclass(frozen=True)
class ModelConfig:
    """Shape-defining hyperparameters of a trained TCN--BiLSTM model."""

    in_ch: int = 5
    stem: int = 64
    tcn_ch: int = 128
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    lstm_hidden: int = 192
    lstm_layers: int = 2
    head_hidden: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation."""

        value = asdict(self)
        value["tcn_dilations"] = list(self.tcn_dilations)
        return value


@dataclass(frozen=True)
class MaterialSpec:
    """Human-readable and architecture metadata for one material model."""

    material_id: str
    grade: str
    checkpoint_filename: str
    model: ModelConfig
    parameter_count: int
    default_mode: str
    supported_modes: Tuple[str, ...]
    description: str
    known_h_placeholder_zero: bool = False
    input_features: Tuple[str, ...] = ("B", "dB", "T", "H_known", "mask")

    def to_dict(self) -> dict:
        """Return public metadata without exposing host file-system paths."""

        return {
            "id": self.material_id,
            "grade": self.grade,
            "label": f"Material {self.material_id} ({self.grade})",
            "checkpoint_filename": self.checkpoint_filename,
            "default_mode": self.default_mode,
            "supported_modes": list(self.supported_modes),
            "description": self.description,
            "parameter_count": self.parameter_count,
            "known_h_placeholder_zero": self.known_h_placeholder_zero,
            "input_features": list(self.input_features),
            "config": self.model.to_dict(),
        }


_LARGE = ModelConfig()

MATERIALS: Mapping[str, MaterialSpec] = {
    "A": MaterialSpec(
        material_id="A",
        grade="3C92",
        checkpoint_filename="A_best_model.pth",
        model=ModelConfig(
            tcn_ch=64,
            tcn_dilations=(1, 2, 4, 8, 16),
            lstm_hidden=96,
            dropout=0.25,
        ),
        parameter_count=497_793,
        default_mode="100-900",
        supported_modes=("100-900", "500-500", "900-100"),
        description="Small-data 3C92 model.",
    ),
    "B": MaterialSpec(
        material_id="B",
        grade="3C95",
        checkpoint_filename="B_best_model.pth",
        model=_LARGE,
        parameter_count=2_092_545,
        default_mode="100-900",
        supported_modes=("100-900", "500-500", "900-100"),
        description="3C95 model used for cross-frequency evaluation.",
    ),
    "C": MaterialSpec(
        material_id="C",
        grade="FEC007",
        checkpoint_filename="C_best_model.pth",
        model=_LARGE,
        parameter_count=2_092_545,
        default_mode="1-999",
        supported_modes=("1-999",),
        description="Cold-start FEC007 model with one zero placeholder for H.",
        known_h_placeholder_zero=True,
    ),
    "D": MaterialSpec(
        material_id="D",
        grade="FEC014",
        checkpoint_filename="D_best_model.pth",
        model=ModelConfig(tcn_ch=144, lstm_hidden=208),
        parameter_count=2_512_593,
        default_mode="100-900",
        supported_modes=("100-900", "500-500", "900-100"),
        description="Higher-capacity FEC014 model for complex hysteresis.",
    ),
    "E": MaterialSpec(
        material_id="E",
        grade="T37",
        checkpoint_filename="E_best_model.pth",
        model=ModelConfig(in_ch=6),
        parameter_count=2_092_865,
        default_mode="100-900",
        supported_modes=("100-900", "500-500", "900-100"),
        description="T37 model with the additional second flux increment ddB.",
        input_features=("B", "dB", "ddB", "T", "H_known", "mask"),
    ),
}

# A separate registry is used by the browser-facing cold-start API.  Keeping
# these specifications distinct from ``MATERIALS`` is deliberate: the released
# scientific checkpoints and their 100/500/900-sample history contracts remain
# immutable while the newly trained checkpoints use one H=0 placeholder and
# predict samples 1--999 for every material.
COLD_START_MATERIALS: Mapping[str, MaterialSpec] = {
    material_id: MaterialSpec(
        material_id=material_id,
        grade=scientific.grade,
        checkpoint_filename=f"{material_id}_cold_start_best_model.pth",
        model=scientific.model,
        parameter_count=scientific.parameter_count,
        default_mode="1-999",
        supported_modes=("1-999",),
        description=(
            f"Cold-start {scientific.grade} model with one zero placeholder for H."
        ),
        known_h_placeholder_zero=True,
        input_features=scientific.input_features,
    )
    for material_id, scientific in MATERIALS.items()
    if material_id in ("A", "B", "D", "E")
}

# Backwards-friendly alias used by examples and external scripts.
MODEL_REGISTRY = MATERIALS

MODE_HISTORY = {
    "1-999": 1,
    "100-900": 100,
    "500-500": 500,
    "900-100": 900,
}


def get_material(
    material: str, registry: Mapping[str, MaterialSpec] = MATERIALS
) -> MaterialSpec:
    """Return a material specification or raise a concise ``KeyError``."""

    key = str(material).strip().upper()
    if key not in registry:
        choices = ", ".join(registry)
        raise KeyError(f"unknown material {material!r}; choose {choices}")
    return registry[key]


def resolve_mode(spec: MaterialSpec, mode: str | None = None) -> tuple[str, int]:
    """Resolve and validate a named history-to-prediction mode."""

    selected = (mode or spec.default_mode).strip()
    if selected not in spec.supported_modes:
        allowed = ", ".join(spec.supported_modes)
        raise ValueError(
            f"mode {selected!r} is not supported by Material {spec.material_id}; "
            f"choose {allowed}"
        )
    return selected, MODE_HISTORY[selected]
