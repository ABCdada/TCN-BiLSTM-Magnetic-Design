"""Public API for TCN--BiLSTM transient magnetic inference."""

from .config import (
    COLD_START_MATERIALS,
    MATERIALS,
    MODEL_REGISTRY,
    SAMPLING_RATE_HZ,
    WINDOW_LENGTH,
    MaterialSpec,
    ModelConfig,
)
from .inference import ModelManager, PredictionResult, Predictor
from .converter import (
    SUPPORTED_TOPOLOGIES,
    ConverterWindow,
    simulate_converter_window,
)
from .model import MagnetSeq2Seq, build_model, count_parameters, count_params
from .physics import (
    closed_cycle_loss_density,
    closed_path_energy_density,
    current_to_field_strength,
    field_strength_to_magnetizing_current,
    integrate_h_db,
    relative_energy_error_percent,
    relative_rmse_percent,
    voltage_to_flux_density,
)
from .service import InferenceService

__all__ = [
    "COLD_START_MATERIALS",
    "MATERIALS",
    "MODEL_REGISTRY",
    "SAMPLING_RATE_HZ",
    "WINDOW_LENGTH",
    "MaterialSpec",
    "ModelConfig",
    "ModelManager",
    "PredictionResult",
    "Predictor",
    "SUPPORTED_TOPOLOGIES",
    "ConverterWindow",
    "simulate_converter_window",
    "MagnetSeq2Seq",
    "build_model",
    "count_parameters",
    "count_params",
    "integrate_h_db",
    "closed_cycle_loss_density",
    "closed_path_energy_density",
    "voltage_to_flux_density",
    "current_to_field_strength",
    "field_strength_to_magnetizing_current",
    "relative_rmse_percent",
    "relative_energy_error_percent",
    "InferenceService",
]
