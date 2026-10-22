"""Safe checkpoint loading and window/long-sequence inference."""

from __future__ import annotations

import os
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .config import (
    MATERIALS,
    MAX_SEQUENCE_WINDOWS,
    SAMPLING_RATE_HZ,
    WINDOW_LENGTH,
    MaterialSpec,
    get_material,
    resolve_mode,
)
from .model import MagnetSeq2Seq, build_model, count_parameters
from .physics import closed_path_energy_density, integrate_h_db


def normalize_cold_start_state_dict_keys(state: Mapping) -> dict:
    """Translate the training-only TCN aliases used by new cold-start weights.

    The released scientific checkpoints and the existing Material C checkpoint
    use ``pw``/``proj``.  Newly trained cold-start checkpoints use the equivalent
    ``pointwise``/``residual`` names.  Only exact TCN parameter tokens are
    translated, collisions are rejected, and callers still load strictly.
    """

    aliases = {"pointwise": "pw", "residual": "proj"}
    normalized = {}
    for raw_key, value in state.items():
        key = str(raw_key)
        parts = key.split(".")
        if (
            len(parts) == 4
            and parts[0] == "tcn"
            and parts[1].isdigit()
            and parts[2] in aliases
            and parts[3] in {"weight", "bias"}
        ):
            parts[2] = aliases[parts[2]]
            key = ".".join(parts)
        if key in normalized:
            raise ValueError(f"cold-start checkpoint key alias collision for {key!r}")
        normalized[key] = value
    return normalized


def _closed_loss_or_none(B, H) -> float | None:
    """Return loop loss only for an explicitly closed path."""

    try:
        return closed_path_energy_density(B, H)
    except ValueError:
        return None


@dataclass(frozen=True)
class NormalizationStats:
    """Mean and standard deviation pairs embedded in a checkpoint."""

    B: tuple[float, float]
    dB: tuple[float, float]
    H: tuple[float, float]
    T: tuple[float, float]
    ddB: tuple[float, float] | None = None

    @classmethod
    def from_mapping(cls, value: Mapping) -> "NormalizationStats":
        required = ("B", "dB", "H", "T")
        parsed = {}
        for key in required:
            if key in value:
                raw_pair = value[key]
            elif f"{key}_mean" in value and f"{key}_std" in value:
                # Accept an explicit flat representation used by some exported
                # checkpoints while retaining the original pair representation.
                raw_pair = (value[f"{key}_mean"], value[f"{key}_std"])
            else:
                raise ValueError(f"checkpoint normalization is missing {key!r}")
            pair = np.asarray(raw_pair, dtype=np.float64).reshape(-1)
            if pair.size != 2 or not np.isfinite(pair).all() or pair[1] <= 0:
                raise ValueError(f"invalid normalization pair for {key!r}")
            parsed[key] = (float(pair[0]), float(pair[1]))
        if "ddB" in value:
            raw_pair = value["ddB"]
        elif "ddB_mean" in value and "ddB_std" in value:
            raw_pair = (value["ddB_mean"], value["ddB_std"])
        else:
            raw_pair = None
        if raw_pair is not None:
            pair = np.asarray(raw_pair, dtype=np.float64).reshape(-1)
            if pair.size != 2 or not np.isfinite(pair).all() or pair[1] <= 0:
                raise ValueError("invalid normalization pair for 'ddB'")
            parsed["ddB"] = (float(pair[0]), float(pair[1]))
        return cls(**parsed)

    def normalize(self, values, key: str) -> np.ndarray:
        pair = getattr(self, key)
        if pair is None:
            raise ValueError(f"checkpoint normalization is missing {key!r}")
        mean, scale = pair
        return (np.asarray(values, dtype=np.float64) - mean) / scale

    def denormalize(self, values, key: str) -> np.ndarray:
        mean, scale = getattr(self, key)
        return np.asarray(values, dtype=np.float64) * scale + mean


@dataclass
class LoadedModel:
    """One lazily loaded model, its normalization, and a per-model lock."""

    spec: MaterialSpec
    model: MagnetSeq2Seq
    norm: NormalizationStats
    checkpoint_path: Path
    parameter_count: int
    lock: threading.RLock


@dataclass(frozen=True)
class PredictionResult:
    """Result returned by a 1,000-sample prediction."""

    material: str
    grade: str
    mode: str
    history_length: int
    sampling_rate_hz: float
    H: np.ndarray
    predicted_H: np.ndarray
    known_H: np.ndarray
    h_db_integral_j_per_m3: float
    closed_path_energy_density_j_per_m3: float | None

    def to_dict(self) -> dict:
        return {
            "material": self.material,
            "grade": self.grade,
            "mode": self.mode,
            "history_length": self.history_length,
            "prediction_length": int(self.H.size - self.history_length),
            "sampling_rate_hz": self.sampling_rate_hz,
            "H": self.H.tolist(),
            "known_H": self.known_H.tolist(),
            "predicted_H": self.predicted_H.tolist(),
            "energy": {
                "h_db_integral_j_per_m3": self.h_db_integral_j_per_m3,
                "closed_path_energy_density_j_per_m3": (
                    self.closed_path_energy_density_j_per_m3
                ),
                "note": (
                    "The signed interval integral is transient energy exchange; "
                    "the closed-path value covers the complete supplied path and "
                    "equals per-cycle loss only when that path is one period."
                ),
            },
        }


class ModelManager:
    """Resolve trusted checkpoint names and lazily cache material models."""

    def __init__(
        self,
        model_dir: str | os.PathLike | None = None,
        device: str | None = None,
        *,
        registry: Mapping[str, MaterialSpec] = MATERIALS,
        manifest_filename: str = "manifest.json",
        checkpoint_env_prefix: str = "MAGNET_CHECKPOINT_",
        model_dir_env: str = "MAGNET_MODEL_DIR",
        expected_profile: str | None = None,
    ):
        project_dir = Path(__file__).resolve().parents[1]
        self.registry = registry
        self.checkpoint_env_prefix = checkpoint_env_prefix
        self.expected_profile = expected_profile
        self.model_dir = Path(
            model_dir or os.environ.get(model_dir_env, project_dir / "weights")
        ).expanduser().resolve()
        requested = device or os.environ.get("MAGNET_DEVICE", "auto")
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.device = torch.device(requested)
        self._models: dict[str, LoadedModel] = {}
        self._manager_lock = threading.RLock()
        self.manifest_path = self.model_dir / manifest_filename
        self._manifest = self._read_manifest(
            self.manifest_path, self.registry, self.expected_profile
        )
        self._integrity_cache: dict[str, tuple[int, int, str]] = {}

    @staticmethod
    def _read_manifest(
        path: Path,
        registry: Mapping[str, MaterialSpec],
        expected_profile: str | None,
    ) -> Mapping:
        if not path.is_file():
            raise FileNotFoundError(
                f"trusted checkpoint manifest was not found at {path}; keep the "
                "released weights/manifest.json beside the checkpoints"
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint manifest {path.name}: {exc}") from exc
        if payload.get("format_version") != 1 or payload.get("algorithm") != "sha256":
            raise ValueError("unsupported checkpoint-manifest format or algorithm")
        if expected_profile is not None and payload.get("profile") != expected_profile:
            raise ValueError(
                f"checkpoint manifest profile must be {expected_profile!r}"
            )
        records = payload.get("models")
        if not isinstance(records, Mapping) or set(records) != set(registry):
            expected = ", ".join(registry)
            raise ValueError(
                f"checkpoint manifest must contain exactly these models: {expected}"
            )
        return payload

    def get_material(self, material: str) -> MaterialSpec:
        """Resolve a material in this manager's isolated registry."""

        return get_material(material, self.registry)

    def _checkpoint_override_name(self, material_id: str) -> str:
        return f"{self.checkpoint_env_prefix}{material_id}"

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

    def checkpoint_path(self, material: str) -> Path:
        """Return the configured checkpoint path without accepting user paths."""

        spec = self.get_material(material)
        override = os.environ.get(self._checkpoint_override_name(spec.material_id))
        return Path(override).expanduser().resolve() if override else self.model_dir / spec.checkpoint_filename

    def is_available(self, material: str) -> bool:
        try:
            self.verify_checkpoint(material)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def verify_checkpoint(self, material: str) -> str:
        """Verify a checkpoint against the immutable release manifest."""

        spec = self.get_material(material)
        record = self._manifest["models"][spec.material_id]
        if record.get("status") == "pending":
            raise FileNotFoundError(
                f"cold-start checkpoint for Material {spec.material_id} is pending"
            )
        path = self.checkpoint_path(spec.material_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        if record.get("material") != spec.grade:
            raise ValueError(f"manifest material mismatch for {spec.material_id}")
        if int(record.get("parameter_count", -1)) != spec.parameter_count:
            raise ValueError(f"manifest parameter-count mismatch for {spec.material_id}")
        if not os.environ.get(self._checkpoint_override_name(spec.material_id)):
            if record.get("path") != spec.checkpoint_filename:
                raise ValueError(f"manifest filename mismatch for {spec.material_id}")
        stat = path.stat()
        expected_size = int(record.get("bytes", -1))
        expected_hash = str(record.get("sha256", ""))
        cache_key = str(path)
        cached = self._integrity_cache.get(cache_key)
        if cached == (stat.st_size, stat.st_mtime_ns, expected_hash):
            return expected_hash
        if stat.st_size != expected_size:
            raise ValueError(f"checkpoint byte-count mismatch for {spec.material_id}")
        actual_hash = self._sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"checkpoint SHA-256 mismatch for {spec.material_id}")
        self._integrity_cache[cache_key] = (stat.st_size, stat.st_mtime_ns, expected_hash)
        return expected_hash

    def public_models(self) -> list[dict]:
        """Return registry metadata and local availability."""

        result = []
        for material, spec in self.registry.items():
            item = spec.to_dict()
            item["available"] = self.is_available(material)
            item["integrity"] = "verified" if item["available"] else "unavailable_or_invalid"
            item["parameters"] = spec.parameter_count
            result.append(item)
        return result

    def load(self, material: str) -> LoadedModel:
        """Load a ``{'model': state_dict, 'norm': stats}`` checkpoint strictly."""

        spec = self.get_material(material)
        with self._manager_lock:
            if spec.material_id in self._models:
                return self._models[spec.material_id]
            path = self.checkpoint_path(spec.material_id)
            # Consult the manifest first so a deliberately pending cold-start
            # model cannot be mistaken for an ordinary missing release file.
            self.verify_checkpoint(spec.material_id)
            if not path.is_file():
                raise FileNotFoundError(
                    f"checkpoint for Material {spec.material_id} ({spec.grade}) was not found; "
                    f"expected {spec.checkpoint_filename} in the configured model directory "
                    f"or {self._checkpoint_override_name(spec.material_id)}"
                )
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:  # PyTorch before the weights_only keyword.
                checkpoint = torch.load(path, map_location="cpu")
            if not isinstance(checkpoint, Mapping):
                raise ValueError(f"checkpoint {path.name} must contain a mapping")
            if "model" not in checkpoint or "norm" not in checkpoint:
                raise ValueError(
                    f"checkpoint {path.name} must contain both 'model' and 'norm'"
                )
            state = checkpoint["model"]
            if not isinstance(state, Mapping):
                raise ValueError("checkpoint 'model' entry must be a state dictionary")
            if state and all(str(key).startswith("module.") for key in state):
                state = {str(key)[7:]: value for key, value in state.items()}
            if self.expected_profile == "cold_start_1_999":
                state = normalize_cold_start_state_dict_keys(state)
            model = build_model(spec.model)
            model.load_state_dict(state, strict=True)
            model.eval().to(self.device)
            loaded = LoadedModel(
                spec=spec,
                model=model,
                norm=NormalizationStats.from_mapping(checkpoint["norm"]),
                checkpoint_path=path,
                parameter_count=count_parameters(model),
                lock=threading.RLock(),
            )
            self._models[spec.material_id] = loaded
            return loaded

    @property
    def loaded_materials(self) -> list[str]:
        with self._manager_lock:
            return sorted(self._models)


class Predictor:
    """High-level API for a single model window or stitched long sequence."""

    def __init__(self, manager: ModelManager):
        self.manager = manager

    def _material(self, material: str) -> MaterialSpec:
        # The fallback keeps lightweight unit-test doubles backwards compatible;
        # production managers always resolve through their isolated registry.
        resolver = getattr(self.manager, "get_material", None)
        return resolver(material) if resolver is not None else get_material(material)

    @staticmethod
    def _temperature_profile(temperature, length: int) -> np.ndarray:
        value = np.asarray(temperature, dtype=np.float64)
        if value.ndim == 0:
            value = np.full(length, float(value), dtype=np.float64)
        if value.ndim != 1 or value.size != length:
            raise ValueError(f"temperature must be a scalar or contain {length} samples")
        if not np.isfinite(value).all():
            raise ValueError("temperature contains NaN or infinite values")
        return value

    @staticmethod
    def _finite_vector(values, name: str, *, length: int | None = None) -> np.ndarray:
        value = np.asarray(values, dtype=np.float64)
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional sequence")
        if length is not None and value.size != length:
            raise ValueError(f"{name} must contain exactly {length} samples")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinite values")
        return value

    def predict_window(
        self,
        material: str,
        B,
        temperature,
        known_H,
        *,
        mode: str | None = None,
        sampling_rate_hz: float = SAMPLING_RATE_HZ,
    ) -> PredictionResult:
        """Predict the masked tail of one exactly 1,000-sample window."""

        spec = self._material(material)
        selected_mode, history_length = resolve_mode(spec, mode)
        if not np.isclose(sampling_rate_hz, SAMPLING_RATE_HZ, rtol=0.0, atol=1.0):
            raise ValueError("sampling_rate_hz must be 16000000 for the released models")
        b = self._finite_vector(B, "B", length=WINDOW_LENGTH)
        temp = self._temperature_profile(temperature, WINDOW_LENGTH)
        known = self._finite_vector(known_H, "known_H", length=history_length)
        loaded = self.manager.load(spec.material_id)

        db = np.diff(b, prepend=b[:1])
        x = np.zeros((WINDOW_LENGTH, loaded.spec.model.in_ch), dtype=np.float32)
        feature_index = {
            name: index for index, name in enumerate(loaded.spec.input_features)
        }
        x[:, feature_index["B"]] = loaded.norm.normalize(b, "B").astype(np.float32)
        x[:, feature_index["dB"]] = loaded.norm.normalize(db, "dB").astype(np.float32)
        if "ddB" in feature_index:
            ddb = np.diff(db, prepend=db[:1])
            x[:, feature_index["ddB"]] = loaded.norm.normalize(ddb, "ddB").astype(
                np.float32
            )
        x[:, feature_index["T"]] = loaded.norm.normalize(temp, "T").astype(np.float32)
        x[:history_length, feature_index["H_known"]] = loaded.norm.normalize(
            known, "H"
        ).astype(np.float32)
        x[:history_length, feature_index["mask"]] = 1.0

        tensor = torch.from_numpy(x).unsqueeze(0).to(self.manager.device)
        with loaded.lock, torch.inference_mode():
            normalized = loaded.model(tensor)[0].detach().float().cpu().numpy()
        predicted = loaded.norm.denormalize(normalized, "H")
        full_h = predicted.astype(np.float64, copy=True)
        full_h[:history_length] = known
        return PredictionResult(
            material=spec.material_id,
            grade=spec.grade,
            mode=selected_mode,
            history_length=history_length,
            sampling_rate_hz=float(sampling_rate_hz),
            H=full_h,
            predicted_H=full_h[history_length:].copy(),
            known_H=known.copy(),
            h_db_integral_j_per_m3=integrate_h_db(b, full_h),
            closed_path_energy_density_j_per_m3=_closed_loss_or_none(b, full_h),
        )

    def predict_sequence(
        self,
        material: str,
        B,
        temperature,
        known_H,
        *,
        mode: str | None = None,
        hop_length: int | None = None,
        sampling_rate_hz: float = SAMPLING_RATE_HZ,
    ) -> dict:
        """Predict a long sequence by history-respecting window stitching.

        Each subsequent window uses only already reconstructed H values as its
        known prefix.  If ``hop_length`` is smaller than the newly predicted
        tail, overlapping estimates are averaged; it may never exceed that
        tail, which prevents gaps and information leakage.
        """

        spec = self._material(material)
        selected_mode, history_length = resolve_mode(spec, mode)
        b = self._finite_vector(B, "B")
        if b.size < WINDOW_LENGTH:
            raise ValueError(f"B must contain at least {WINDOW_LENGTH} samples")
        temp = self._temperature_profile(temperature, b.size)
        known = self._finite_vector(known_H, "known_H", length=history_length)
        tail_length = WINDOW_LENGTH - history_length
        hop = tail_length if hop_length is None else int(hop_length)
        if hop < 1 or hop > tail_length:
            raise ValueError(f"hop_length must be between 1 and {tail_length}")

        last_start = b.size - WINDOW_LENGTH
        starts = list(range(0, last_start + 1, hop))
        if starts[-1] != last_start:
            starts.append(last_start)
        if len(starts) > MAX_SEQUENCE_WINDOWS:
            raise ValueError(
                f"sequence would require {len(starts)} model windows; the service "
                f"limit is {MAX_SEQUENCE_WINDOWS}. Increase hop_length, reduce the "
                "sequence length, or use a controlled offline batch job"
            )

        sums = np.zeros(b.size, dtype=np.float64)
        counts = np.zeros(b.size, dtype=np.int64)
        sums[:history_length] = known
        counts[:history_length] = 1
        windows = []
        for start in starts:
            prefix_slice = slice(start, start + history_length)
            if spec.known_h_placeholder_zero:
                # Native cold-start checkpoints were trained with an
                # independent raw zero placeholder in every evaluation window.
                prefix = np.zeros(history_length, dtype=np.float64)
            else:
                if np.any(counts[prefix_slice] == 0):
                    raise RuntimeError("internal stitching gap before a model window")
                prefix = sums[prefix_slice] / counts[prefix_slice]
            result = self.predict_window(
                spec.material_id,
                b[start : start + WINDOW_LENGTH],
                temp[start : start + WINDOW_LENGTH],
                prefix,
                mode=selected_mode,
                sampling_rate_hz=sampling_rate_hz,
            )
            target_start = start + history_length
            target_stop = start + WINDOW_LENGTH
            sums[target_start:target_stop] += result.predicted_H
            counts[target_start:target_stop] += 1
            windows.append({"start": start, "stop": target_stop})
        if np.any(counts == 0):
            raise RuntimeError("internal stitching left uncovered samples")
        h = sums / counts
        return {
            "material": spec.material_id,
            "grade": spec.grade,
            "mode": selected_mode,
            "history_length": history_length,
            "hop_length": hop,
            "window_length": WINDOW_LENGTH,
            "window_count": len(windows),
            "sampling_rate_hz": float(sampling_rate_hz),
            "H": h.tolist(),
            "predicted_H": h[history_length:].tolist(),
            "energy": {
                "h_db_integral_j_per_m3": integrate_h_db(b, h),
                "closed_path_energy_density_j_per_m3": _closed_loss_or_none(b, h),
            },
            "windows": windows,
        }

    def predict_recursive_sequence(
        self,
        material: str,
        B,
        temperature,
        known_H,
        *,
        mode: str | None = None,
        sampling_rate_hz: float = SAMPLING_RATE_HZ,
        max_windows: int | None = None,
    ) -> dict:
        """Predict the Design Center sequence using recursive window carry-over.

        This separate compatibility path implements the circuit demo's native
        cold-start semantics. The first 1,000-point window starts
        from the caller-supplied ``H[0]``. Each following window overlaps the
        preceding one by one sample, carrying its final ``B`` and predicted
        ``H`` into the next window, and appends only 999 new predictions. A
        short final window is edge-padded and only its valid output is retained.

        :meth:`predict_sequence` remains unchanged and continues to use an
        independent zero placeholder in every cold-start evaluation window, as required
        by the manuscript evaluation protocol and that model's training setup.
        """

        spec = self._material(material)
        selected_mode, history_length = resolve_mode(spec, mode)
        if (
            not spec.known_h_placeholder_zero
            or selected_mode != "1-999"
            or history_length != 1
        ):
            raise ValueError(
                "recursive compatibility inference is available only for "
                "a cold-start material in 1-999 mode"
            )
        if not np.isclose(sampling_rate_hz, SAMPLING_RATE_HZ, rtol=0.0, atol=1.0):
            raise ValueError("sampling_rate_hz must be 16000000 for the released models")

        b = self._finite_vector(B, "B")
        if b.size < 2:
            raise ValueError("B must contain at least 2 samples")
        temp = self._temperature_profile(temperature, b.size)
        known = self._finite_vector(known_H, "known_H", length=1)

        hop = WINDOW_LENGTH - 1
        required_windows = int(np.ceil((b.size - 1) / hop))
        if max_windows is not None:
            if isinstance(max_windows, bool) or int(max_windows) != max_windows:
                raise ValueError("max_windows must be a positive integer or None")
            window_limit = int(max_windows)
            if window_limit < 1:
                raise ValueError("max_windows must be a positive integer or None")
            if required_windows > window_limit:
                raise ValueError(
                    f"sequence requires {required_windows} recursive model windows; "
                    f"max_windows is {window_limit}"
                )

        h = np.empty(b.size, dtype=np.float64)
        h[0] = known[0]
        windows = []
        start = 0
        while start < b.size - 1:
            valid_length = min(WINDOW_LENGTH, b.size - start)
            b_window = b[start : start + valid_length]
            temp_window = temp[start : start + valid_length]
            padded_samples = WINDOW_LENGTH - valid_length
            if padded_samples:
                b_window = np.pad(b_window, (0, padded_samples), mode="edge")
                temp_window = np.pad(temp_window, (0, padded_samples), mode="edge")

            result = self.predict_window(
                spec.material_id,
                b_window,
                temp_window,
                [h[start]],
                mode=selected_mode,
                sampling_rate_hz=sampling_rate_hz,
            )
            new_sample_count = valid_length - 1
            stop = start + valid_length
            h[start + 1 : stop] = result.predicted_H[:new_sample_count]
            windows.append(
                {
                    "start": start,
                    "stop": stop,
                    "model_stop": start + WINDOW_LENGTH,
                    "valid_length": valid_length,
                    "padded_samples": padded_samples,
                }
            )
            start += hop

        return {
            "material": spec.material_id,
            "grade": spec.grade,
            "mode": selected_mode,
            "history_length": history_length,
            "hop_length": hop,
            "window_length": WINDOW_LENGTH,
            "window_count": len(windows),
            "sampling_rate_hz": float(sampling_rate_hz),
            "stitching": "carry_last_h_overlap_one",
            "H": h.tolist(),
            "predicted_H": h[history_length:].tolist(),
            "known_H": known.tolist(),
            "energy": {
                "h_db_integral_j_per_m3": integrate_h_db(b, h),
                "closed_path_energy_density_j_per_m3": _closed_loss_or_none(b, h),
            },
            "windows": windows,
        }
