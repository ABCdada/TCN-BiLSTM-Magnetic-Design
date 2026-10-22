"""Flask application for the magnetic-design website and inference APIs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from magnetic_model import COLD_START_MATERIALS, InferenceService, ModelManager
from magnetic_model.validation import RequestValidationError


def create_app(config: dict | None = None) -> Flask:
    """Create an application without loading any checkpoint eagerly."""

    source_root = Path(__file__).resolve().parent
    installed_root = Path(sys.prefix) / "share" / "tcn_bilstm_magnetic_circuit"
    template_root = (
        source_root / "templates"
        if (source_root / "templates").is_dir()
        else installed_root / "templates"
    )
    static_root = (
        source_root / "static"
        if (source_root / "static").is_dir()
        else installed_root / "static"
    )
    weights_root = (
        source_root / "weights"
        if (source_root / "weights").is_dir()
        else installed_root / "weights"
    )
    application = Flask(
        __name__,
        template_folder=str(template_root),
        static_folder=str(static_root),
    )
    application.config.update(
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        JSON_SORT_KEYS=False,
    )
    if config:
        application.config.update(config)

    model_dir = application.config.get("MODEL_DIR")
    if model_dir is None and "MAGNET_MODEL_DIR" not in os.environ:
        model_dir = weights_root
    manager = ModelManager(
        model_dir=model_dir,
        device=application.config.get("MODEL_DEVICE"),
    )
    service = InferenceService(manager)
    application.extensions["magnetic_service"] = service
    cold_model_dir = application.config.get("COLD_MODEL_DIR") or application.config.get(
        "MODEL_DIR"
    )
    if cold_model_dir is None and "MAGNET_COLD_START_MODEL_DIR" not in os.environ:
        cold_model_dir = weights_root
    cold_manager = ModelManager(
        model_dir=cold_model_dir,
        device=application.config.get("MODEL_DEVICE"),
        registry=COLD_START_MATERIALS,
        manifest_filename="cold_start_manifest.json",
        checkpoint_env_prefix="MAGNET_COLD_START_CHECKPOINT_",
        model_dir_env="MAGNET_COLD_START_MODEL_DIR",
        expected_profile="cold_start_1_999",
    )
    cold_service = InferenceService(cold_manager)
    application.extensions["cold_start_magnetic_service"] = cold_service

    @application.get("/")
    def index():
        return render_template("index.html")

    @application.get("/test")
    def reference_test_page():
        return render_template("test.html")

    @application.get("/design")
    def reference_design_page():
        return render_template("design.html")

    @application.get("/document")
    def reference_document_page():
        return render_template("document.html")

    @application.get("/fine_tune")
    def reference_model_library_page():
        return render_template("fine_tune.html")

    @application.get("/health")
    def health():
        models = service.manager.public_models()
        cold_models = cold_service.manager.public_models()
        return jsonify(
            {
                "status": "ok",
                "version": "1.3.3",
                "device": str(service.manager.device),
                "loaded_models": service.manager.loaded_materials,
                "available_models": [item["id"] for item in models if item["available"]],
                "cold_start": {
                    "device": str(cold_service.manager.device),
                    "loaded_models": cold_service.manager.loaded_materials,
                    "available_models": [
                        item["id"] for item in cold_models if item["available"]
                    ],
                },
            }
        )

    @application.get("/api/models")
    def models():
        return jsonify(service.list_models())

    @application.get("/api/cold-start-models")
    def cold_start_models():
        """List the separately versioned native one-known/999-predicted family."""

        return jsonify(cold_service.list_models())

    def json_payload():
        if not request.is_json:
            raise RequestValidationError("Content-Type must be application/json")
        payload = request.get_json(silent=False)
        if payload is None:
            raise RequestValidationError("request body must contain JSON")
        return payload

    @application.post("/api/predict-window")
    def predict_window():
        return jsonify(service.predict_window_payload(json_payload()))

    @application.post("/api/predict")
    def reference_predict():
        return jsonify(service.predict_legacy_payload(json_payload()))

    @application.post("/api/predict-cold-start")
    def predict_cold_start():
        """Predict with the isolated one-known/999-predicted model family."""

        return jsonify(cold_service.predict_legacy_payload(json_payload()))

    @application.post("/api/upload")
    @application.post("/api/finetune")
    def browser_training_unavailable():
        return (
            jsonify(
                {
                    "error": (
                        "Browser-side training is not enabled in this inference release. "
                        "Use a separately maintained offline training environment."
                    )
                }
            ),
            501,
        )

    @application.post("/api/predict-sequence")
    def predict_sequence():
        return jsonify(service.predict_sequence_payload(json_payload()))

    @application.post("/api/circuit-demo")
    def circuit_demo():
        return jsonify(service.circuit_demo_payload(json_payload()))

    @application.post("/api/converter-demo")
    def converter_demo():
        return jsonify(service.converter_demo_payload(json_payload()))

    @application.errorhandler(RequestValidationError)
    def bad_request(error):
        return (
            jsonify(
                {
                    "error": {
                        "code": "invalid_request",
                        "message": str(error),
                        "details": error.details,
                    }
                }
            ),
            400,
        )

    @application.errorhandler(FileNotFoundError)
    def missing_checkpoint(error):
        return (
            jsonify(
                {
                    "error": {
                        "code": "model_unavailable",
                        "message": str(error),
                    }
                }
            ),
            503,
        )

    @application.errorhandler(ValueError)
    def invalid_value(error):
        return jsonify({"error": {"code": "invalid_value", "message": str(error)}}), 400

    @application.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return (
            jsonify(
                {
                    "error": {
                        "code": "request_too_large",
                        "message": "request body exceeds the configured size limit",
                    }
                }
            ),
            413,
        )

    @application.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith("/api/") or request.path == "/health":
            return (
                jsonify(
                    {
                        "error": {
                            "code": error.name.lower().replace(" ", "_"),
                            "message": error.description,
                        }
                    }
                ),
                error.code,
            )
        return error

    @application.errorhandler(Exception)
    def internal_error(error):
        application.logger.exception("unhandled API error")
        if request.path.startswith("/api/") or request.path == "/health":
            return (
                jsonify(
                    {
                        "error": {
                            "code": "internal_error",
                            "message": "the request could not be completed",
                        }
                    }
                ),
                500,
            )
        raise error

    return application


app = create_app()


if __name__ == "__main__":
    # Debug mode is deliberately disabled for model and host confidentiality.
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=False,
    )
