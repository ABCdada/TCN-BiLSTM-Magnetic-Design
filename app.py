"""Flask application exposing reproducible magnetic-model inference APIs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
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
    license_root = (
        source_root / "licenses"
        if (source_root / "licenses").is_dir()
        else installed_root / "licenses"
    )
    artifact_root = (
        source_root
        if (source_root / "results").is_dir()
        else installed_root
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
    cold_model_dir = application.config.get("COLD_MODEL_DIR") or model_dir
    if "MAGNET_COLD_START_MODEL_DIR" in os.environ:
        cold_model_dir = None
    elif "MAGNET_MODEL_DIR" in os.environ and application.config.get("COLD_MODEL_DIR") is None:
        cold_model_dir = os.environ["MAGNET_MODEL_DIR"]
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

    downloadable_artifacts = {
        "paper-p95-csv": {
            "title": "A-E manuscript P95 metrics (CSV)",
            "path": artifact_root / "results" / "paper_evaluation" / "p95_metrics.csv",
            "kind": "Reported manuscript metric",
        },
        "paper-metrics-json": {
            "title": "A-E manuscript metrics (JSON)",
            "path": artifact_root / "results" / "paper_evaluation" / "paper_metrics.json",
            "kind": "Reported manuscript metric",
        },
        "pfc-validation-csv": {
            "title": "Independent PFC validation metrics (CSV)",
            "path": artifact_root / "results" / "pfc_validation_metrics.csv",
            "kind": "Independent measured validation",
        },
        "pareto-sequence-svg": {
            "title": "Parameter versus sequence-P95 comparison (SVG)",
            "path": artifact_root / "results" / "paper_evaluation" / "pareto_comparison" / "pareto_sequence_p95.svg",
            "kind": "Editable vector figure",
        },
        "pareto-energy-svg": {
            "title": "Parameter versus energy-P95 comparison (SVG)",
            "path": artifact_root / "results" / "paper_evaluation" / "pareto_comparison" / "pareto_energy_p95.svg",
            "kind": "Editable vector figure",
        },
        "synthetic-upload-csv": {
            "title": "Four-material Test Center B-only upload example (CSV)",
            "path": artifact_root / "static" / "resources" / "B_TestCenter.csv",
            "kind": "Material-independent browser input fixture",
        },
        "synthetic-finetune-format": {
            "title": "Fine-tuning data-format example (ZIP)",
            "path": artifact_root / "examples" / "data" / "synthetic_finetune_format.zip",
            "kind": "Synthetic format-only fixture",
        },
        "synthetic-manifest-json": {
            "title": "Reproducible-example manifest (JSON)",
            "path": artifact_root / "results" / "reproducible_examples" / "manifest.json",
            "kind": "Synthetic software verification",
        },
        "gpu-verification-json": {
            "title": "AutoDL GPU checkpoint verification (JSON)",
            "path": artifact_root / "results" / "verification" / "verification_autodl_gpu.json",
            "kind": "Software and artifact verification",
        },
    }

    @application.get("/")
    def index():
        return render_template("index.html")

    @application.get("/predict")
    def predict_page():
        return render_template("predict.html")

    @application.get("/test")
    def reference_test_page():
        return render_template("test.html")

    @application.get("/circuit")
    def circuit_page():
        return render_template("circuit.html")

    @application.get("/design")
    def reference_design_page():
        return render_template("design.html")

    @application.get("/docs")
    def docs_page():
        return render_template("docs.html")

    @application.get("/document")
    def reference_document_page():
        return render_template("document.html")

    @application.get("/fine_tune")
    def reference_model_library_page():
        return render_template("fine_tune.html")

    @application.get("/resources")
    def resources_page():
        artifacts = [
            {
                "id": artifact_id,
                "title": item["title"],
                "kind": item["kind"],
                "available": item["path"].is_file(),
                "filename": item["path"].name,
            }
            for artifact_id, item in downloadable_artifacts.items()
        ]
        return render_template("resources.html", artifacts=artifacts)

    @application.get("/downloads/<artifact_id>")
    def download_artifact(artifact_id: str):
        item = downloadable_artifacts.get(artifact_id)
        if item is None or not item["path"].is_file():
            abort(404)
        return send_file(
            item["path"],
            as_attachment=True,
            download_name=item["path"].name,
            conditional=True,
        )

    @application.get("/resources/<path:resource_name>")
    def reference_resource(resource_name: str):
        return send_from_directory(static_root / "resources", resource_name)

    @application.get("/licenses/<path:license_name>")
    def license_text(license_name: str):
        return send_from_directory(license_root, license_name)

    @application.get("/health")
    def health():
        models = service.manager.public_models()
        cold_models = cold_service.manager.public_models()
        return jsonify(
            {
                "status": "ok",
                "version": "1.3.2",
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
                        "Use the reproducible offline training workflow and the downloadable "
                        "data-format example from the Resources page."
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
