"""
SkyGuardAI — Flask Backend API
================================
Endpoints
---------
GET  /health            → Health-check; confirms model artefacts are loaded.
GET  /stations          → Lists all configured monitoring stations.
POST /predict           → Runs anomaly detection on incoming sensor readings.
POST /ingest            → Fetches fresh Open-Meteo data and pushes to MongoDB.
POST /train             → Kicks off the full 4-stage training pipeline.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Environment ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── SkyGuard internals ───────────────────────────────────────────────────────
from skyguard.pipeline.prediction_pipeline import PredictionPipeline
from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging as sky_logger

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins — tighten in production with origins=[...]

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("skyguard.app")

# ── Station Registry (mirrors push_data.py) ──────────────────────────────────
STATIONS: dict = {
    "delhi":     {"latitude": 28.61,  "longitude": 77.20,  "label": "Delhi"},
    "mumbai":    {"latitude": 19.07,  "longitude": 72.88,  "label": "Mumbai"},
    "jodhpur":   {"latitude": 26.29,  "longitude": 73.02,  "label": "Jodhpur"},
    "lucknow":   {"latitude": 26.85,  "longitude": 80.94,  "label": "Lucknow"},
    "kochi":     {"latitude": 9.93,   "longitude": 76.27,  "label": "Kochi"},
    "manali":    {"latitude": 32.27,  "longitude": 77.17,  "label": "Manali"},
}

# ── Lazy-load prediction pipeline (avoids crash if artefacts are missing) ────
_pipeline: PredictionPipeline | None = None

def _get_pipeline() -> PredictionPipeline:
    """Return a cached PredictionPipeline, initialising it on first call."""
    global _pipeline
    if _pipeline is None:
        log.info("Loading PredictionPipeline artefacts …")
        _pipeline = PredictionPipeline()
        log.info("PredictionPipeline ready.")
    return _pipeline


# ═══════════════════════════════════════════════════════════════════════════
#  Helper — safe JSON serialisation (converts numpy types → Python natives)
# ═══════════════════════════════════════════════════════════════════════════
def _safe_jsonify(data):
    """Recursively coerce numpy / non-serialisable types to JSON-safe ones."""
    import numpy as np  # local import — not always present at module level

    if isinstance(data, dict):
        return {k: _safe_jsonify(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_safe_jsonify(v) for v in data]
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, (np.bool_,)):
        return bool(data)
    if isinstance(data, (np.ndarray,)):
        return data.tolist()
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  GET /health
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    """
    Returns HTTP 200 when the service is running.
    Also reports whether model artefacts are loaded.
    """
    artefacts_loaded = _pipeline is not None and _pipeline.model is not None
    return jsonify({
        "status": "ok",
        "artefacts_loaded": artefacts_loaded,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), 200


# ═══════════════════════════════════════════════════════════════════════════
#  GET /stations
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/stations", methods=["GET"])
def stations():
    """
    Returns the list of stations SkyGuardAI monitors.

    Response body
    -------------
    {
      "stations": {
        "delhi":   { "latitude": 28.61, "longitude": 77.20, "label": "Delhi" },
        ...
      }
    }
    """
    return jsonify({"stations": STATIONS}), 200


# ═══════════════════════════════════════════════════════════════════════════
#  POST /predict
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    """
    Run anomaly detection on one or more sensor readings.

    Request body (JSON)
    -------------------
    Single reading:
        {
            "time":                   "2026-09-29T14:00",   ← optional (defaults to now)
            "station":                "delhi_2025",          ← optional (defaults to "unknown")
            "temperature_2m":         28.5,
            "relative_humidity_2m":   62.0,
            "surface_pressure":       988.0
        }

    Batch of readings (pass a list):
        [{ ... }, { ... }]

    Response body
    -------------
    {
      "predictions": [
        {
          "time":                    "2026-09-29T14:00",
          "station":                 "delhi_2025",
          "temperature_2m":          28.5,
          "relative_humidity_2m":    62.0,
          "surface_pressure":        988.0,
          "is_anomaly":              false,
          "anomaly_score":           0.3214,
          "severity":                "NORMAL",
          "anomaly_type":            "Normal Reading",
          "reason":                  "All sensor parameters ...",
          "top_contributing_features": [...]
        }
      ],
      "count": 1,
      "timestamp": "..."
    }
    """
    try:
        body = request.get_json(force=True, silent=True)
        if body is None:
            return jsonify({"error": "Request body must be valid JSON."}), 400

        # Accept single dict OR a list
        if isinstance(body, dict):
            payload = [body]
        elif isinstance(body, list):
            payload = body
        else:
            return jsonify({"error": "Body must be a JSON object or a JSON array."}), 400

        if len(payload) == 0:
            return jsonify({"error": "Payload list is empty."}), 400

        pipeline = _get_pipeline()

        if pipeline.model is None or pipeline.preprocessor is None:
            return jsonify({
                "error": (
                    "Model artefacts are not loaded. "
                    "Run POST /train first to train the model."
                )
            }), 503

        predictions = pipeline.predict(payload)
        predictions = _safe_jsonify(predictions)

        return jsonify({
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }), 200

    except CustomException as ce:
        log.error(f"[/predict] CustomException: {ce}")
        return jsonify({"error": str(ce)}), 500
    except Exception as exc:
        log.error(f"[/predict] Unexpected error:\n{traceback.format_exc()}")
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  POST /ingest
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/ingest", methods=["POST"])
def ingest():
    """
    Fetch fresh weather data from Open-Meteo for one or all stations
    and push the records to MongoDB.

    Request body (JSON)  — all fields optional
    -------------------------------------------
    {
        "station":   "delhi",        ← key from STATIONS dict; omit to ingest ALL stations
        "days":      7               ← forecast horizon (1-16), default 7
    }

    Response body
    -------------
    {
      "inserted": { "delhi": 168, "mumbai": 168, ... },
      "total_inserted": 336,
      "timestamp": "..."
    }
    """
    try:
        import requests as http_requests
        import certifi
        import pymongo
        from skyguard.constant.training_pipeline import (
            DATABASE_NAME,
            COLLECTION_NAME,
        )

        body     = request.get_json(force=True, silent=True) or {}
        target   = body.get("station", None)         # None → all stations
        days     = int(body.get("days", 7))
        days     = max(1, min(days, 16))              # clamp to Open-Meteo range

        # Station selection
        targets: dict = {}
        if target:
            target = target.lower()
            if target not in STATIONS:
                return jsonify({
                    "error": f"Unknown station '{target}'. Valid keys: {list(STATIONS.keys())}"
                }), 400
            targets = {target: STATIONS[target]}
        else:
            targets = STATIONS

        # MongoDB connection
        mongo_url = os.getenv("Mongo_DB_URL") or os.getenv("MONGO_DB_URL")
        if not mongo_url:
            return jsonify({"error": "Mongo_DB_URL environment variable is not set."}), 500

        mongo_client = pymongo.MongoClient(mongo_url, tlsCAFile=certifi.where())
        collection   = mongo_client[DATABASE_NAME][COLLECTION_NAME]

        inserted_counts: dict = {}
        HOURLY_FIELDS = "temperature_2m,relative_humidity_2m,surface_pressure"

        for station_key, station_meta in targets.items():
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={station_meta['latitude']}"
                f"&longitude={station_meta['longitude']}"
                f"&hourly={HOURLY_FIELDS}"
                f"&forecast_days={days}"
            )
            log.info(f"[/ingest] Fetching Open-Meteo for {station_key}: {url}")

            resp = http_requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            times       = hourly.get("time", [])
            temps       = hourly.get("temperature_2m", [])
            humids      = hourly.get("relative_humidity_2m", [])
            pressures   = hourly.get("surface_pressure", [])

            if not times:
                log.warning(f"[/ingest] No hourly data returned for {station_key}.")
                inserted_counts[station_key] = 0
                continue

            records = [
                {
                    "time":                  times[i],
                    "temperature_2m":        temps[i],
                    "relative_humidity_2m":  humids[i],
                    "surface_pressure":      pressures[i],
                    "station":               f"{station_key}_2025",
                    "ingested_at":           datetime.utcnow().isoformat() + "Z",
                }
                for i in range(len(times))
            ]

            result = collection.insert_many(records)
            inserted_counts[station_key] = len(result.inserted_ids)
            log.info(f"[/ingest] {station_key}: inserted {inserted_counts[station_key]} records.")

        mongo_client.close()

        return jsonify({
            "inserted": inserted_counts,
            "total_inserted": sum(inserted_counts.values()),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }), 200

    except Exception as exc:
        log.error(f"[/ingest] Error:\n{traceback.format_exc()}")
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  POST /train
# ═══════════════════════════════════════════════════════════════════════════
@app.route("/train", methods=["POST"])
def train():
    """
    Kick off the full 4-stage SkyGuardAI training pipeline:
        DataIngestion → DataValidation → DataTransformation → ModelTrainer

    This runs SYNCHRONOUSLY (blocks until training completes).
    For large datasets, consider spawning a background thread or Celery task.

    Response body
    -------------
    {
      "status":  "success",
      "message": "Training pipeline completed successfully.",
      "timestamp": "..."
    }
    """
    global _pipeline  # reset cached pipeline so next /predict reloads fresh artefacts

    try:
        log.info("[/train] Starting training pipeline …")

        # Import here to avoid circular imports at module load time
        from skyguard.components.data_ingestion import DataIngestion
        from skyguard.components.data_validation import DataValidation
        from skyguard.components.data_transformation import DataTransformation
        from skyguard.components.model_trainer import ModelTrainer
        from skyguard.entity.config_entity import (
            TrainingPipelineConfig,
            DataIngestionConfig,
            DataValidationConfig,
            DataTransformationConfig,
            ModelTrainerConfig,
        )

        training_pipeline_config  = TrainingPipelineConfig()

        # ── Stage 1: Data Ingestion ───────────────────────────────────────
        log.info("[/train] Stage 1/4 — Data Ingestion")
        data_ingestion_config  = DataIngestionConfig(training_pipeline_config)
        data_ingestion         = DataIngestion(data_ingestion_config)
        ingestion_artifact     = data_ingestion.initiate_data_ingestion()

        # ── Stage 2: Data Validation ──────────────────────────────────────
        log.info("[/train] Stage 2/4 — Data Validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation        = DataValidation(ingestion_artifact, data_validation_config)
        validation_artifact    = data_validation.initiate_data_validation()

        # ── Stage 3: Data Transformation ──────────────────────────────────
        log.info("[/train] Stage 3/4 — Data Transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation        = DataTransformation(validation_artifact, data_transformation_config)
        transformation_artifact    = data_transformation.initiate_data_transformation()

        # ── Stage 4: Model Training ───────────────────────────────────────
        log.info("[/train] Stage 4/4 — Model Training")
        model_trainer_config   = ModelTrainerConfig(training_pipeline_config)
        model_trainer          = ModelTrainer(transformation_artifact, model_trainer_config)
        trainer_artifact       = model_trainer.initiate_model_trainer()

        # Reset the cached pipeline so the next /predict call loads fresh artefacts
        _pipeline = None
        log.info("[/train] Training pipeline completed — pipeline cache invalidated.")

        return jsonify({
            "status":    "success",
            "message":   "Training pipeline completed successfully.",
            "model":     str(getattr(trainer_artifact, "trained_model_file_path", "n/a")),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }), 200

    except CustomException as ce:
        log.error(f"[/train] CustomException:\n{traceback.format_exc()}")
        return jsonify({"error": str(ce), "status": "failed"}), 500
    except Exception as exc:
        log.error(f"[/train] Unexpected error:\n{traceback.format_exc()}")
        return jsonify({"error": str(exc), "status": "failed"}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    log.info(f"SkyGuardAI backend starting on http://0.0.0.0:{port}  (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)
