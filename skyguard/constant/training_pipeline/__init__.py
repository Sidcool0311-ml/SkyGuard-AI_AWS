import os
from datetime import datetime
import pandas as pd
import numpy as np

# ---------------- Pipeline-level ----------------
PIPELINE_NAME: str = "SkyGuardAI"
ARTIFACT_DIR: str = "Artifacts"

# ---------------- Columns ----------------
TEMPERATURE_COLUMN: str = "temperature_2m"
PRESSURE_COLUMN: str = "surface_pressure"
HUMIDITY_COLUMN: str = "relative_humidity_2m"
STATION_COLUMN: str = "station"
TIME_COLUMN: str = "time"
TARGET_COLUMN: str = "is_anomaly"   # only present after synthetic anomaly injection

# ---------------- Schema ----------------

#SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")

# ---------------- MongoDB ----------------
DATABASE_NAME: str = "SkyGuardDB"
COLLECTION_NAME: str = "Sensor_readings"
# ---------------- Data Ingestion (filenames) ----------------
FILE_NAME: str = "sensor_data.csv"          # raw combined data pulled from MongoDB
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# ---------------- Data Ingestion ----------------

DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2
"""
# ---------------- Data Validation ----------------
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# ---------------- Data Transformation ----------------
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessor.pkl"

# ---------------- Model Trainer ----------------
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_FILE_NAME: str = "model.pkl"

# ---------------- Timestamp ----------------
def get_timestamp() -> str:
    return datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
"""