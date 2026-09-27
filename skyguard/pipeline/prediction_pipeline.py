import os
import sys
import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Optional

from skyguard.constant.training_pipeline import (
    ARTIFACT_DIR,
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_FILE_NAME,
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    PREPROCESSING_OBJECT_FILE_NAME,
    MODEL_TRAINER_EXPLAINER_DIR,
    SHAP_EXPLAINER_FILE_NAME,
    LIME_EXPLAINER_FILE_NAME,
    BACKGROUND_DATA_FILE_NAME,
    TEMPERATURE_COLUMN,
    HUMIDITY_COLUMN,
    PRESSURE_COLUMN,
    STATION_COLUMN,
    TIME_COLUMN,
)
from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging
from skyguard.utils.main_utils.utils import load_object, load_numpy_array_data

try:
    import shap
except ImportError:
    shap = None

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    lime = None
    LimeTabularExplainer = None


class PredictionPipeline:
    """
    Production-Ready Real-Time Inference Pipeline:
    1. Loads trained preprocessor pipeline, champion anomaly detection model, and XAI explainers.
    2. Ingests raw telemetry (dict, list of dicts, or DataFrame) directly from IoT streams or Open-Meteo API.
    3. Transforms and predicts continuous anomaly scores and binary decisions.
    4. Applies SHAP and LIME to discover root-cause feature attributions.
    5. Classifies the specific Anomaly Type (Spike, Flatline, Inconsistency, Out-of-bounds) and
       synthesizes a human-readable diagnostic reason instead of a simple yes/no.
    """

    def __init__(
        self,
        model_file_path: Optional[str] = None,
        preprocessor_file_path: Optional[str] = None,
        shap_explainer_file_path: Optional[str] = None,
        lime_explainer_file_path: Optional[str] = None,
        background_data_file_path: Optional[str] = None,
    ):
        try:
            self.model_file_path = model_file_path or os.path.join(
                ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME
            )
            self.preprocessor_file_path = preprocessor_file_path or os.path.join(
                ARTIFACT_DIR, DATA_TRANSFORMATION_DIR_NAME, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME
            )
            self.shap_explainer_file_path = shap_explainer_file_path or os.path.join(
                ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_EXPLAINER_DIR, SHAP_EXPLAINER_FILE_NAME
            )
            self.lime_explainer_file_path = lime_explainer_file_path or os.path.join(
                ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_EXPLAINER_DIR, LIME_EXPLAINER_FILE_NAME
            )
            self.background_data_file_path = background_data_file_path or os.path.join(
                ARTIFACT_DIR, MODEL_TRAINER_DIR_NAME, MODEL_TRAINER_EXPLAINER_DIR, BACKGROUND_DATA_FILE_NAME
            )

            self.model = None
            self.preprocessor = None
            self.shap_explainer = None
            self.lime_explainer = None
            self.background_data = None
            self.feature_names = []

            self._load_artifacts()
        except Exception as e:
            raise CustomException(e, sys)

    def _load_artifacts(self):
        """
        Loads models, pipelines, and explainers with graceful status logging.
        """
        try:
            if os.path.exists(self.preprocessor_file_path):
                self.preprocessor = load_object(self.preprocessor_file_path)
                logging.info(f"Loaded preprocessor from {self.preprocessor_file_path}")
                self._extract_feature_names()
            else:
                logging.warning(f"Preprocessor file not found at {self.preprocessor_file_path}")

            if os.path.exists(self.model_file_path):
                self.model = load_object(self.model_file_path)
                logging.info(f"Loaded champion model from {self.model_file_path}")
            else:
                logging.warning(f"Model file not found at {self.model_file_path}")

            if os.path.exists(self.background_data_file_path):
                self.background_data = load_numpy_array_data(self.background_data_file_path)

            if os.path.exists(self.shap_explainer_file_path):
                try:
                    self.shap_explainer = load_object(self.shap_explainer_file_path)
                    logging.info("Loaded SHAP explainer successfully.")
                except Exception as e:
                    logging.warning(f"Could not load SHAP explainer: {e}")

            if os.path.exists(self.lime_explainer_file_path):
                try:
                    self.lime_explainer = load_object(self.lime_explainer_file_path)
                    logging.info("Loaded LIME explainer successfully.")
                except Exception as e:
                    logging.warning(f"Could not load LIME explainer: {e}")

        except Exception as e:
            raise CustomException(e, sys)

    def _extract_feature_names(self):
        """Extracts human-readable feature names from the loaded preprocessor."""
        try:
            if hasattr(self.preprocessor, "named_steps"):
                col_transformer = self.preprocessor.named_steps.get("preprocessor")
                if col_transformer and hasattr(col_transformer, "get_feature_names_out"):
                    raw_names = list(col_transformer.get_feature_names_out())
                    self.feature_names = [
                        n.replace("num_pipeline__", "").replace("cat_pipeline__", "")
                        for n in raw_names
                    ]
        except Exception:
            self.feature_names = []

    def _compute_shap_contributions(self, transformed_row: np.ndarray) -> List[Dict[str, Any]]:
        """
        Computes SHAP local feature importance for a single sample.
        """
        contributions = []
        if self.shap_explainer is not None and shap is not None:
            try:
                shap_values = self.shap_explainer(transformed_row.reshape(1, -1))
                values = shap_values.values[0]
                if isinstance(values, list):
                    values = values[0]
                for idx, val in enumerate(values):
                    feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                    contributions.append({
                        "feature": feat_name,
                        "impact": float(val),
                        "abs_impact": abs(float(val)),
                        "method": "SHAP",
                    })
                contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
                return contributions
            except Exception as e:
                logging.warning(f"SHAP explanation calculation failed: {e}")

        return contributions

    def _compute_lime_contributions(self, transformed_row: np.ndarray) -> List[Dict[str, Any]]:
        """
        Computes LIME local surrogate feature importance for a single sample.
        """
        contributions = []
        if self.lime_explainer is not None and self.model is not None and hasattr(self.model, "decision_function"):
            try:
                exp = self.lime_explainer.explain_instance(
                    data_row=transformed_row,
                    predict_fn=self.model.decision_function,
                    num_features=5,
                )
                for feat, weight in exp.as_list():
                    contributions.append({
                        "feature": feat,
                        "impact": float(weight),
                        "abs_impact": abs(float(weight)),
                        "method": "LIME",
                    })
                contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
                return contributions
            except Exception as e:
                logging.warning(f"LIME explanation calculation failed: {e}")

        return contributions

    def _diagnose_anomaly(
        self,
        raw_row: pd.Series,
        transformed_row: np.ndarray,
        is_anomaly: bool,
        anomaly_score: float,
        top_features: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        """
        Synthesizes the specific Type of Anomaly and an actionable natural language Reason.
        """
        if not is_anomaly:
            return "Normal Operation", "All atmospheric telemetry parameters conform to historical station baselines."

        temp = float(raw_row.get(TEMPERATURE_COLUMN, 0.0))
        humidity = float(raw_row.get(HUMIDITY_COLUMN, 50.0))
        pressure = float(raw_row.get(PRESSURE_COLUMN, 1013.25))

        top_feature_names = [f["feature"] for f in top_features[:3]]
        feature_summary = ", ".join([f"{f['feature']} ({f['impact']:+.2f})" for f in top_features[:2]])

        # 1. Physical Domain Out-of-Bounds Check
        if temp < -40.0 or temp > 55.0:
            return (
                "Out-of-Bounds Extreme Reading",
                f"Extreme atmospheric temperature violation ({temp:.1f}°C) exceeding physical Earth limits. "
                f"Top contributing risk factors: {feature_summary}."
            )
        if humidity < 0.0 or humidity > 100.0:
            return (
                "Out-of-Bounds Extreme Reading",
                f"Relative humidity transducer saturation ({humidity:.1f}%) outside 0-100% valid range. "
                f"Top contributing risk factors: {feature_summary}."
            )
        if pressure < 600.0 or pressure > 1080.0:
            return (
                "Out-of-Bounds Extreme Reading",
                f"Barometric pressure reading ({pressure:.1f} hPa) violates plausible meteorological ranges. "
                f"Top contributing risk factors: {feature_summary}."
            )

        # 2. Rate-of-Change / Impulse Spike Check
        has_spike_driver = any("delta_" in name for name in top_feature_names)
        if has_spike_driver:
            spike_features = [f for f in top_features if "delta_" in f["feature"]]
            primary_spike = spike_features[0]["feature"] if spike_features else "sensor delta"
            return (
                "Impulse Spike / Sudden Jump",
                f"Abrupt transient jump detected driven primarily by {primary_spike}. "
                f"Rate of change significantly exceeds typical weather gradients. "
                f"Key drivers: {feature_summary}."
            )

        # 3. Thermodynamic / Physical Inconsistency Check
        inconsistency_keywords = ["discomfort_index", "dew_point", "temp_pressure_ratio", "interaction"]
        has_inconsistency = any(any(kw in name for kw in inconsistency_keywords) for name in top_feature_names)
        if has_inconsistency:
            return (
                "Thermodynamic Inconsistency",
                f"Violation of multivariate atmospheric physics: temperature ({temp:.1f}°C), "
                f"relative humidity ({humidity:.1f}%), and pressure ({pressure:.1f} hPa) moved in contradictory directions. "
                f"Identified by XAI: {feature_summary}."
            )

        # 4. Sensor Freezing / Flatline Check
        if any("abs_delta" in name for name in top_feature_names):
            return (
                "Sensor Freezing / Flatline",
                f"Telemetry indicates static sensor output with zero variance across consecutive readings while regional conditions shifted. "
                f"Attributed to: {feature_summary}."
            )

        # 5. Multivariate Outlier (General Manifold Deviation)
        return (
            "Multivariate Density Outlier",
            f"Sensor combination deviates significantly from normal correlated multi-parameter distribution at this station. "
            f"Anomaly score: {anomaly_score:.2f}. Primary contributing drivers: {feature_summary}."
        )

    def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Executes end-to-end prediction and diagnostic reasoning on new data.
        Returns a rich list of diagnostic dictionaries.
        """
        try:
            if self.model is None or self.preprocessor is None:
                self._load_artifacts()
                if self.model is None or self.preprocessor is None:
                    raise Exception("Model or Preprocessor artifact is missing. Please run ModelTrainer first.")

            # Standardize input to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Input data must be a dictionary, list of dictionaries, or pandas DataFrame.")

            # Keep copy of raw inputs for reporting
            raw_df = df.copy()

            # Ensure required columns exist, impute defaults if missing from single streaming points
            if TIME_COLUMN not in df.columns:
                df[TIME_COLUMN] = pd.Timestamp.now().isoformat()
            if STATION_COLUMN not in df.columns:
                df[STATION_COLUMN] = "unknown_station"

            # Transform raw features using end-to-end preprocessor pipeline
            transformed_features = self.preprocessor.transform(df)

            # Raw prediction: -1 = Anomaly, 1 = Normal
            raw_predictions = self.model.predict(transformed_features)

            # Continuous anomaly scores via decision function
            if hasattr(self.model, "decision_function"):
                decision_scores = self.model.decision_function(transformed_features)
                # Map decision function to normalized anomaly score in [0.0, 1.0]
                # Lower decision score = more anomalous
                anomaly_scores = 1.0 / (1.0 + np.exp(decision_scores * 4.0))
            else:
                anomaly_scores = np.where(raw_predictions == -1, 0.85, 0.15)

            results = []

            for i in range(len(df)):
                raw_row = raw_df.iloc[i]
                transformed_row = transformed_features[i]
                is_anomaly = bool(raw_predictions[i] == -1)
                score = float(anomaly_scores[i])

                # Severity classification
                if score >= 0.75:
                    severity = "CRITICAL"
                elif score >= 0.55:
                    severity = "WARNING"
                elif score >= 0.40:
                    severity = "ELEVATED"
                else:
                    severity = "NORMAL"

                # Explainability: Combine SHAP and LIME
                top_contributions = []
                if is_anomaly or score > 0.40:
                    shap_contribs = self._compute_shap_contributions(transformed_row)
                    if shap_contribs:
                        top_contributions = shap_contribs[:5]
                    else:
                        lime_contribs = self._compute_lime_contributions(transformed_row)
                        top_contributions = lime_contribs[:5]

                # Fallback heuristic contribution if SHAP and LIME packages are not active
                if not top_contributions and (is_anomaly or score > 0.40):
                    abs_vals = np.abs(transformed_row)
                    top_indices = np.argsort(abs_vals)[::-1][:4]
                    for idx in top_indices:
                        feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                        top_contributions.append({
                            "feature": feat_name,
                            "impact": float(transformed_row[idx]),
                            "abs_impact": float(abs_vals[idx]),
                            "method": "Heuristic Attribution",
                        })

                # Determine specific Type of Anomaly and Detailed Reason
                anomaly_type, anomaly_reason = self._diagnose_anomaly(
                    raw_row=raw_row,
                    transformed_row=transformed_row,
                    is_anomaly=is_anomaly,
                    anomaly_score=score,
                    top_features=top_contributions,
                )

                result_entry = {
                    "time": str(raw_row.get(TIME_COLUMN, "")),
                    "station": str(raw_row.get(STATION_COLUMN, "")),
                    "temperature_2m": float(raw_row.get(TEMPERATURE_COLUMN, 0.0)),
                    "relative_humidity_2m": float(raw_row.get(HUMIDITY_COLUMN, 0.0)),
                    "surface_pressure": float(raw_row.get(PRESSURE_COLUMN, 0.0)),
                    "is_anomaly": is_anomaly,
                    "anomaly_score": round(score, 4),
                    "severity": severity,
                    "anomaly_type": anomaly_type,
                    "reason": anomaly_reason,
                    "top_contributing_features": [
                        {"feature": item["feature"], "impact": round(item["impact"], 4)}
                        for item in top_contributions[:3]
                    ],
                }

                results.append(result_entry)

            return results

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Quick Test Execution of Prediction Pipeline
    try:
        pipeline = PredictionPipeline()

        test_samples = [
            # 1. Normal Sample
            {
                "time": "2026-09-27T12:00",
                "station": "delhi_2025",
                "temperature_2m": 28.5,
                "relative_humidity_2m": 62.0,
                "surface_pressure": 988.0,
            },
            # 2. Extreme Spike Anomaly
            {
                "time": "2026-09-27T13:00",
                "station": "delhi_2025",
                "temperature_2m": 49.8,
                "relative_humidity_2m": 12.0,
                "surface_pressure": 988.0,
            },
            # 3. Thermodynamic Inconsistency Anomaly
            {
                "time": "2026-09-27T14:00",
                "station": "jodhpur_2025",
                "temperature_2m": 38.0,
                "relative_humidity_2m": 99.0,
                "surface_pressure": 820.0,
            },
        ]

        print("Running diagnostic prediction pipeline...")
        diagnostics = pipeline.predict(test_samples)
        for diag in diagnostics:
            print("\n-------------------------------------------")
            print(f"Station: {diag['station']} | Time: {diag['time']}")
            print(f"Is Anomaly: {diag['is_anomaly']} | Severity: {diag['severity']} (Score: {diag['anomaly_score']})")
            print(f"Anomaly Type: {diag['anomaly_type']}")
            print(f"Reason: {diag['reason']}")
            if diag['top_contributing_features']:
                print("Contributing Factors:", diag['top_contributing_features'])

    except Exception as e:
        print(f"Error executing prediction pipeline test: {e}")

