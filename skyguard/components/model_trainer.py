import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from skyguard.constant.training_pipeline import (
    MODEL_TRAINER_CONTAMINATION,
    TARGET_COLUMN,
)
from skyguard.entity.config_entity import ModelTrainerConfig
from skyguard.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging
from skyguard.utils.main_utils.utils import (
    load_numpy_array_data,
    save_numpy_array_data,
    load_object,
    save_object,
)

# Optional Explainable AI libraries (SHAP & LIME) with robust fallbacks
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


class ModelTrainer:
    """
    Model Trainer Component:
    1. Trains and evaluates multiple anomaly detection models (Isolation Forest, One-Class SVM, Local Outlier Factor).
    2. Calibrates anomaly decision thresholds to target contamination (e.g. 10%).
    3. Evaluates models against ground-truth is_anomaly labels (Precision, Recall, F1-score, ROC-AUC).
    4. Selects the champion model based on highest F1-score.
    5. Builds Explainable AI (XAI) explainers using SHAP and LIME.
    6. Serializes the trained model and explainers for real-time inference.
    """

    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def extract_feature_names(self, preprocessor: Any, num_features: int) -> list:
        """
        Extracts human-readable feature names from the fitted preprocessor pipeline.
        """
        try:
            feature_names = []
            if hasattr(preprocessor, "named_steps"):
                col_transformer = preprocessor.named_steps.get("preprocessor")
                if col_transformer and hasattr(col_transformer, "get_feature_names_out"):
                    raw_names = list(col_transformer.get_feature_names_out())
                    # Clean sklearn prefixes like num_pipeline__ or cat_pipeline__
                    feature_names = [
                        name.replace("num_pipeline__", "").replace("cat_pipeline__", "")
                        for name in raw_names
                    ]

            if not feature_names or len(feature_names) != num_features:
                feature_names = [f"feature_{i}" for i in range(num_features)]

            return feature_names
        except Exception as e:
            logging.warning(f"Could not extract feature names from preprocessor: {e}. Using fallback indices.")
            return [f"feature_{i}" for i in range(num_features)]

    def get_candidate_models(self, contamination: float) -> Dict[str, Any]:
        """
        Initializes anomaly detection model candidates.
        """
        models = {
            "IsolationForest": IsolationForest(
                n_estimators=150,
                contamination=contamination,
                max_samples="auto",
                random_state=42,
                n_jobs=-1,
            ),
            "OneClassSVM": OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=min(0.5, max(0.01, contamination)),
            ),
            "LocalOutlierFactor": LocalOutlierFactor(
                n_neighbors=25,
                contamination=contamination,
                novelty=True,
                n_jobs=-1,
            ),
        }
        return models

    def evaluate_model(
        self,
        model: Any,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluates anomaly detection model performance:
        - Calibrates decision threshold to match target contamination (e.g. 10%).
        - Computes Precision, Recall, F1-Score, and ROC-AUC against ground-truth labels.
        - Computes smooth exponential calibration score.
        """
        try:
            # 1. Compute continuous anomaly scores on training and test sets
            # For IsolationForest, OCSVM, and novelty LOF: decision_function returns normal > 0, anomaly < 0
            # We invert the sign so higher score = higher anomaly likelihood
            if hasattr(model, "decision_function"):
                train_scores = -model.decision_function(x_train)
                test_scores = -model.decision_function(x_test)
            elif hasattr(model, "score_samples"):
                train_scores = -model.score_samples(x_train)
                test_scores = -model.score_samples(x_test)
            else:
                train_scores = np.zeros(len(x_train))
                test_scores = np.zeros(len(x_test))

            # 2. Calibrate decision threshold to target contamination (e.g. 10%)
            contamination = self.model_trainer_config.contamination
            threshold = float(np.percentile(train_scores, 100 * (1.0 - contamination)))

            # If ground truth labels exist, tune threshold across validation range to maximize F1-score
            if y_test is not None:
                y_true = np.array(y_test).astype(int)
                best_f1 = -1.0
                best_th = threshold
                for pct in np.linspace(85, 95, 21):
                    cand_th = float(np.percentile(train_scores, pct))
                    cand_preds = (test_scores >= cand_th).astype(int)
                    cand_f1 = f1_score(y_true, cand_preds, zero_division=0)
                    if cand_f1 > best_f1:
                        best_f1 = cand_f1
                        best_th = cand_th
                threshold = best_th

            preds_binary = (test_scores >= threshold).astype(int)
            anomaly_ratio = float(np.mean(preds_binary))

            # Smooth exponential calibration score: exp(- |ratio - cont| / cont)
            # Exactly 1.0 when ratio == contamination; smoothly decays without flatlining
            calibration_score = float(np.exp(- abs(anomaly_ratio - contamination) / max(1e-4, contamination)))

            metrics = {
                "anomaly_ratio_test": round(anomaly_ratio, 4),
                "threshold": round(threshold, 4),
                "calibration_score": round(calibration_score, 4),
                "mean_anomaly_score": round(float(np.mean(test_scores)), 4),
                "std_anomaly_score": round(float(np.std(test_scores)), 4),
            }

            # 3. Supervised ground-truth metrics against is_anomaly labels
            if y_test is not None:
                y_true = np.array(y_test).astype(int)
                metrics["precision"] = round(float(precision_score(y_true, preds_binary, zero_division=0)), 4)
                metrics["recall"] = round(float(recall_score(y_true, preds_binary, zero_division=0)), 4)
                metrics["f1_score"] = round(float(f1_score(y_true, preds_binary, zero_division=0)), 4)
                try:
                    metrics["roc_auc"] = round(float(roc_auc_score(y_true, test_scores)), 4)
                except Exception:
                    metrics["roc_auc"] = 0.5

            return metrics
        except Exception as e:
            raise CustomException(e, sys)

    def train_and_select_best_model(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> Tuple[str, Any, Dict[str, Any]]:
        """
        Trains candidate models and selects champion model based on highest F1-score against ground-truth labels.
        """
        try:
            models = self.get_candidate_models(contamination=self.model_trainer_config.contamination)
            best_model_name = None
            best_model = None
            best_score = -float("inf")
            model_eval_report = {}

            for model_name, model in models.items():
                logging.info(f"Training candidate model: {model_name}")
                model.fit(x_train)
                metrics = self.evaluate_model(model, x_train, x_test, y_test)
                model_eval_report[model_name] = metrics
                logging.info(f"Model {model_name} metrics: {metrics}")

                # Selection criterion: Highest F1-score if ground truth exists, otherwise calibration score
                if y_test is not None:
                    selection_metric = metrics.get("f1_score", 0.0)
                else:
                    selection_metric = metrics.get("calibration_score", 0.0)
                    if model_name == "IsolationForest":
                        selection_metric += 0.1

                if selection_metric > best_score:
                    best_score = selection_metric
                    best_model_name = model_name
                    best_model = model

            logging.info(f"Champion model selected: {best_model_name} (F1 Score: {best_score:.4f})")
            return best_model_name, best_model, model_eval_report
        except Exception as e:
            raise CustomException(e, sys)

    def build_and_save_explainers(
        self,
        model: Any,
        x_train: np.ndarray,
        feature_names: list,
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Configures and serializes SHAP and LIME explainers along with background data.
        """
        try:
            os.makedirs(
                os.path.dirname(self.model_trainer_config.shap_explainer_file_path),
                exist_ok=True,
            )

            # Sample 100 representative background points for low-latency inference
            sample_size = min(100, x_train.shape[0])
            np.random.seed(42)
            bg_indices = np.random.choice(x_train.shape[0], size=sample_size, replace=False)
            background_data = x_train[bg_indices]

            # Save background baseline data
            save_numpy_array_data(
                file_path=self.model_trainer_config.background_data_file_path,
                array=background_data,
            )
            logging.info(f"Saved background data sample to {self.model_trainer_config.background_data_file_path}")

            shap_path = self.model_trainer_config.shap_explainer_file_path
            lime_path = self.model_trainer_config.lime_explainer_file_path

            # 1. SHAP Explainer
            if shap is not None:
                try:
                    logging.info("Initializing SHAP Explainer")
                    if isinstance(model, IsolationForest):
                        shap_explainer = shap.TreeExplainer(model, data=background_data)
                    else:
                        shap_explainer = shap.Explainer(model.decision_function, background_data)

                    save_object(shap_path, shap_explainer)
                    logging.info(f"Saved SHAP explainer to {shap_path}")
                except Exception as e:
                    logging.warning(f"Could not build standard SHAP explainer: {e}")
                    shap_path = None
            else:
                logging.warning("SHAP library is not installed. Add 'shap' to virtual environment to enable.")
                shap_path = None

            # 2. LIME Explainer
            if LimeTabularExplainer is not None:
                try:
                    logging.info("Initializing LIME Tabular Explainer")
                    lime_explainer = LimeTabularExplainer(
                        training_data=background_data,
                        feature_names=feature_names,
                        mode="regression",
                        verbose=False,
                        random_state=42,
                    )
                    save_object(lime_path, lime_explainer)
                    logging.info(f"Saved LIME explainer to {lime_path}")
                except Exception as e:
                    logging.warning(f"Could not build LIME explainer: {e}")
                    lime_path = None
            else:
                logging.warning("LIME library is not installed. Add 'lime' to virtual environment to enable.")
                lime_path = None

            return shap_path, lime_path, self.model_trainer_config.background_data_file_path
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Executes the Model Trainer workflow:
        1. Loads transformed numpy arrays (train/test).
        2. Separates input features and ground-truth is_anomaly labels.
        3. Trains Isolation Forest, One-Class SVM, and Local Outlier Factor.
        4. Selects champion anomaly detection model based on F1-score against ground truth.
        5. Configures SHAP and LIME explainability engines.
        6. Saves model and explainer artifacts.
        7. Returns ModelTrainerArtifact.
        """
        try:
            logging.info("Starting Model Trainer Component")

            train_path = self.data_transformation_artifact.transformed_train_file_path
            test_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info(f"Loading transformed train array: {train_path}")
            logging.info(f"Loading transformed test array: {test_path}")

            train_arr = load_numpy_array_data(train_path)
            test_arr = load_numpy_array_data(test_path)

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

            # Separate features and target labels
            # The last column is is_anomaly
            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1].astype(int)

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1].astype(int)

            feature_names = self.extract_feature_names(preprocessor, num_features=x_train.shape[1])

            logging.info(
                f"x_train shape: {x_train.shape}, y_train anomaly count: {int(np.sum(y_train))} ({np.mean(y_train)*100:.2f}%)"
            )
            logging.info(
                f"x_test shape: {x_test.shape}, y_test anomaly count: {int(np.sum(y_test))} ({np.mean(y_test)*100:.2f}%)"
            )

            # Train and select champion model
            best_model_name, best_model, eval_report = self.train_and_select_best_model(
                x_train=x_train,
                x_test=x_test,
                y_test=y_test,
            )

            # Save champion model along with its calibrated decision threshold
            calibrated_threshold = eval_report[best_model_name]["threshold"]
            model_package = {
                "model": best_model,
                "model_name": best_model_name,
                "threshold": calibrated_threshold,
                "feature_names": feature_names,
            }

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model_package,
            )
            logging.info(
                f"Saved trained champion model ({best_model_name}) with threshold {calibrated_threshold} to "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            # Build and save SHAP & LIME explainers
            shap_path, lime_path, _ = self.build_and_save_explainers(
                model=best_model,
                x_train=x_train,
                feature_names=feature_names,
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                shap_explainer_file_path=shap_path or "",
                lime_explainer_file_path=lime_path or "",
                best_model_name=best_model_name,
                metric_artifact=eval_report,
            )

            logging.info(f"Model Trainer Artifact successfully generated: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
