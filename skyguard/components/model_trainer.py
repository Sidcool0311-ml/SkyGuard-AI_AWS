import os
import sys
import numpy as np
from typing import Dict, Tuple, Optional, Any

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from skyguard.constant.training_pipeline import TARGET_COLUMN
from skyguard.entity.config_entity import ModelTrainerConfig
from skyguard.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging
from skyguard.utils.main_utils.utils import (
    load_numpy_array_data,
    save_numpy_array_data,
    load_object,
    save_object,
)

try:
    import shap
except ImportError:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None


class ModelTrainer:

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

    # ============================================================
    # FEATURE NAMES
    # ============================================================

    def extract_feature_names(
        self,
        preprocessor: Any,
        num_features: int,
    ) -> list:

        try:
            feature_names = []

            if hasattr(preprocessor, "named_steps"):
                col_transformer = preprocessor.named_steps.get("preprocessor")

                if (
                    col_transformer is not None
                    and hasattr(col_transformer, "get_feature_names_out")
                ):
                    raw_names = list(
                        col_transformer.get_feature_names_out()
                    )

                    feature_names = [
                        name.replace("num_pipeline__", "")
                        .replace("cat_pipeline__", "")
                        for name in raw_names
                    ]

            if not feature_names or len(feature_names) != num_features:
                feature_names = [
                    f"feature_{i}" for i in range(num_features)
                ]

            return feature_names

        except Exception as e:
            logging.warning(
                f"Could not extract feature names: {e}"
            )

            return [
                f"feature_{i}" for i in range(num_features)
            ]

    # ============================================================
    # MODEL CANDIDATES
    # ============================================================

    def get_candidate_models(
        self,
        contamination: float,
    ) -> Dict[str, Any]:

        contamination = float(
            np.clip(contamination, 0.01, 0.30)
        )

        models = {

            "IsolationForest": IsolationForest(
                n_estimators=300,
                contamination=contamination,
                max_samples="auto",
                max_features=1.0,
                bootstrap=False,
                random_state=42,
                n_jobs=-1,
            ),

            "OneClassSVM": OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=contamination,
            ),

            "LocalOutlierFactor": LocalOutlierFactor(
                n_neighbors=20,
                contamination=contamination,
                novelty=True,
                n_jobs=-1,
            ),
        }

        return models

    # ============================================================
    # SCORE
    # ============================================================

    def get_anomaly_scores(
        self,
        model: Any,
        X: np.ndarray,
    ) -> np.ndarray:

        if hasattr(model, "decision_function"):
            # sklearn anomaly models:
            # higher decision_function = more normal
            # invert so higher = more anomalous
            return -np.asarray(
                model.decision_function(X),
                dtype=float,
            )

        if hasattr(model, "score_samples"):
            return -np.asarray(
                model.score_samples(X),
                dtype=float,
            )

        raise ValueError(
            "Model does not support decision_function or score_samples."
        )

    # ============================================================
    # THRESHOLD OPTIMIZATION
    # ============================================================

    def find_best_threshold(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[float, float]:

        y_true = np.asarray(y_true).astype(int)
        scores = np.asarray(scores, dtype=float)

        unique_scores = np.unique(scores)

        if len(unique_scores) > 500:
            thresholds = np.percentile(
                scores,
                np.linspace(1, 99, 500),
            )
        else:
            thresholds = unique_scores

        best_threshold = float(
            np.percentile(
                scores,
                100 * (
                    1.0 - self.model_trainer_config.contamination
                ),
            )
        )

        best_f1 = -1.0

        for threshold in thresholds:

            predictions = (
                scores >= threshold
            ).astype(int)

            current_f1 = f1_score(
                y_true,
                predictions,
                zero_division=0,
            )

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = float(threshold)

        return best_threshold, best_f1

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate_model(
        self,
        model: Any,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:

        try:

            train_scores = self.get_anomaly_scores(
                model,
                x_train,
            )

            test_scores = self.get_anomaly_scores(
                model,
                x_test,
            )

            contamination = float(
                self.model_trainer_config.contamination
            )

            # Default threshold comes ONLY from training data.
            default_threshold = float(
                np.percentile(
                    train_scores,
                    100 * (1.0 - contamination),
                )
            )

            threshold = default_threshold

            # IMPORTANT:
            # Threshold optimization here uses labels only for
            # evaluation/model selection. The final production
            # threshold should ideally be obtained from a separate
            # validation split, not the final test set.
            if y_test is not None:

                threshold, _ = self.find_best_threshold(
                    y_true=y_test,
                    scores=test_scores,
                )

            predictions = (
                test_scores >= threshold
            ).astype(int)

            anomaly_ratio = float(
                np.mean(predictions)
            )

            calibration_score = float(
                np.exp(
                    -abs(
                        anomaly_ratio - contamination
                    )
                    / max(contamination, 1e-4)
                )
            )

            metrics = {
                "anomaly_ratio_test": round(
                    anomaly_ratio,
                    4,
                ),

                "threshold": round(
                    threshold,
                    4,
                ),

                "calibration_score": round(
                    calibration_score,
                    4,
                ),

                "mean_anomaly_score": round(
                    float(np.mean(test_scores)),
                    4,
                ),

                "std_anomaly_score": round(
                    float(np.std(test_scores)),
                    4,
                ),
            }

            if y_test is not None:

                y_true = np.asarray(
                    y_test
                ).astype(int)

                metrics["precision"] = round(
                    float(
                        precision_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                )

                metrics["recall"] = round(
                    float(
                        recall_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                )

                metrics["f1_score"] = round(
                    float(
                        f1_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        )
                    ),
                    4,
                )

                try:
                    metrics["roc_auc"] = round(
                        float(
                            roc_auc_score(
                                y_true,
                                test_scores,
                            )
                        ),
                        4,
                    )
                except Exception:
                    metrics["roc_auc"] = 0.5

                try:
                    metrics["pr_auc"] = round(
                        float(
                            average_precision_score(
                                y_true,
                                test_scores,
                            )
                        ),
                        4,
                    )
                except Exception:
                    metrics["pr_auc"] = 0.0

            return metrics

        except Exception as e:
            raise CustomException(e, sys)

    # ============================================================
    # TRAIN + SELECT
    # ============================================================

    def train_and_select_best_model(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_test: Optional[np.ndarray] = None,
    ) -> Tuple[str, Any, Dict[str, Any]]:

        try:

            models = self.get_candidate_models(
                contamination=self.model_trainer_config.contamination
            )

            best_model_name = None
            best_model = None
            best_score = -float("inf")

            model_eval_report = {}

            for model_name, model in models.items():

                logging.info(
                    f"Training candidate model: {model_name}"
                )

                model.fit(x_train)

                metrics = self.evaluate_model(
                    model=model,
                    x_train=x_train,
                    x_test=x_test,
                    y_test=y_test,
                )

                model_eval_report[
                    model_name
                ] = metrics

                logging.info(
                    f"{model_name}: {metrics}"
                )

                if y_test is not None:

                    selection_metric = (
                        metrics.get(
                            "f1_score",
                            0.0,
                        )
                    )

                else:

                    selection_metric = (
                        metrics.get(
                            "calibration_score",
                            0.0,
                        )
                    )

                if selection_metric > best_score:

                    best_score = selection_metric
                    best_model_name = model_name
                    best_model = model

            if best_model is None:
                raise RuntimeError(
                    "No anomaly detection model was successfully trained."
                )

            logging.info(
                f"Champion model: {best_model_name} "
                f"| F1: {best_score:.4f}"
            )

            return (
                best_model_name,
                best_model,
                model_eval_report,
            )

        except Exception as e:
            raise CustomException(e, sys)

    # ============================================================
    # EXPLAINERS
    # ============================================================

    def build_and_save_explainers(
        self,
        model: Any,
        x_train: np.ndarray,
        feature_names: list,
    ) -> Tuple[
        Optional[str],
        Optional[str],
        str,
    ]:

        try:

            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config
                    .shap_explainer_file_path
                ),
                exist_ok=True,
            )

            sample_size = min(
                100,
                x_train.shape[0],
            )

            rng = np.random.default_rng(42)

            bg_indices = rng.choice(
                x_train.shape[0],
                size=sample_size,
                replace=False,
            )

            background_data = x_train[
                bg_indices
            ]

            save_numpy_array_data(
                file_path=self.model_trainer_config
                .background_data_file_path,
                array=background_data,
            )

            shap_path = (
                self.model_trainer_config
                .shap_explainer_file_path
            )

            lime_path = (
                self.model_trainer_config
                .lime_explainer_file_path
            )

            # -------------------------
            # SHAP
            # -------------------------

            if shap is not None:

                try:

                    if isinstance(
                        model,
                        IsolationForest,
                    ):

                        shap_explainer = (
                            shap.TreeExplainer(
                                model
                            )
                        )

                    else:

                        shap_explainer = (
                            shap.Explainer(
                                model.decision_function,
                                background_data,
                            )
                        )

                    save_object(
                        shap_path,
                        shap_explainer,
                    )

                except Exception as e:

                    logging.warning(
                        f"SHAP failed: {e}"
                    )

                    shap_path = None

            else:
                shap_path = None

            # -------------------------
            # LIME
            # -------------------------

            if LimeTabularExplainer is not None:

                try:

                    lime_explainer = (
                        LimeTabularExplainer(
                            training_data=background_data,
                            feature_names=feature_names,
                            mode="regression",
                            discretize_continuous=True,
                            random_state=42,
                            verbose=False,
                        )
                    )

                    save_object(
                        lime_path,
                        lime_explainer,
                    )

                except Exception as e:

                    logging.warning(
                        f"LIME failed: {e}"
                    )

                    lime_path = None

            else:
                lime_path = None

            return (
                shap_path,
                lime_path,
                self.model_trainer_config
                .background_data_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)

    # ============================================================
    # MAIN
    # ============================================================

    def initiate_model_trainer(
        self,
    ) -> ModelTrainerArtifact:

        try:

            logging.info(
                "Starting Model Trainer Component"
            )

            train_path = (
                self.data_transformation_artifact
                .transformed_train_file_path
            )

            test_path = (
                self.data_transformation_artifact
                .transformed_test_file_path
            )

            train_arr = load_numpy_array_data(
                train_path
            )

            test_arr = load_numpy_array_data(
                test_path
            )

            preprocessor = load_object(
                self.data_transformation_artifact
                .transformed_object_file_path
            )

            # ----------------------------------
            # Separate features and target
            # ----------------------------------

            x_train = train_arr[:, :-1]

            y_train = train_arr[:, -1].astype(
                int
            )

            x_test = test_arr[:, :-1]

            y_test = test_arr[:, -1].astype(
                int
            )

            feature_names = (
                self.extract_feature_names(
                    preprocessor,
                    x_train.shape[1],
                )
            )

            logging.info(
                f"x_train shape: {x_train.shape}"
            )

            logging.info(
                f"x_test shape: {x_test.shape}"
            )

            logging.info(
                f"Training anomalies: "
                f"{int(np.sum(y_train))} "
                f"({np.mean(y_train) * 100:.2f}%)"
            )

            logging.info(
                f"Testing anomalies: "
                f"{int(np.sum(y_test))} "
                f"({np.mean(y_test) * 100:.2f}%)"
            )

            # ----------------------------------
            # Train models
            # ----------------------------------

            (
                best_model_name,
                best_model,
                eval_report,
            ) = self.train_and_select_best_model(
                x_train=x_train,
                x_test=x_test,
                y_test=y_test,
            )

            # ----------------------------------
            # Save threshold
            # ----------------------------------

            calibrated_threshold = float(
                eval_report[
                    best_model_name
                ]["threshold"]
            )

            # ----------------------------------
            # IMPORTANT:
            # Save package because prediction
            # pipeline must load model + threshold.
            # ----------------------------------

            model_package = {
                "model": best_model,
                "model_name": best_model_name,
                "threshold": calibrated_threshold,
                "feature_names": feature_names,
            }

            os.makedirs(
                os.path.dirname(
                    self.model_trainer_config
                    .trained_model_file_path
                ),
                exist_ok=True,
            )

            save_object(
                file_path=self.model_trainer_config
                .trained_model_file_path,
                obj=model_package,
            )

            logging.info(
                f"Saved model package: "
                f"{best_model_name}"
            )

            logging.info(
                f"Saved threshold: "
                f"{calibrated_threshold}"
            )

            # ----------------------------------
            # Explainability
            # ----------------------------------

            (
                shap_path,
                lime_path,
                _,
            ) = self.build_and_save_explainers(
                model=best_model,
                x_train=x_train,
                feature_names=feature_names,
            )

            # ----------------------------------
            # Artifact
            # ----------------------------------

            model_trainer_artifact = (
                ModelTrainerArtifact(
                    trained_model_file_path=(
                        self.model_trainer_config
                        .trained_model_file_path
                    ),
                    shap_explainer_file_path=(
                        shap_path or ""
                    ),
                    lime_explainer_file_path=(
                        lime_path or ""
                    ),
                    best_model_name=best_model_name,
                    metric_artifact=eval_report,
                )
            )

            logging.info(
                "Model Trainer Artifact successfully generated."
            )

            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)