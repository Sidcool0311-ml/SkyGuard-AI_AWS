import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

from skyguard.constant.training_pipeline import SCHEMA_FILE_PATH
from skyguard.entity.config_entity import DataValidationConfig
from skyguard.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging
from skyguard.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=self.data_validation_config.schema_file_path)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads CSV file into a pandas DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates if the dataframe contains the required number of columns defined in schema.
        """
        try:
            number_of_columns = len(self._schema_config.get("columns", []))
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe column count: {len(dataframe.columns)}")

            # If dataframe contains an unnamed index column, ignore it in comparison
            actual_cols = [c for c in dataframe.columns if not c.startswith("Unnamed")]
            if len(actual_cols) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e, sys)

    def is_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates whether all mandatory columns exist in the dataframe.
        """
        try:
            dataframe_columns = list(dataframe.columns)
            missing_numerical_columns = []
            missing_categorical_columns = []

            # Check numerical columns
            for column in self._schema_config.get("numerical_columns", []):
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            # Check categorical columns
            for column in self._schema_config.get("categorical_columns", []):
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")

            # Check mandatory columns if defined
            mandatory_columns = self._schema_config.get("mandatory_columns", [])
            missing_mandatory = [col for col in mandatory_columns if col not in dataframe_columns]
            if len(missing_mandatory) > 0:
                logging.info(f"Missing mandatory columns: {missing_mandatory}")
                return False

            return len(missing_numerical_columns) == 0 and len(missing_categorical_columns) == 0
        except Exception as e:
            raise CustomException(e, sys)

    def validate_data_types(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates that numerical sensor features can be safely converted to numeric types.
        """
        try:
            numerical_columns = self._schema_config.get("numerical_columns", [])
            for col in numerical_columns:
                if col in dataframe.columns:
                    converted = pd.to_numeric(dataframe[col], errors="coerce")
                    null_count = converted.isnull().sum()
                    if null_count > 0.1 * len(dataframe):
                        logging.warning(
                            f"Column {col} has high invalid/non-numeric values: {null_count}/{len(dataframe)}"
                        )
                        return False
            return True
        except Exception as e:
            raise CustomException(e, sys)

    def validate_domain_value_ranges(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates that weather sensor readings fall within plausible physical atmospheric ranges.
        """
        try:
            domain_ranges = self._schema_config.get("domain_value_range", {})
            for col, bounds in domain_ranges.items():
                if col in dataframe.columns:
                    numeric_col = pd.to_numeric(dataframe[col], errors="coerce")
                    min_val = bounds.get("min")
                    max_val = bounds.get("max")

                    if min_val is not None:
                        violations_min = (numeric_col < min_val).sum()
                        if violations_min > 0:
                            logging.warning(f"Feature {col} has {violations_min} readings below min bound {min_val}")

                    if max_val is not None:
                        violations_max = (numeric_col > max_val).sum()
                        if violations_max > 0:
                            logging.warning(f"Feature {col} has {violations_max} readings above max bound {max_val}")

            return True
        except Exception as e:
            raise CustomException(e, sys)

    def detect_dataset_drift(
        self,
        base_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.05
    ) -> bool:
        """
        Calculates feature distribution drift between train (base) and test (current) sets
        using the two-sample Kolmogorov-Smirnov (KS) test.
        """
        try:
            status = True
            report = {
                "overall_drift_status": False,
                "drift_significance_threshold": threshold,
                "metrics": {}
            }
            drifted_features = []

            numerical_columns = self._schema_config.get("numerical_columns", [])

            for col in numerical_columns:
                if col in base_df.columns and col in current_df.columns:
                    base_data = pd.to_numeric(base_df[col], errors="coerce").dropna()
                    curr_data = pd.to_numeric(current_df[col], errors="coerce").dropna()

                    ks_result = ks_2samp(base_data, curr_data)
                    p_value = float(ks_result.pvalue)
                    statistic = float(ks_result.statistic)

                    # Null hypothesis: two distributions are identical.
                    # If p_value < threshold, reject null hypothesis => drift detected.
                    drift_detected = p_value < threshold
                    if drift_detected:
                        status = False
                        drifted_features.append(col)

                    report["metrics"][col] = {
                        "p_value": p_value,
                        "statistic": statistic,
                        "drift_detected": drift_detected
                    }

            report["overall_drift_status"] = not status
            report["drifted_features"] = drifted_features

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            write_yaml_file(file_path=drift_report_file_path, content=report, replace=True)
            logging.info(f"Drift report generated at: {drift_report_file_path}")

            return status
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Runs the complete data validation pipeline:
        1. Reads train and test data
        2. Validates column counts and existence
        3. Validates data types and domain bounds
        4. Detects dataset drift
        5. Exports validated or invalid datasets
        6. Produces DataValidationArtifact
        """
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info("Initiating Data Validation Component")
            logging.info(f"Reading train dataset: {train_file_path}")
            logging.info(f"Reading test dataset: {test_file_path}")

            train_dataframe = self.read_data(train_file_path)
            test_dataframe = self.read_data(test_file_path)

            validation_errors = []

            # 1. Validate number of columns
            train_col_len_status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not train_col_len_status:
                validation_errors.append("Train dataframe column count does not match schema.")

            test_col_len_status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not test_col_len_status:
                validation_errors.append("Test dataframe column count does not match schema.")

            # 2. Validate mandatory columns existence
            train_cols_exist = self.is_columns_exist(dataframe=train_dataframe)
            if not train_cols_exist:
                validation_errors.append("Train dataframe is missing mandatory schema columns.")

            test_cols_exist = self.is_columns_exist(dataframe=test_dataframe)
            if not test_cols_exist:
                validation_errors.append("Test dataframe is missing mandatory schema columns.")

            # 3. Validate numerical datatypes
            train_types_valid = self.validate_data_types(dataframe=train_dataframe)
            test_types_valid = self.validate_data_types(dataframe=test_dataframe)
            if not (train_types_valid and test_types_valid):
                validation_errors.append("Non-numeric/invalid values found in sensor numerical features.")

            # 4. Validate domain bounds
            self.validate_domain_value_ranges(dataframe=train_dataframe)
            self.validate_domain_value_ranges(dataframe=test_dataframe)

            # 5. Check for data drift between train and test sets
            drift_status = self.detect_dataset_drift(base_df=train_dataframe, current_df=test_dataframe)
            if not drift_status:
                logging.warning("Data drift detected between training and testing sets.")

            # Determine overall validation status
            validation_status = len(validation_errors) == 0

            # 6. Save datasets to validated or invalid directories
            if validation_status:
                logging.info("Data validation succeeded. Saving validated datasets.")
                os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
                train_dataframe.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.valid_test_file_path, index=False)
                message = "Data validation completed successfully."
            else:
                logging.error(f"Data validation failed with errors: {validation_errors}")
                os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)
                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                message = "; ".join(validation_errors)

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path if validation_status else None,
                valid_test_file_path=self.data_validation_config.valid_test_file_path if validation_status else None,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path if not validation_status else None,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path if not validation_status else None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(f"Data Validation Artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)

