import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from skyguard.constant.training_pipeline import (
    TARGET_COLUMN,
    TEMPERATURE_COLUMN,
    HUMIDITY_COLUMN,
    PRESSURE_COLUMN,
    STATION_COLUMN,
    TIME_COLUMN,
)
from skyguard.entity.config_entity import DataTransformationConfig
from skyguard.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging
from skyguard.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object,
)


class SensorFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Scikit-Learn Transformer for meteorological sensor telemetry:
    1. Time-based cyclical feature engineering (sin/cos for hour, month, day-of-year)
    2. Rolling window statistics per station (3-hour and 6-hour mean and std)
    3. Rate-of-change features (delta and absolute delta per sensor per station)
    4. Cross-parameter physical consistency features (Discomfort Index, Dew Point Depression,
       thermodynamic Temp/Pressure ratio, and Delta interaction)
    """

    def __init__(self, time_col: str = TIME_COLUMN, station_col: str = STATION_COLUMN):
        self.time_col = time_col
        self.station_col = station_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            if not isinstance(X, pd.DataFrame):
                df = pd.DataFrame(X)
            else:
                df = X.copy()

            # Ensure chronological ordering per station if time and station columns exist
            if self.time_col in df.columns:
                df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
                if self.station_col in df.columns:
                    df = df.sort_values(by=[self.station_col, self.time_col]).reset_index(drop=True)
                else:
                    df = df.sort_values(by=self.time_col).reset_index(drop=True)

                # 1. Cyclical time encoding (hour, month, day of year)
                hour = df[self.time_col].dt.hour.fillna(0)
                month = df[self.time_col].dt.month.fillna(1)
                day_of_year = df[self.time_col].dt.dayofyear.fillna(1)

                df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
                df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
                df["sin_month"] = np.sin(2.0 * np.pi * (month - 1.0) / 12.0)
                df["cos_month"] = np.cos(2.0 * np.pi * (month - 1.0) / 12.0)
                df["sin_day_of_year"] = np.sin(2.0 * np.pi * (day_of_year - 1.0) / 365.25)
                df["cos_day_of_year"] = np.cos(2.0 * np.pi * (day_of_year - 1.0) / 365.25)

            # Core meteorological sensors
            sensor_cols = [TEMPERATURE_COLUMN, HUMIDITY_COLUMN, PRESSURE_COLUMN]
            for col in sensor_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if self.station_col in df.columns:
                        df[col] = df.groupby(self.station_col)[col].transform(lambda s: s.ffill().bfill())
                    df[col] = df[col].fillna(df[col].median() if not df[col].dropna().empty else 0.0)

            # 2. Rolling window statistics (3-hour and 6-hour mean & std per station)
            if self.station_col in df.columns:
                grouped = df.groupby(self.station_col)
                for col in sensor_cols:
                    if col in df.columns:
                        df[f"{col}_rolling_mean_3h"] = grouped[col].transform(lambda s: s.rolling(3, min_periods=1).mean())
                        df[f"{col}_rolling_std_3h"] = grouped[col].transform(lambda s: s.rolling(3, min_periods=1).std()).fillna(0.0)
                        df[f"{col}_rolling_mean_6h"] = grouped[col].transform(lambda s: s.rolling(6, min_periods=1).mean())
                        df[f"{col}_rolling_std_6h"] = grouped[col].transform(lambda s: s.rolling(6, min_periods=1).std()).fillna(0.0)
            else:
                for col in sensor_cols:
                    if col in df.columns:
                        df[f"{col}_rolling_mean_3h"] = df[col].rolling(3, min_periods=1).mean()
                        df[f"{col}_rolling_std_3h"] = df[col].rolling(3, min_periods=1).std().fillna(0.0)
                        df[f"{col}_rolling_mean_6h"] = df[col].rolling(6, min_periods=1).mean()
                        df[f"{col}_rolling_std_6h"] = df[col].rolling(6, min_periods=1).std().fillna(0.0)

            # 3. Rate-of-change features (delta and abs_delta per sensor per station)
            if self.station_col in df.columns:
                grouped = df.groupby(self.station_col)
                for col in sensor_cols:
                    if col in df.columns:
                        delta = grouped[col].diff().fillna(0.0)
                        df[f"delta_{col}"] = delta
                        df[f"abs_delta_{col}"] = delta.abs()
            else:
                for col in sensor_cols:
                    if col in df.columns:
                        delta = df[col].diff().fillna(0.0)
                        df[f"delta_{col}"] = delta
                        df[f"abs_delta_{col}"] = delta.abs()

            # 4. Cross-parameter consistency features
            temp = df.get(TEMPERATURE_COLUMN, pd.Series(0.0, index=df.index))
            rh = df.get(HUMIDITY_COLUMN, pd.Series(50.0, index=df.index))
            pres = df.get(PRESSURE_COLUMN, pd.Series(1013.25, index=df.index))

            # Thom's Discomfort Index (DI): combines thermal and hygrometric impact
            df["discomfort_index"] = temp - 0.55 * (1.0 - 0.01 * rh) * (temp - 14.5)

            # Dew Point Depression approximation (T - T_dew ~ (100 - RH) / 5)
            df["dew_point_depression"] = (100.0 - rh) / 5.0

            # Thermodynamic Temperature-Pressure Ratio ((T + 273.15) / Surface Pressure)
            df["temp_pressure_ratio"] = (temp + 273.15) / (pres.replace(0, 1013.25) + 1e-5)

            # Multivariate spike consistency interaction
            delta_t = df.get(f"delta_{TEMPERATURE_COLUMN}", pd.Series(0.0, index=df.index))
            delta_p = df.get(f"delta_{PRESSURE_COLUMN}", pd.Series(0.0, index=df.index))
            df["delta_temp_pressure_interaction"] = delta_t * delta_p

            # Drop original time column as cyclical numeric features have replaced it
            if self.time_col in df.columns:
                df = df.drop(columns=[self.time_col])

            return df
        except Exception as e:
            raise CustomException(e, sys)


class SyntheticAnomalyInjector:
    """
    Injects synthetic atmospheric telemetry anomalies to establish ground-truth evaluation benchmarks:
    - Target Anomaly Ratio: ~10% (8% - 12%)
    - Injected Types:
      1. Impulse Spikes / Sudden Jumps (~3%)
      2. Sensor Freezing / Flatlines (~3%)
      3. Thermodynamic / Physical Inconsistencies (~2.5%)
      4. Out-of-Bounds Extreme Telemetry (~1.5%)
    """

    def __init__(self, anomaly_rate: float = 0.10, random_state: int = 42):
        self.anomaly_rate = anomaly_rate
        self.random_state = random_state

    def inject_anomalies(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()
        
        # Force sensor columns to float so decimal spike values fit
        for col in [TEMPERATURE_COLUMN, HUMIDITY_COLUMN, PRESSURE_COLUMN]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        np.random.seed(self.random_state)
        n_rows = len(df)
        total_anomalies = max(1, int(n_rows * self.anomaly_rate))

        is_anomaly = np.zeros(n_rows, dtype=int)

        # Disjoint random allocation of anomaly archetypes
        permuted_indices = np.random.permutation(n_rows)
        n_spike = int(0.30 * total_anomalies)
        n_freeze = int(0.30 * total_anomalies)
        n_inconsistent = int(0.25 * total_anomalies)
        n_out_of_bounds = total_anomalies - (n_spike + n_freeze + n_inconsistent)

        spike_indices = permuted_indices[:n_spike]
        freeze_start_indices = permuted_indices[n_spike : n_spike + n_freeze]
        inconsistent_indices = permuted_indices[n_spike + n_freeze : n_spike + n_freeze + n_inconsistent]
        bounds_indices = permuted_indices[n_spike + n_freeze + n_inconsistent : total_anomalies]

        # 1. Impulse Spikes
        for idx in spike_indices:
            sensor_choice = np.random.choice(["temp", "humidity", "pressure"])
            if sensor_choice == "temp" and TEMPERATURE_COLUMN in df.columns:
                jump = np.random.choice([-1, 1]) * np.random.uniform(12.0, 22.0)
                df.at[idx, TEMPERATURE_COLUMN] = float(df.at[idx, TEMPERATURE_COLUMN]) + jump
            elif sensor_choice == "humidity" and HUMIDITY_COLUMN in df.columns:
                jump = np.random.choice([-1, 1]) * np.random.uniform(40.0, 60.0)
                df.at[idx, HUMIDITY_COLUMN] = np.clip(float(df.at[idx, HUMIDITY_COLUMN]) + jump, 5.0, 95.0)
            elif sensor_choice == "pressure" and PRESSURE_COLUMN in df.columns:
                jump = np.random.choice([-1, 1]) * np.random.uniform(30.0, 60.0)
                df.at[idx, PRESSURE_COLUMN] = float(df.at[idx, PRESSURE_COLUMN]) + jump
            is_anomaly[idx] = 1

        # 2. Sensor Freezing / Flatlines (repeat reading across 4-6 consecutive rows)
        for start_idx in freeze_start_indices:
            freeze_len = np.random.randint(4, 7)
            end_idx = min(n_rows, start_idx + freeze_len)
            target_sensor = np.random.choice([TEMPERATURE_COLUMN, HUMIDITY_COLUMN, PRESSURE_COLUMN])
            if target_sensor in df.columns:
                static_val = df.at[start_idx, target_sensor]
                for r in range(start_idx, end_idx):
                    df.at[r, target_sensor] = static_val
                    is_anomaly[r] = 1

        # 3. Thermodynamic / Physical Inconsistency
        for idx in inconsistent_indices:
            if TEMPERATURE_COLUMN in df.columns:
                df.at[idx, TEMPERATURE_COLUMN] = float(np.random.uniform(42.0, 48.0))
            if HUMIDITY_COLUMN in df.columns:
                df.at[idx, HUMIDITY_COLUMN] = float(np.random.uniform(95.0, 99.0))
            if PRESSURE_COLUMN in df.columns:
                df.at[idx, PRESSURE_COLUMN] = float(np.random.uniform(810.0, 850.0))
            is_anomaly[idx] = 1

        # 4. Out-of-Bounds Extreme Violations
        for idx in bounds_indices:
            bound_type = np.random.choice(["temp_high", "temp_low", "humidity_high", "humidity_low"])
            if bound_type == "temp_high" and TEMPERATURE_COLUMN in df.columns:
                df.at[idx, TEMPERATURE_COLUMN] = float(np.random.uniform(58.0, 68.0))
            elif bound_type == "temp_low" and TEMPERATURE_COLUMN in df.columns:
                df.at[idx, TEMPERATURE_COLUMN] = float(np.random.uniform(-55.0, -42.0))
            elif bound_type == "humidity_high" and HUMIDITY_COLUMN in df.columns:
                df.at[idx, HUMIDITY_COLUMN] = float(np.random.uniform(105.0, 125.0))
            elif bound_type == "humidity_low" and HUMIDITY_COLUMN in df.columns:
                df.at[idx, HUMIDITY_COLUMN] = float(np.random.uniform(-20.0, -5.0))
            is_anomaly[idx] = 1

        df[TARGET_COLUMN] = is_anomaly
        return df


class DataTransformation:
    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig,
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
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

    def get_data_transformer_object(self, feature_engineered_df: pd.DataFrame) -> Pipeline:
        """
        Builds the scikit-learn preprocessing pipeline:
        - Numerical features: SimpleImputer(median) -> RobustScaler()
        - Categorical features: SimpleImputer(most_frequent) -> OneHotEncoder()
        - Full Pipeline incorporates the SensorFeatureEngineer transformer
        """
        try:
            logging.info("Initiating get_data_transformer_object")

            # Identify columns post feature-engineering
            candidate_cols = list(feature_engineered_df.columns)
            if TARGET_COLUMN in candidate_cols:
                candidate_cols.remove(TARGET_COLUMN)
            if TIME_COLUMN in candidate_cols:
                candidate_cols.remove(TIME_COLUMN)

            categorical_cols = [STATION_COLUMN] if STATION_COLUMN in candidate_cols else []
            numerical_cols = [c for c in candidate_cols if c not in categorical_cols]

            logging.info(f"Identified {len(numerical_cols)} numerical features: {numerical_cols}")
            logging.info(f"Identified categorical features: {categorical_cols}")

            # Numerical Pipeline: Median Imputer + RobustScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]
            )

            # Categorical Pipeline: Most Frequent Imputer + OneHotEncoder
            try:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", ohe),
                ]
            )

            # Combine pipelines via ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_cols),
                    ("cat_pipeline", cat_pipeline, categorical_cols),
                ],
                remainder="drop",
            )

            # Complete end-to-end pipeline: Raw DataFrame -> Engineered Features -> Scaled/Encoded Arrays
            full_pipeline = Pipeline(
                steps=[
                    ("feature_engineer", SensorFeatureEngineer()),
                    ("preprocessor", preprocessor),
                ]
            )

            logging.info("Created full preprocessing pipeline successfully")
            return full_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Executes complete data transformation:
        1. Reads validated train and test datasets.
        2. Injects synthetic anomalies (~10% rate) to establish ground-truth is_anomaly benchmark.
        3. Fits complete feature engineering + scaling pipeline on training data.
        4. Transforms training and testing datasets using the exact fitted object.
        5. Saves fitted preprocessor.pkl using save_object.
        6. Saves transformed arrays with target labels as train.npy and test.npy.
        7. Returns DataTransformationArtifact.
        """
        try:
            logging.info("Starting Data Transformation Component")

            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path

            if not train_file_path or not os.path.exists(train_file_path):
                raise Exception(f"Valid train file path not found: {train_file_path}")
            if not test_file_path or not os.path.exists(test_file_path):
                raise Exception(f"Valid test file path not found: {test_file_path}")

            logging.info(f"Loading validated train dataset from: {train_file_path}")
            logging.info(f"Loading validated test dataset from: {test_file_path}")

            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)
            # Sort FIRST so row order matches what SensorFeatureEngineer will produce
            sort_cols = [STATION_COLUMN, TIME_COLUMN]
            train_df = train_df.sort_values(by=sort_cols).reset_index(drop=True)
            test_df = test_df.sort_values(by=sort_cols).reset_index(drop=True)

            # Synthetic Anomaly Injection (ensuring ~10% ground-truth labels if not already injected)
            if TARGET_COLUMN not in train_df.columns:
                logging.info("Injecting synthetic anomalies into train and test datasets (~10% rate: spikes, flatlines, physics violations, bounds)")
                injector_train = SyntheticAnomalyInjector(anomaly_rate=0.10, random_state=42)
                injector_test = SyntheticAnomalyInjector(anomaly_rate=0.10, random_state=101)
                train_df = injector_train.inject_anomalies(train_df)
                test_df = injector_test.inject_anomalies(test_df)

            train_anom_cnt = int(train_df[TARGET_COLUMN].sum())
            test_anom_cnt = int(test_df[TARGET_COLUMN].sum())
            logging.info(
                f"Train ground-truth anomalies: {train_anom_cnt}/{len(train_df)} ({train_df[TARGET_COLUMN].mean()*100:.2f}%)"
            )
            logging.info(
                f"Test ground-truth anomalies: {test_anom_cnt}/{len(test_df)} ({test_df[TARGET_COLUMN].mean()*100:.2f}%)"
            )

            # Separate target labels from feature dataframes
            input_feature_train_df = train_df.drop([TARGET_COLUMN], axis=1)
            target_feature_train_arr = train_df[TARGET_COLUMN].values.astype(int)

            input_feature_test_df = test_df.drop([TARGET_COLUMN], axis=1)
            target_feature_test_arr = test_df[TARGET_COLUMN].values.astype(int)

            # Dry-run feature engineering to discover post-engineering schema columns
            engineer_preview = SensorFeatureEngineer().transform(input_feature_train_df.head(50))
            pipeline_obj = self.get_data_transformer_object(feature_engineered_df=engineer_preview)

            logging.info("Fitting preprocessing pipeline on training dataset")
            input_feature_train_arr = pipeline_obj.fit_transform(input_feature_train_df)

            logging.info("Transforming testing dataset using fitted pipeline")
            input_feature_test_arr = pipeline_obj.transform(input_feature_test_df)

            # Assemble final transformed arrays with target column as the last column
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]


            logging.info(f"Transformed train array shape: {train_arr.shape}")
            logging.info(f"Transformed test array shape: {test_arr.shape}")

            # Save transformed numpy arrays
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            logging.info(
                f"Saved transformed train array to {self.data_transformation_config.transformed_train_file_path}"
            )

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            logging.info(
                f"Saved transformed test array to {self.data_transformation_config.transformed_test_file_path}"
            )

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=pipeline_obj,
            )
            logging.info(
                f"Saved fitted preprocessor object to {self.data_transformation_config.transformed_object_file_path}"
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info(f"Data Transformation Artifact created: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys)

