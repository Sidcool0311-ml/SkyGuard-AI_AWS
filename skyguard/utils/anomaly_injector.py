import pandas as pd
import numpy as np
import os
import sys
import glob

from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging
from skyguard.constant.training_pipeline import TEMPERATURE_COLUMN,HUMIDITY_COLUMN,STATION_COLUMN,PRESSURE_COLUMN

SENSOR_COLUMNS=[TEMPERATURE_COLUMN,PRESSURE_COLUMN,STATION_COLUMN,HUMIDITY_COLUMN]

class AnomalyInjector:
    def __init__(self,seed:int=42):
        self.rng=np.random.default_rng(seed)
    def inject_spikes(self,df:pd.DataFrame,fraction:float=0.02):
        try:
            n=int(len(df)*fraction) ##randomly selecting a dataset of n*0.02 datapoints
            idx=self.rng.choice(df.index,size=n,replace=False)
            for i in idx:
                col=self.rng.choice(SENSOR_COLUMNS)
                if col==TEMPERATURE_COLUMN:
                    df.loc[i,col]=self.rng.uniform(50,70)
                elif col==PRESSURE_COLUMN:
                    df.loc[i,col]=self.rng.choice([self.rng.uniform(700, 850), self.rng.uniform(1100, 1200)])
                else:
                    df.loc[i, col] = self.rng.choice([0, self.rng.uniform(150, 200)])
            df.loc[i,"is_anomaly"]=1
            df.loc[i,"anomaly_type"]="spike"
            return df
            
        except Exception as e:
            raise CustomException(e,sys)

    def inject_frozen_values(self,df:pd.DataFrame,fraction:float=0.02):
        try:
            stations = df[STATION_COLUMN].unique()
            n_windows = max(1, int(len(df) * fraction / 10))
            for _ in range(n_windows):
                station = self.rng.choice(stations)
                station_idx = df[df[STATION_COLUMN] == station].index
                if len(station_idx) < 15:
                    continue
                start = int(self.rng.integers(0, len(station_idx) - 15))
                window_len = int(self.rng.integers(5, 15))
                window = station_idx[start:start + window_len]
                col = self.rng.choice(SENSOR_COLUMNS)
                frozen_value = df.loc[window[0], col]
                df.loc[window, col] = frozen_value
                df.loc[window, "is_anomaly"] = 1
                df.loc[window, "anomaly_type"] = "frozen"
            return df
   
        except Exception as e:
            raise CustomException(e,sys)

    def inject_dropouts(self,df:pd.DataFrame,fraction:float=0.02):
        try:
            n=int(len(df)*fraction)
            idx=self.rng.choice(df.index,size=n,replace=False)
            for i in idx:
                col=self.rng.choice(SENSOR_COLUMNS)
                df.loc[i,col]=np.nan
                df.loc[i,"is_anomaly"]=1
                df.loc[i,"anomaly_type"]="dropout"
            return df

        except Exception as e:
            raise CustomException(e,sys)
    
    def inject_drift(self, df: pd.DataFrame, fraction: float = 0.02):
        try:
            stations = df[STATION_COLUMN].unique()
            n_stations_affected = max(1, int(len(stations) * fraction * 10))
            affected = self.rng.choice(stations, size=min(n_stations_affected, len(stations)), replace=False)
            for station in affected:
                station_idx = df[df[STATION_COLUMN] == station].index
                if len(station_idx) < 20:
                    continue
                start_pos = self.rng.integers(0, len(station_idx) // 2)
                drift_idx = station_idx[start_pos:]
                col = self.rng.choice(SENSOR_COLUMNS)
                drift_amount = np.linspace(0, self.rng.uniform(5, 15), len(drift_idx))
                df.loc[drift_idx, col] = df.loc[drift_idx, col] + drift_amount
                df.loc[drift_idx, "is_anomaly"] = 1
                df.loc[drift_idx, "anomaly_type"] = "drift"
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def inject_cross_param_inconsistency(self,df:pd.DataFrame,fraction:float=0.02):
        try:
            n = int(len(df) * fraction)
            idx = self.rng.choice(df.index, size=n, replace=False)
            for i in idx:
                df.loc[i, TEMPERATURE_COLUMN] = self.rng.uniform(40, 48)
                df.loc[i, HUMIDITY_COLUMN] = self.rng.uniform(90, 100)
                df.loc[i, PRESSURE_COLUMN] = self.rng.uniform(750, 820)
                df.loc[i, "is_anomaly"] = 1
                df.loc[i, "anomaly_type"] = "cross_param"
            return df
        except Exception as e:
            raise CustomException(e, sys)

    
    def inject_all(self,df:pd.DataFrame):
        try:
            df = df.copy()
            df["is_anomaly"] = 0
            df["anomaly_type"] = "normal"

            df = self.inject_spikes(df)
            df = self.inject_frozen_values(df)
            df = self.inject_dropouts(df)
            df = self.inject_drift(df)
            df = self.inject_cross_param_inconsistency(df)

            logging.info(f"Anomaly injection complete. Distribution:\n{df['anomaly_type'].value_counts()}")
            return df
        except Exception as e:
            raise CustomException(e,sys)

if __name__ == "__main__":
    try:
        train_path = os.path.join("Artifacts", "data_ingestion", "ingested", "train.csv")
        test_path = os.path.join("Artifacts", "data_ingestion", "ingested", "test.csv")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        for col in SENSOR_COLUMNS:
              train_df[col] = pd.to_numeric(train_df[col], errors="coerce").astype(float)
              test_df[col] = pd.to_numeric(test_df[col], errors="coerce").astype(float)

        injector = AnomalyInjector()
        train_injected = injector.inject_all(train_df)
        test_injected = injector.inject_all(test_df)

        out_dir = os.path.join("Artifacts", "data_ingestion", "injected")
        os.makedirs(out_dir, exist_ok=True)

        train_injected.to_csv(os.path.join(out_dir, "train_injected.csv"), index=False)
        test_injected.to_csv(os.path.join(out_dir, "test_injected.csv"), index=False)

        print(train_injected["anomaly_type"].value_counts())
        print(test_injected["anomaly_type"].value_counts())

    except Exception as e:
        raise CustomException(e, sys)