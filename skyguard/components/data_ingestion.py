import pandas as pd
import numpy as np
import os
import pymongo
import certifi
import sys

from skyguard.entity.config_entity import DataIngestionConfig
from skyguard.entity.artifact_entity import DataIngestionArtifact

from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()
Mongo_DB_URL=os.getenv("MONGO_DB_URL")
ca=certifi.where()


class DataIngestion:
    def __init__(self,data_ingestion_config=DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)
    def export_collection_as_df(self):
        try:
            database_name=self.data_ingestion_config.database_name

            collecton_name=self.data_ingestion_config.collection_name

            self.mongo_client=pymongo.MongoClient(Mongo_DB_URL)
            collection=self.mongo_client[database_name][collecton_name]
            df=pd.DataFrame(list(collection.find()))

            if "_id" in df.columns:
                df = df.drop("_id", axis=1)

            df.replace({"na":np.nan},inplace=True)

            logging.info("Collected data from MongoDb")
            logging.info(f"pulled{df.shape[0]} rows,{df.shape[1]} columns from MongoDb")

            return df
        
        except Exception as e:
            raise CustomException(e,sys)
    def export_data_to_feature_store(self,dataframe=pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            print("Feature store file path:", feature_store_file_path)
            print("Directory path:", dir_path)

            dataframe.to_csv(feature_store_file_path)
            return dataframe
        
        except Exception as e:
            raise CustomException(e,sys)
    def split_data_as_train_test(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set=train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("performed split on data")
            logging.info("acquired file path for train and test data")
            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)   
            ##saving train data
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False
                )
            ##saving test data
            test_set.to_csv(
               self.data_ingestion_config.testing_file_path,
             index=False
               )
            logging.info("saved train and test datasets")

            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
     
    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_df()

            dataframe = self.export_data_to_feature_store(
                dataframe
            )
            data_ingestion_artifact = (
                self.split_data_as_train_test(
                    dataframe
                )
            )

            logging.info(
                "Data ingestion completed successfully"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys)