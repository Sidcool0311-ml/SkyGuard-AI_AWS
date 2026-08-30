import pandas as pd
import os
import sys
from skyguard.entity.config_entity import DataIngestionConfig , TrainingPipelineConfig
from skyguard.entity.artifact_entity import DataIngestionArtifact
from skyguard.components.data_ingestion import DataIngestion
from  skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging


if __name__=="__main__":
    try:
        logging.info("starting the data ingestion pipelines")
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        dataingestionartifact=(data_ingestion.initiate_data_ingestion())
        logging.info("data ingestion completed")
        print(dataingestionartifact)
    except Exception as e:
        raise CustomException(e,sys)
