import pandas as pd
import os
import sys
from skyguard.entity.config_entity import (
    DataIngestionConfig,
    TrainingPipelineConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)
from skyguard.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from skyguard.components.data_ingestion import DataIngestion
from skyguard.components.data_validation import DataValidation
from skyguard.components.data_transformation import DataTransformation
from skyguard.components.model_trainer import ModelTrainer
from skyguard.exception.exception import CustomException
from skyguard.logger.logger import logging


if __name__ == "__main__":
    try:
        logging.info("Starting training pipeline")
        training_pipeline_config = TrainingPipelineConfig()

        # 1. Data Ingestion
        logging.info("Starting Data Ingestion")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        print("Data Ingestion Artifact:", data_ingestion_artifact)

        # 2. Data Validation
        logging.info("Starting Data Validation")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config=data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed")
        print("Data Validation Artifact:", data_validation_artifact)

        # 3. Data Transformation
        logging.info("Starting Data Transformation")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation completed")
        print("Data Transformation Artifact:", data_transformation_artifact)

        # 4. Model Training & XAI Setup (Isolation Forest, SHAP, LIME)
        logging.info("Starting Model Training")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Training completed")
        print("Model Trainer Artifact:", model_trainer_artifact)

    except Exception as e:
        raise CustomException(e, sys)
