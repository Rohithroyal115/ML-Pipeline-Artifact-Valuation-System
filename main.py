'''from dotenv import load_dotenv
load_dotenv()
import os
from src.logging import logging
from src.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_2_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_3_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_4_data_preprocessing import DataPreprocessingTrainingPipeline
from src.pipeline.stage_5_model_trainer import ModelTrainerTrainingPipeline
from src.pipeline.stage_6_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME = "Data Validation Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    data_validation_pipeline = DataValidationTrainingPipeline()
    data_validation_pipeline.initiate_data_validation()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME = "Data Preprocessing Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    data_preprocessing_pipeline = DataPreprocessingTrainingPipeline()
    data_preprocessing_pipeline.initiate_data_preprocessing()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME = "Model Trainer Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    model_trainer_pipeline = ModelTrainerTrainingPipeline()
    model_trainer_pipeline.initiate_model_training()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e


STAGE_NAME = "Model Evaluation Stage"

try:
    logging.info(f"{STAGE_NAME} initiated")
    model_evaluation_pipeline = ModelEvaluationTrainingPipeline()
    model_evaluation_pipeline.initiate_model_evaluation()
    logging.info(f"{STAGE_NAME} completed")
except Exception as e:
    logging.exception(e)
    raise e
'''

from dotenv import load_dotenv
load_dotenv()
import os
from src.logging import logging
from src.pipeline.stage_1_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_2_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_3_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_4_data_preprocessing import DataPreprocessingTrainingPipeline
from src.pipeline.stage_5_model_trainer import ModelTrainerTrainingPipeline
from src.pipeline.stage_6_model_evaluation import ModelEvaluationTrainingPipeline


def run_stage(stage_name, pipeline_class, method_name):
    try:
        logging.info(f"{stage_name} initiated")
        pipeline = pipeline_class()
        getattr(pipeline, method_name)()
        logging.info(f"{stage_name} completed")
    except Exception as e:
        logging.exception(f"{stage_name} failed: {e}")
        raise e


if __name__ == "__main__":
    '''run_stage("Data Ingestion Stage", DataIngestionTrainingPipeline, "initiate_data_ingestion")
    run_stage("Data Validation Stage", DataValidationTrainingPipeline, "initiate_data_validation")
    run_stage("Data Transformation Stage", DataTransformationTrainingPipeline, "initiate_data_transformation")
    run_stage("Data Preprocessing Stage", DataPreprocessingTrainingPipeline, "initiate_data_preprocessing")
    run_stage("Model Trainer Stage", ModelTrainerTrainingPipeline, "initiate_model_training")'''
    run_stage("Model Evaluation Stage", ModelEvaluationTrainingPipeline, "initiate_model_evaluation")
