import os
import mlflow
import pandas as pd
import numpy as np
import json
from src.utils.common import *
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.data_preprocessing import DataPreprocessing
from src.components.data_validation import DataValidation
from src.pipeline.stage_3_data_transformation import DataTransformationTrainingPipeline


class DataPreprocessingTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_preprocessing(self):
        logging.info("Starting Data Preprocessing pipeline.")
        try:
            config = ConfigurationManager()
            data_preprocessed_config = config.get_data_preprocessed_config()
            data_validation_config = config.get_data_validation_config()
            data_transformation_config = config.get_data_transformation_config()
            mlops_config = config.get_mlops_config()
            schema_path = config.schema_path

            os.environ["MLFLOW_TRACKING_URI"] = mlops_config.mlflow_uri
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlops_config.dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

            with mlflow.start_run(run_name="Data_Preprocessing_Run"):
                (
                    X_train_meta_np, X_val_meta_np, X_test_meta_np,
                    _, _, _,
                    _, _, _,
                    _,
                    y_train, y_val, y_test
                ) = DataTransformationTrainingPipeline().initiate_data_transformation()

                meta_columns_path = os.path.join(data_transformation_config.root_dir, 'meta_columns.json')
                with open(meta_columns_path, 'r') as f:
                    meta_columns = json.load(f)

                X_train_meta_df = pd.DataFrame(X_train_meta_np, columns=meta_columns)
                X_val_meta_df = pd.DataFrame(X_val_meta_np, columns=meta_columns)
                X_test_meta_df = pd.DataFrame(X_test_meta_np, columns=meta_columns)

                data_validation = DataValidation(
                    config=data_validation_config,
                    data_transformation_config=data_transformation_config,
                    schema_path=schema_path
                )

                data_preprocessed = DataPreprocessing(
                    config=data_preprocessed_config,
                    data_validation=data_validation,
                    data_transformation_config=data_transformation_config
                )

                data_preprocessed.initiate_data_preprocess(
                    X_train_meta=X_train_meta_df,
                    X_val_meta=X_val_meta_df,
                    X_test_meta=X_test_meta_df,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test
                )

                mlflow.log_param("data_preprocessing_root_dir", str(data_preprocessed_config.root_dir))
                mlflow.log_param("numeric_features", str(data_preprocessed_config.numeric_features))
                mlflow.log_param("categorical_features", str(data_preprocessed_config.categorical_features))
                mlflow.log_param("imputation_strategy_numeric", data_preprocessed_config.imputation_strategy_numeric)
                mlflow.log_param("imputation_strategy_categorical", data_preprocessed_config.imputation_strategy_categorical)
                mlflow.log_param("scaler_type", data_preprocessed_config.scaler_type)
                mlflow.log_param("encoder_type", data_preprocessed_config.encoder_type)

                structured_features_dir = os.path.join(data_preprocessed_config.root_dir, 'structured_features')
                combined_features_dir = os.path.join(data_preprocessed_config.root_dir, 'combined_features')

                if os.path.exists(structured_features_dir):
                    mlflow.log_artifact(local_path=structured_features_dir, artifact_path="structured_features_artifacts")
                else:
                    logging.warning(f"Structured features directory not found: {structured_features_dir}")

                if os.path.exists(combined_features_dir):
                    mlflow.log_artifact(local_path=combined_features_dir, artifact_path="combined_features_artifacts")
                else:
                    logging.warning(f"Combined features directory not found: {combined_features_dir}")

                for target_file in ['y_train.npy', 'y_val.npy', 'y_test.npy']:
                    target_path = os.path.join(data_preprocessed_config.root_dir, target_file)
                    if os.path.exists(target_path):
                        mlflow.log_artifact(local_path=target_path, artifact_path="target_arrays")
                    else:
                        logging.warning(f"Target array file not found: {target_path}")

            logging.info("Data Preprocessing pipeline finished.")
        except Exception as e:
            logging.error(f"Data Preprocessing pipeline failed: {e}")
            raise e
