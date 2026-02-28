import os
import mlflow
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.constants import SCHEMA_FILE_PATH
from src.components.data_validation import DataValidation


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_validation(self) -> None:
        logging.info("Starting Data Validation pipeline.")
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_transformation_config = config.get_data_transformation_config()
            mlops_config = config.get_mlops_config()

            os.environ["MLFLOW_TRACKING_URI"] = mlops_config.mlflow_uri
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlops_config.dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

            with mlflow.start_run(run_name="Data_Validation_Run"):
                data_validation = DataValidation(
                    config=data_validation_config,
                    data_transformation_config=data_transformation_config,
                    schema_path=SCHEMA_FILE_PATH
                )
                status, train_df, valid_df, test_df = data_validation.validate_columns()

                mlflow.log_param("data_validation_status", status)
                mlflow.log_param("data_validation_root_dir", str(data_validation_config.root_dir))
                mlflow.log_param("data_validation_local_file", str(data_validation_config.local_file))

                if status:
                    if os.path.exists(str(data_validation_config.validation_status_path)):
                        mlflow.log_artifact(local_path=str(data_validation_config.validation_status_path), artifact_path="data_validation_artifacts")

                    for path in [
                        data_transformation_config.train_data_path,
                        data_transformation_config.validation_data_path,
                        data_transformation_config.test_data_path,
                    ]:
                        if os.path.exists(str(path)):
                            mlflow.log_artifact(local_path=str(path), artifact_path="data_split_artifacts")
                        else:
                            logging.warning(f"Data split artifact not found: {path}")
                else:
                    logging.warning("Data validation failed, no data splits or status file logged as artifacts.")

            logging.info("Data Validation pipeline finished.")
        except Exception as e:
            logging.error(f"Data Validation pipeline failed: {e}")
            raise e
