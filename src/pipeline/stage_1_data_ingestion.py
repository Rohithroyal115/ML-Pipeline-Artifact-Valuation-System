import os
import mlflow
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def initiate_data_ingestion(self) -> None:
        logging.info("Starting Data Ingestion pipeline.")
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            mlops_config = config.get_mlops_config()

            os.environ["MLFLOW_TRACKING_URI"] = mlops_config.mlflow_uri
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlops_config.dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

            with mlflow.start_run(run_name="Data_Ingestion_Run"):
                data_ingestion = DataIngestion(config=data_ingestion_config)
                df = data_ingestion.read_data()

                mlflow.log_param("data_ingestion_root_dir", str(data_ingestion_config.root_dir))
                mlflow.log_param("data_ingestion_local_file", str(data_ingestion_config.local_file))

                artifact_path = os.path.join(data_ingestion_config.root_dir, 'data.csv')
                mlflow.log_artifact(local_path=artifact_path, artifact_path="data_ingestion_artifacts")

            logging.info("Data Ingestion pipeline finished.")
        except Exception as e:
            logging.error(f"Data Ingestion pipeline failed: {e}")
            raise e
