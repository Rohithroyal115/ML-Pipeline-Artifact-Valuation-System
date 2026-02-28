import os
import mlflow
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src.entity import ModelTrainerConfig

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self) -> None:
        logging.info("Starting model training pipeline.")
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            mlops_config = config.get_mlops_config()

            os.environ["MLFLOW_TRACKING_URI"] = mlops_config.mlflow_uri
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlops_config.dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

            if mlflow.active_run() is not None:
                mlflow.end_run()

            with mlflow.start_run(run_name="Model_Training_Run"):
                model_trainer = ModelTrainer(config=model_trainer_config)
                model_trainer.train_model()

            logging.info("Model training pipeline completed successfully.")

        except Exception as e:
            logging.error(f"Model training pipeline failed: {e}")
            raise e
