import os
import mlflow
from src.utils.common import *
from src.logging import logging
from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_evaluation(self) -> None:
        logging.info("Starting model evaluation pipeline.")
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            mlops_config = config.get_mlops_config()

            os.environ["MLFLOW_TRACKING_URI"] = mlops_config.mlflow_uri
            os.environ["MLFLOW_TRACKING_USERNAME"] = mlops_config.dagshub_user
            os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

            if mlflow.active_run() is not None:
                mlflow.end_run()

            with mlflow.start_run(run_name="Model_Evaluation_Run"):
                model_evaluation = ModelEvaluation(config=model_evaluation_config)
                model_evaluation.initiate_model_evaluation()

            logging.info("Model evaluation pipeline completed successfully.")

        except Exception as e:
            logging.error(f"Model evaluation pipeline failed: {e}")
            raise e
