import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

from src.logging import logging
from src.utils.common import load_bin
from src.entity import ModelEvaluationConfig
from src.config.configuration import ConfigurationManager
from src.entity import MLOpsConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        os.makedirs(self.config.root_dir, exist_ok=True)

    def _evaluate_model(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    def initiate_model_evaluation(self):
        logging.info("Starting model evaluation.")

        config_manager = ConfigurationManager()
        mlops_config = config_manager.get_mlops_config()

        os.environ["MLFLOW_TRACKING_URI"] = mlops_config.mlflow_uri
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlops_config.dagshub_user
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

        y_test = np.load(os.path.join(self.config.test_data_dir, 'y_test.npy'))

        combined_features_dir = os.path.join(self.config.test_data_dir, 'combined_features')
        text_features_dir = getattr(self.config, "text_vectorizer_path", None)
        if text_features_dir is None:
            logging.error("text_vectorizer_path attribute not found in ModelEvaluationConfig.")
            raise AttributeError("text_vectorizer_path attribute missing in config.")

        test_feature_data = {
            "Goal1_tfidf": np.load(os.path.join(combined_features_dir, 'X_test_goal1_tfidf.npy')),
            "Goal1_count": np.load(os.path.join(combined_features_dir, 'X_test_goal1_count.npy')),
            "Goal1_mini": np.load(os.path.join(combined_features_dir, 'X_test_goal1_mini.npy')),
            "Goal1_mpnet": np.load(os.path.join(combined_features_dir, 'X_test_goal1_mpnet.npy')),
            "Goal1_distilbert": np.load(os.path.join(combined_features_dir, 'X_test_goal1_distilbert.npy')),
            "Goal2_tfidf": np.load(os.path.join(text_features_dir, 'X_test_tfidf.npy')),
            "Goal2_count": np.load(os.path.join(text_features_dir, 'X_test_count.npy')),
            "Goal2_mini": np.load(os.path.join(text_features_dir, 'X_test_mini.npy')),
            "Goal2_mpnet": np.load(os.path.join(text_features_dir, 'X_test_mpnet.npy')),
            "Goal2_distilbert": np.load(os.path.join(text_features_dir, 'X_test_distilbert.npy'))
        }

        all_evaluation_metrics = {}

        for model_file in os.listdir(self.config.trained_model_dir):
            if model_file.endswith(".pkl"):
                model_name_with_features = os.path.splitext(model_file)[0]
                model_path = os.path.join(self.config.trained_model_dir, model_file)

                logging.info(f"Loading and evaluating model: {model_name_with_features}")
                try:
                    model = load_bin(model_path)

                    if model_name_with_features.startswith("Goal1_"):
                        goal = "Goal1"
                        parts = model_name_with_features.split('_')
                        model_name = parts[1] if len(parts) > 2 else "unknown_model"
                        feature_type = parts[-1]
                        test_feature_key = f"Goal1_{feature_type}"
                    elif model_name_with_features.startswith("Goal2_"):
                        goal = "Goal2"
                        parts = model_name_with_features.split('_')
                        model_name = parts[1] if len(parts) > 2 else "unknown_model"
                        feature_type = parts[-1]
                        test_feature_key = f"Goal2_{feature_type}"
                    else:
                        logging.warning(f"Unknown model naming convention for {model_name_with_features}. Skipping.")
                        continue

                    if test_feature_key in test_feature_data:
                        X_test = test_feature_data[test_feature_key]
                        y_pred_test = model.predict(X_test)
                        metrics = self._evaluate_model(y_test, y_pred_test)
                        all_evaluation_metrics[model_name_with_features] = metrics
                        logging.info(f"Metrics for {model_name_with_features}: {metrics}")

                        mlflow.log_param("goal", goal)
                        mlflow.log_param("model_name", model_name)
                        mlflow.log_param("feature_set", feature_type)
                        mlflow.log_metrics(metrics)
                        mlflow.log_artifact(local_path=model_path, artifact_path="evaluated_models")
                    else:
                        logging.error(f"Test data for key '{test_feature_key}' not found. Skipping evaluation for {model_name_with_features}.")

                except Exception as e:
                    logging.error(f"Error evaluating model {model_name_with_features}: {e}")
                    all_evaluation_metrics[model_name_with_features] = {"error": str(e)}

        with open(self.config.metric_file_path, 'w') as f:
            json.dump(all_evaluation_metrics, f, indent=4)
        logging.info(f"All model evaluation metrics saved to {self.config.metric_file_path}")
