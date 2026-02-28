import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from src.logging import logging
from src.utils.common import save_bin
from src.entity import ModelTrainerConfig
from src.config.configuration import ConfigurationManager
from src.entity import MLOpsConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        os.makedirs(self.config.trained_model_dir, exist_ok=True)

    def _evaluate_model(self, y_true_val, y_pred_val):
        mae = mean_absolute_error(y_true_val, y_pred_val)
        mse = mean_squared_error(y_true_val, y_pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_val, y_pred_val)
        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    def train_goal1_models(self):
        logging.info("Starting training for Goal 1 models.")

        y_train = np.load(os.path.join(self.config.train_data_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(self.config.validation_data_dir, 'y_val.npy'))

        combined_features_dir = os.path.join(self.config.train_data_dir, 'combined_features')

        feature_sets = {
            "tfidf": {
                "train": np.load(os.path.join(combined_features_dir, 'X_train_goal1_tfidf.npy')),
                "val": np.load(os.path.join(combined_features_dir, 'X_val_goal1_tfidf.npy'))
            },
            "count": {
                "train": np.load(os.path.join(combined_features_dir, 'X_train_goal1_count.npy')),
                "val": np.load(os.path.join(combined_features_dir, 'X_val_goal1_count.npy'))
            },
            "mini": {
                "train": np.load(os.path.join(combined_features_dir, 'X_train_goal1_mini.npy')),
                "val": np.load(os.path.join(combined_features_dir, 'X_val_goal1_mini.npy'))
            },
            "mpnet": {
                "train": np.load(os.path.join(combined_features_dir, 'X_train_goal1_mpnet.npy')),
                "val": np.load(os.path.join(combined_features_dir, 'X_val_goal1_mpnet.npy'))
            },
            "distilbert": {
                "train": np.load(os.path.join(combined_features_dir, 'X_train_goal1_distilbert.npy')),
                "val": np.load(os.path.join(combined_features_dir, 'X_val_goal1_distilbert.npy'))
            }
        }

        models = {
            "XGBoostRegressor": XGBRegressor(),
            "LightGBMRegressor": LGBMRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=0),
            "MLPRegressor": MLPRegressor(max_iter=500, random_state=42),
            "RandomForestRegressor": RandomForestRegressor(random_state=42)
        }

        all_metrics = {}

        for fs_name, fs_data in feature_sets.items():
            X_train = fs_data["train"]
            X_val = fs_data["val"]

            for model_name, model in models.items():
                logging.info(f"Training Goal 1: {model_name} with {fs_name} features.")
                try:
                    with mlflow.start_run(nested=True):
                        model.fit(X_train, y_train)
                        y_pred_val = model.predict(X_val)
                        metrics = self._evaluate_model(y_val, y_pred_val)

                        model_key = f"Goal1_{model_name}_{fs_name}"
                        all_metrics[model_key] = metrics

                        model_path = os.path.join(self.config.trained_model_dir, f"{model_key}.pkl")
                        save_bin(model, model_path)
                        logging.info(f"Saved model: {model_path}")
                        logging.info(f"Metrics for {model_key}: {metrics}")

                        mlflow.log_param("goal", "Goal1")
                        mlflow.log_param("model_name", model_name)
                        mlflow.log_param("feature_set", fs_name)
                        mlflow.log_metrics(metrics)
                        mlflow.log_artifact(local_path=model_path, artifact_path="goal1_models")

                except Exception as e:
                    logging.error(f"Error training {model_name} with {fs_name} features: {e}")
                    all_metrics[f"Goal1_{model_name}_{fs_name}"] = {"error": str(e)}
                    raise e

        with open(self.config.metric_file_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logging.info(f"All Goal 1 model metrics saved to {self.config.metric_file_path}")

    def train_goal2_models(self):
        logging.info("Starting training for Goal 2 models.")

        y_train = np.load(os.path.join(self.config.train_data_dir, 'y_train.npy'))
        y_val = np.load(os.path.join(self.config.validation_data_dir, 'y_val.npy'))

        text_features_dir = self.config.text_vectorizer_path

        text_feature_sets = {
            "tfidf": {
                "train": np.load(os.path.join(text_features_dir, 'X_train_tfidf.npy')),
                "val": np.load(os.path.join(text_features_dir, 'X_val_tfidf.npy'))
            },
            "count": {
                "train": np.load(os.path.join(text_features_dir, 'X_train_count.npy')),
                "val": np.load(os.path.join(text_features_dir, 'X_val_count.npy'))
            },
            "mini": {
                "train": np.load(os.path.join(text_features_dir, 'X_train_mini.npy')),
                "val": np.load(os.path.join(text_features_dir, 'X_val_mini.npy'))
            },
            "mpnet": {
                "train": np.load(os.path.join(text_features_dir, 'X_train_mpnet.npy')),
                "val": np.load(os.path.join(text_features_dir, 'X_val_mpnet.npy'))
            },
            "distilbert": {
                "train": np.load(os.path.join(text_features_dir, 'X_train_distilbert.npy')),
                "val": np.load(os.path.join(text_features_dir, 'X_val_distilbert.npy'))
            }
        }

        models = {
            "RidgeRegressor": Ridge(random_state=42),
            "SVR": SVR(),
            "XGBoostRegressor": XGBRegressor(),
            "LightGBMRegressor": LGBMRegressor(),
            "CatBoostRegressor": CatBoostRegressor(verbose=0),
            "MLPRegressor": MLPRegressor(max_iter=500, random_state=42)
        }

        all_metrics = {}

        for fs_name, fs_data in text_feature_sets.items():
            X_train = fs_data["train"]
            X_val = fs_data["val"]

            for model_name, model in models.items():
                logging.info(f"Training Goal 2: {model_name} with {fs_name} text features.")
                try:
                    with mlflow.start_run(nested=True):
                        model.fit(X_train, y_train)
                        y_pred_val = model.predict(X_val)
                        metrics = self._evaluate_model(y_val, y_pred_val)

                        model_key = f"Goal2_{model_name}_{fs_name}"
                        all_metrics[model_key] = metrics

                        model_path = os.path.join(self.config.trained_model_dir, f"{model_key}.pkl")
                        save_bin(model, model_path)
                        logging.info(f"Saved model: {model_path}")
                        logging.info(f"Metrics for {model_key}: {metrics}")

                        mlflow.log_param("goal", "Goal2")
                        mlflow.log_param("model_name", model_name)
                        mlflow.log_param("feature_set", fs_name)
                        mlflow.log_metrics(metrics)
                        mlflow.log_artifact(local_path=model_path, artifact_path="goal2_models")

                except Exception as e:
                    logging.error(f"Error training {model_name} with {fs_name} text features: {e}")
                    all_metrics[f"Goal2_{model_name}_{fs_name}"] = {"error": str(e)}
                    raise e

        if os.path.exists(self.config.metric_file_path):
            with open(self.config.metric_file_path, 'r') as f:
                existing_metrics = json.load(f)
            existing_metrics.update(all_metrics)
            final_metrics = existing_metrics
        else:
            final_metrics = all_metrics

        with open(self.config.metric_file_path, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        logging.info(f"All Goal 2 model metrics saved to {self.config.metric_file_path}")

    def train_model(self):
        logging.info("Orchestrating model training for both goals.")
        self.train_goal1_models()
        self.train_goal2_models()
        logging.info("Model training orchestration finished.")
