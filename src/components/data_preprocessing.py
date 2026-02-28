import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import json

from src.utils.common import *
from src.entity import DataPreprocessingConfig, DataTransformationConfig
from src.components.data_validation import DataValidation
from src.logging import logging

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig, data_validation: DataValidation, data_transformation_config: DataTransformationConfig):
        self.config = config
        self.data_validation = data_validation
        self.data_transformation_config = data_transformation_config

    def initiate_data_preprocess(self, X_train_meta, X_val_meta, X_test_meta, y_train, y_val, y_test):
        logging.info("Starting data preprocessing.")
        try:
            numeric_features = self.config.numeric_features
            categorical_features = self.config.categorical_features

            numeric_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy=self.config.imputation_strategy_numeric)),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy=self.config.imputation_strategy_categorical, fill_value='Not Mentioned')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            logging.info("Applying preprocessing to metadata features.")
            X_train_meta_processed = preprocessor.fit_transform(X_train_meta)
            X_val_meta_processed = preprocessor.transform(X_val_meta)
            X_test_meta_processed = preprocessor.transform(X_test_meta)

            structured_features_dir = os.path.join(self.config.root_dir, 'structured_features')
            os.makedirs(structured_features_dir, exist_ok=True)

            np.save(os.path.join(structured_features_dir, 'X_train_meta_processed.npy'), X_train_meta_processed)
            np.save(os.path.join(structured_features_dir, 'X_val_meta_processed.npy'), X_val_meta_processed)
            np.save(os.path.join(structured_features_dir, 'X_test_meta_processed.npy'), X_test_meta_processed)
            logging.info("Processed metadata features saved.")

            np.save(os.path.join(self.config.root_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(self.config.root_dir, 'y_val.npy'), y_val)
            np.save(os.path.join(self.config.root_dir, 'y_test.npy'), y_test)
            logging.info("Target arrays (y_train, y_val, y_test) saved.")

            save_bin(preprocessor, os.path.join(structured_features_dir, 'structured_preprocessor.pkl'))
            logging.info("Structured data preprocessor saved.")

            text_features_dir = self.data_transformation_config.text_preprocessor_artifacts_dir

            logging.info("Loading text features for combination.")
            X_train_tfidf = np.load(os.path.join(text_features_dir, 'X_train_tfidf.npy'))
            X_val_tfidf = np.load(os.path.join(text_features_dir, 'X_val_tfidf.npy'))
            X_test_tfidf = np.load(os.path.join(text_features_dir, 'X_test_tfidf.npy'))

            X_train_count = np.load(os.path.join(text_features_dir, 'X_train_count.npy'))
            X_val_count = np.load(os.path.join(text_features_dir, 'X_val_count.npy'))
            X_test_count = np.load(os.path.join(text_features_dir, 'X_test_count.npy'))

            X_train_mini = np.load(os.path.join(text_features_dir, 'X_train_mini.npy'))
            X_val_mini = np.load(os.path.join(text_features_dir, 'X_val_mini.npy'))
            X_test_mini = np.load(os.path.join(text_features_dir, 'X_test_mini.npy'))

            X_train_mpnet = np.load(os.path.join(text_features_dir, 'X_train_mpnet.npy'))
            X_val_mpnet = np.load(os.path.join(text_features_dir, 'X_val_mpnet.npy'))
            X_test_mpnet = np.load(os.path.join(text_features_dir, 'X_test_mpnet.npy'))

            X_train_distilbert = np.load(os.path.join(text_features_dir, 'X_train_distilbert.npy'))
            X_val_distilbert = np.load(os.path.join(text_features_dir, 'X_val_distilbert.npy'))
            X_test_distilbert = np.load(os.path.join(text_features_dir, 'X_test_distilbert.npy'))

            combined_features_dir = os.path.join(self.config.root_dir, 'combined_features')
            os.makedirs(combined_features_dir, exist_ok=True)
            logging.info(f"Created directory for combined features: {combined_features_dir}")

            logging.info("Combining metadata and text features (Goal 1 approaches).")
            X_train_goal1_tfidf = np.hstack([X_train_meta_processed, X_train_tfidf])
            X_val_goal1_tfidf = np.hstack([X_val_meta_processed, X_val_tfidf])
            X_test_goal1_tfidf = np.hstack([X_test_meta_processed, X_test_tfidf])

            X_train_goal1_count = np.hstack([X_train_meta_processed, X_train_count])
            X_val_goal1_count = np.hstack([X_val_meta_processed, X_val_count])
            X_test_goal1_count = np.hstack([X_test_meta_processed, X_test_count])

            X_train_goal1_mini = np.hstack([X_train_meta_processed, X_train_mini])
            X_val_goal1_mini = np.hstack([X_val_meta_processed, X_val_mini])
            X_test_goal1_mini = np.hstack([X_test_meta_processed, X_test_mini])

            X_train_goal1_mpnet = np.hstack([X_train_meta_processed, X_train_mpnet])
            X_val_goal1_mpnet = np.hstack([X_val_meta_processed, X_val_mpnet])
            X_test_goal1_mpnet = np.hstack([X_test_meta_processed, X_test_mpnet])

            X_train_goal1_distilbert = np.hstack([X_train_meta_processed, X_train_distilbert])
            X_val_goal1_distilbert = np.hstack([X_val_meta_processed, X_val_distilbert])
            X_test_goal1_distilbert = np.hstack([X_test_meta_processed, X_test_distilbert])

            print(f"Goal 2 - TF-IDF Only:")
            print(f"  Train: {X_train_tfidf.shape}")
            print(f"  Validation: {X_val_tfidf.shape}")
            print(f"  Test: {X_test_tfidf.shape}")

            print(f"\nGoal 2 - Count Vectorizer Only:")
            print(f"  Train: {X_train_count.shape}")
            print(f"  Validation: {X_val_count.shape}")
            print(f"  Test: {X_test_count.shape}")

            print(f"\nGoal 2 - MiniLM Only:")
            print(f"  Train: {X_train_mini.shape}")
            print(f"  Validation: {X_val_mini.shape}")
            print(f"  Test: {X_test_mini.shape}")

            print(f"\nGoal 2 - MPNet Only:")
            print(f"  Train: {X_train_mpnet.shape}")
            print(f"  Validation: {X_val_mpnet.shape}")
            print(f"  Test: {X_test_mpnet.shape}")

            print(f"\nGoal 2 - DistilBERT Only:")
            print(f"  Train: {X_train_distilbert.shape}")
            print(f"  Validation: {X_val_distilbert.shape}")
            print(f"  Test: {X_test_distilbert.shape}")

            logging.info("Data preprocessing finished.")

        except Exception as e:
            logging.error(f"Data preprocessing failed: {e}")
            raise e
