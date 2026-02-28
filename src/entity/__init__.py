from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

@dataclass
class DataIngestionConfig:
    root_dir: Path
    local_file: Path

@dataclass
class DataValidationConfig:
    root_dir: Path
    local_file: Path
    validation_status_path: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    validation_data_path: Path
    test_data_path: Path
    text_features_column: str
    tfidf_max_features: int
    tfidf_n_gram_range: Tuple[int, int]
    count_vec_max_features: int
    count_vec_ngram_range: Tuple[int, int]
    sentence_transformer_models: List[str]
    text_preprocessor_artifacts_dir: Path

@dataclass
class DataPreprocessingConfig:
    root_dir: Path
    numeric_features: List[str]
    categorical_features: List[str]
    imputation_strategy_numeric: str
    imputation_strategy_categorical: str
    scaler_type: str
    encoder_type: str
    transformed_data_dir: Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_dir: Path
    validation_data_dir: Path
    test_data_dir: Path
    preprocessor_path: Path
    text_vectorizer_path: Path
    target_column: str
    model_params: Dict
    metric_file_path: Path
    trained_model_dir: Path

@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_dir: Path
    trained_model_dir: Path
    metric_file_path: Path
    text_vectorizer_path: Path

@dataclass
class MLOpsConfig:
    mlflow_uri: str
    dagshub_user: str
    dagshub_repo: str
