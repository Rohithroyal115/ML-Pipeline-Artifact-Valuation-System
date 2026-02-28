import pandas as pd
import os
from src.utils.common import read_yaml
from src.entity import DataValidationConfig, DataTransformationConfig
from sklearn.model_selection import train_test_split
from src.logging import logging


class DataValidation:
    def __init__(self, config: DataValidationConfig, data_transformation_config: DataTransformationConfig, schema_path: str):
        self.config = config
        self.data_transformation_config = data_transformation_config
        self.schema_path = schema_path

    def validate_columns(self):
        logging.info("Starting data validation.")
        try:
            status = True
            messages = []

            df = pd.read_csv(self.config.local_file)
            self.schema = read_yaml(self.schema_path)

            expected_columns = list(self.schema.keys())
            data_columns = df.columns.tolist()

            missing_cols = [col for col in expected_columns if col not in data_columns]
            if missing_cols:
                status = False
                messages.append(f"Missing columns: {missing_cols}")

            def normalize_dtype(dtype_str):
                if "float" in dtype_str:
                    return "float"
                if "int" in dtype_str:
                    return "int"
                return dtype_str

            for col, col_type in self.schema.items():
                if col in df.columns:
                    actual_dtype = normalize_dtype(str(df[col].dtype))
                    if actual_dtype != col_type:
                        status = False
                        messages.append(f"Type mismatch for column '{col}': expected {col_type}, got {actual_dtype}")

            with open(self.config.validation_status_path, 'w') as f:
                f.write(str(status))
            logging.info(f"Validation status saved to {self.config.validation_status_path}")

            if status:
                logging.info("Data schema validation successful. Splitting data for transformation.")
                X = df.drop(columns=['preservation_score'])
                y = df['preservation_score']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                text_column = 'description'
                X_train_text = X_train[text_column].copy()
                X_test_text = X_test[text_column].copy()

                X_train_meta = X_train.drop(text_column, axis=1).copy()
                X_test_meta = X_test.drop(text_column, axis=1).copy()

                X_train_meta_final, X_val_meta, X_train_text_final, X_val_text, y_train_final, y_val = train_test_split(
                    X_train_meta, X_train_text, y_train,
                    test_size=0.2, random_state=42
                )

                train_df = pd.concat([X_train_meta_final.assign(description=X_train_text_final), y_train_final], axis=1)
                valid_df = pd.concat([X_val_meta.assign(description=X_val_text), y_val], axis=1)
                test_df = pd.concat([X_test_meta.assign(description=X_test_text), y_test], axis=1)

                os.makedirs(self.data_transformation_config.root_dir, exist_ok=True)
                train_df.to_csv(self.data_transformation_config.train_data_path, index=False)
                valid_df.to_csv(self.data_transformation_config.validation_data_path, index=False)
                test_df.to_csv(self.data_transformation_config.test_data_path, index=False)
                logging.info("Train, validation, and test data saved.")

                return status, train_df, valid_df, test_df
            else:
                logging.warning(f"Data validation failed. Issues: {messages}")
                return status, None, None, None

        except Exception as e:
            logging.error(f"Error during data validation: {e}")
            raise e
