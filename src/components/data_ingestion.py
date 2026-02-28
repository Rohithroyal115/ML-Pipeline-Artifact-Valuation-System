import pandas as pd
import os

from src.utils.common import *
from src.logging import logging
from src.entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def read_data(self) -> pd.DataFrame:
        try:
            logging.info("Starting data ingestion.")
            df = pd.read_csv(self.config.local_file)

            os.makedirs(self.config.root_dir, exist_ok=True)

            output_path = os.path.join(self.config.root_dir, 'data.csv')
            df.to_csv(output_path, index=False)

            logging.info("Data ingestion finished.")
            return df
        except Exception as e:
            logging.error(f"Data ingestion failed: {e}")
            raise e
