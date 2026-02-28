import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import json

from src.utils.common import *
from src.entity import DataTransformationConfig
from src.logging import logging


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        os.makedirs(self.config.text_preprocessor_artifacts_dir, exist_ok=True)

    @staticmethod
    def clean_text(text):
        if pd.isna(text):
            return ""

        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def advanced_clean_text(self, text):
        if pd.isna(text):
            return ""

        text = self.clean_text(text)

        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

            return ' '.join(tokens)
        except Exception:
            return text

    def tfidf_approach(self, X_train_text, X_val_text, X_test_text):
        tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_n_gram_range,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )

        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text).toarray()
        X_val_tfidf = tfidf_vectorizer.transform(X_val_text).toarray()
        X_test_tfidf = tfidf_vectorizer.transform(X_test_text).toarray()

        save_bin(tfidf_vectorizer, os.path.join(self.config.text_preprocessor_artifacts_dir, 'tfidf_vectorizer.pkl'))

        return X_train_tfidf, X_val_tfidf, X_test_tfidf

    def count_vect_approach(self, X_train_text, X_val_text, X_test_text):
        count_vectorizer = CountVectorizer(
            max_features=self.config.count_vec_max_features,
            ngram_range=self.config.count_vec_ngram_range,
            min_df=2,
            max_df=0.9,
            stop_words='english'
        )

        X_train_count = count_vectorizer.fit_transform(X_train_text).toarray()
        X_val_count = count_vectorizer.transform(X_val_text).toarray()
        X_test_count = count_vectorizer.transform(X_test_text).toarray()

        save_bin(count_vectorizer, os.path.join(self.config.text_preprocessor_artifacts_dir, 'count_vectorizer.pkl'))

        return X_train_count, X_val_count, X_test_count

    def sentence_transformers_approach(self, X_train_text, X_val_text, X_test_text):
        st_embeddings = {}

        model_name_to_short_name = {
            "all-MiniLM-L6-v2": "mini",
            "all-mpnet-base-v2": "mpnet",
            "distilbert-base-nli-stsb-mean-tokens": "distilbert"
        }

        for model_name in self.config.sentence_transformer_models:
            model = SentenceTransformer(model_name)

            X_train_embed = model.encode(X_train_text.tolist())
            X_val_embed = model.encode(X_val_text.tolist())
            X_test_embed = model.encode(X_test_text.tolist())

            short_name = model_name_to_short_name.get(model_name)
            if short_name:
                st_embeddings[short_name] = (X_train_embed, X_val_embed, X_test_embed)

        return st_embeddings

    def initiate_data_transformation(self):
        logging.info("Starting data transformation.")
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            valid_df = pd.read_csv(self.config.validation_data_path)
            test_df = pd.read_csv(self.config.test_data_path)

            text_column = self.config.text_features_column

            X_train_text_cleaned = train_df[text_column].apply(self.advanced_clean_text)
            X_val_text_cleaned = valid_df[text_column].apply(self.advanced_clean_text)
            X_test_text_cleaned = test_df[text_column].apply(self.advanced_clean_text)

            X_train_meta = train_df.drop(columns=[text_column, 'preservation_score'])
            X_val_meta = valid_df.drop(columns=[text_column, 'preservation_score'])
            X_test_meta = test_df.drop(columns=[text_column, 'preservation_score'])

            meta_columns = X_train_meta.columns.tolist()
            with open(os.path.join(self.config.root_dir, 'meta_columns.json'), 'w') as f:
                json.dump(meta_columns, f)

            np.save(os.path.join(self.config.root_dir, 'X_train_meta.npy'), X_train_meta)
            np.save(os.path.join(self.config.root_dir, 'X_val_meta.npy'), X_val_meta)
            np.save(os.path.join(self.config.root_dir, 'X_test_meta.npy'), X_test_meta)

            y_train = train_df['preservation_score']
            y_val = valid_df['preservation_score']
            y_test = test_df['preservation_score']

            np.save(os.path.join(self.config.root_dir, 'y_train.npy'), y_train)
            np.save(os.path.join(self.config.root_dir, 'y_val.npy'), y_val)
            np.save(os.path.join(self.config.root_dir, 'y_test.npy'), y_test)

            X_train_tfidf, X_val_tfidf, X_test_tfidf = self.tfidf_approach(X_train_text_cleaned, X_val_text_cleaned, X_test_text_cleaned)
            np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, 'X_train_tfidf.npy'), X_train_tfidf)
            np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, 'X_val_tfidf.npy'), X_val_tfidf)
            np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, 'X_test_tfidf.npy'), X_test_tfidf)

            X_train_count, X_val_count, X_test_count = self.count_vect_approach(X_train_text_cleaned, X_val_text_cleaned, X_test_text_cleaned)
            np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, 'X_train_count.npy'), X_train_count)
            np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, 'X_val_count.npy'), X_val_count)
            np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, 'X_test_count.npy'), X_test_count)

            st_embeddings = self.sentence_transformers_approach(X_train_text_cleaned, X_val_text_cleaned, X_test_text_cleaned)

            for short_name, (train_embed, val_embed, test_embed) in st_embeddings.items():
                np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, f'X_train_{short_name}.npy'), train_embed)
                np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, f'X_val_{short_name}.npy'), val_embed)
                np.save(os.path.join(self.config.text_preprocessor_artifacts_dir, f'X_test_{short_name}.npy'), test_embed)

            logging.info("Data transformation finished.")
            return (
                X_train_meta, X_val_meta, X_test_meta,
                X_train_tfidf, X_val_tfidf, X_test_tfidf,
                X_train_count, X_val_count, X_test_count,
                st_embeddings,
                y_train, y_val, y_test
            )

        except Exception as e:
            logging.error(f"Data transformation failed: {e}")
            raise e
