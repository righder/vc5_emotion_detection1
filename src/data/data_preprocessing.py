import os
import re
import numpy as np
import pandas as pd
import nltk
import logging
import string
from typing import Any, Callable
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def download_nltk_resources() -> None:
    try:
        nltk.download('wordnet')
        nltk.download('stopwords')
        logging.info("NLTK resources downloaded successfully.")
    except Exception as e:
        logging.error(f"Error downloading NLTK resources: {e}")
        raise

def lemmatization(text: str) -> str:
    try:
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        return " ".join(lemmatized)
    except Exception as e:
        logging.error(f"Error in lemmatization: {e}")
        return text

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        words = [word for word in str(text).split() if word not in stop_words]
        return " ".join(words)
    except Exception as e:
        logging.error(f"Error removing stop words: {e}")
        return text

def removing_numbers(text: str) -> str:
    try:
        return ''.join([char for char in text if not char.isdigit()])
    except Exception as e:
        logging.error(f"Error removing numbers: {e}")
        return text

def lower_case(text: str) -> str:
    try:
        return " ".join([word.lower() for word in text.split()])
    except Exception as e:
        logging.error(f"Error converting to lower case: {e}")
        return text

def removing_punctuations(text: str) -> str:
    try:
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
        text = text.replace('؛', "")
        text = re.sub('\s+', ' ', text)
        text = " ".join(text.split())
        return text.strip()
    except Exception as e:
        logging.error(f"Error removing punctuations: {e}")
        return text

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logging.error(f"Error removing URLs: {e}")
        return text

def remove_small_sentences(df: pd.DataFrame) -> None:
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
    except Exception as e:
        logging.error(f"Error removing small sentences: {e}")

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content = df.content.apply(lower_case)
        df.content = df.content.apply(remove_stop_words)
        df.content = df.content.apply(removing_numbers)
        df.content = df.content.apply(removing_punctuations)
        df.content = df.content.apply(removing_urls)
        df.content = df.content.apply(lemmatization)
        logging.info(f"Normalized DataFrame with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error normalizing text: {e}")
        return df

def normalized_sentence(sentence: str) -> str:
    try:
        sentence = lower_case(sentence)
        sentence = remove_stop_words(sentence)
        sentence = removing_numbers(sentence)
        sentence = removing_punctuations(sentence)
        sentence = removing_urls(sentence)
        sentence = lemmatization(sentence)
        return sentence
    except Exception as e:
        logging.error(f"Error normalizing sentence: {e}")
        return sentence

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        logging.info(f"Loaded data from {path} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {path}: {e}")
        raise

def save_data(df: pd.DataFrame, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Saved data to {path} with shape {df.shape}")
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")
        raise

def main() -> None:
    try:
        download_nltk_resources()
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")
        train_data = normalize_text(train_data)
        test_data = normalize_text(test_data)
        save_data(train_data, "data/processed/train.csv")
        save_data(test_data, "data/processed/test.csv")
        logging.info("Data preprocessing completed successfully.")
    except Exception as e:
        logging.critical(f"Data preprocessing failed: {e}")

if __name__ == "__main__":
    main()