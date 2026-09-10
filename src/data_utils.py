"""Loading, cleaning and vectorization utilities for the SMS Spam Collection dataset."""

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = "../data/raw/spam.csv"

_URL_RE = re.compile(r"http\S+|www\.\S+")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle SMS Spam Collection CSV (v1=label, v2=text)."""
    df = pd.read_csv(path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    df["label"] = (df["label"] == "spam").astype(int)
    return df.dropna().reset_index(drop=True)


def clean_text(text: str) -> str:
    """Lowercase, strip URLs and non-letter characters, collapse whitespace."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def split_dataset(
    df: pd.DataFrame,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """Stratified split into train/val/test, done before vectorization to avoid leakage."""
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label"], random_state=random_state
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, stratify=temp_df["label"], random_state=random_state
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def vectorize(train_df, val_df, test_df, max_features: int = 3000, ngram_range=(1, 2)):
    """Fit TF-IDF on the train split only, transform all three splits."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = vectorizer.fit_transform(train_df["text"]).toarray().astype(np.float32)
    X_val = vectorizer.transform(val_df["text"]).toarray().astype(np.float32)
    X_test = vectorizer.transform(test_df["text"]).toarray().astype(np.float32)

    y_train = train_df["label"].to_numpy().astype(np.float32)
    y_val = val_df["label"].to_numpy().astype(np.float32)
    y_test = test_df["label"].to_numpy().astype(np.float32)

    return vectorizer, (X_train, y_train), (X_val, y_val), (X_test, y_test)


def load_and_prepare(path: str = RAW_DATA_PATH, max_features: int = 3000, ngram_range=(1, 2)):
    """Full pipeline: load -> clean -> split -> vectorize. Returns splits + fitted vectorizer."""
    df = load_raw(path)
    df["text"] = df["text"].apply(clean_text)
    train_df, val_df, test_df = split_dataset(df)
    vectorizer, train, val, test = vectorize(
        train_df, val_df, test_df, max_features=max_features, ngram_range=ngram_range
    )
    return vectorizer, train, val, test
