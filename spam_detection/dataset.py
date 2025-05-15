from pathlib import Path
import pandas as pd
import re

from spam_detection.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def load_and_clean_data(path):
    """
    Load the Nazario spam dataset, creates 'text' column combining
    'subject' + 'body'.
    Args:
        path (str): Path to the CSV file.
    Returns:
        DataFrame: Cleaned DataFrame with 'text', 'subject' and 'label' columns.
    """
    df = pd.read_csv(path)
    # Drop rows missing body or label
    df = df.dropna(subset=["body", "label"]).reset_index(drop=True)
    # Ensure label is binary (int)
    df["label"] = df["label"].astype(int)
    # Combine subject and body for analysis
    df["subject"] = df["subject"].fillna("")
    df["text"] = (df["subject"] + " " + df["body"]).astype(str)
    return df


def load_raw_data(filename="Nazario_5.csv"):
    """
    Alternative function to load raw data from a CSV file.
    """
    path = RAW_DATA_DIR / filename
    # df = pd.read_csv(path, encoding="latin-1")
    df = pd.read_csv(path)
    return df
