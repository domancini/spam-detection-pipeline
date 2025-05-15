import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import pandas as pd
import ast
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
SPAMMY_PHRASES = [
    "winner", "click here", "free", "urgent", "cash", "limited time", "claim",
    "no cost", "earn money", "guaranteed", "risk-free",
]


# Retreived from `cycbercrimeinfocenter.org` annual report on of most used spam TLD
SPAMMY_TLDS = [
    ".cn", ".ru", ".tk", ".click", ".biz", ".info", ".xyz", ".top", ".cc", ".pw", ".ga", ".gq", ".net", ".shop", ".work", ".stream"
]


def count_spammy_phrases(text):
    """Count the number of spammy phrases in the text."""
    text = text.lower()
    return sum(phrase in text for phrase in SPAMMY_PHRASES)


def has_suspicious_tld(urls):
    """Check if any URL in the list has a suspicious TLD."""
    for url in urls:
        tld = urlparse(url).netloc.split('.')[-1]
        if any(url.endswith(tld) for tld in SPAMMY_TLDS):
            return 1
    return 0


def parse_html_tags(body: str):
    """
    Parse HTML tags from the text and return the cleaned text.
    """
    soup = BeautifulSoup(body, "html.parser")

    all_tags = soup.find_all(True)
    num_tags = len(all_tags)
    num_links = len(soup.find_all("a"))
    has_script = int(bool(soup.find("script")))
    html_ratio = num_tags / max(len(body), 1)

    return {
        "num_tags": num_tags,
        "num_links": num_links,
        "has_script": has_script,
        "html_ratio": html_ratio
    }


def safe_parse_urls(urls):
    """Safely parse URLs from a string representation of a list."""
    try:
        parsed_urls = ast.literal_eval(urls) if pd.notnull(urls) else []
        return parsed_urls if isinstance(parsed_urls, list) else []
    except (ValueError, SyntaxError):
        return []


def extract_textual_features(df: pd.DataFrame):
    """Extract textual features from the email body and subject.


    Args:
        df: DataFrame containing the email data with 'body', 'subject', and
            'urls' columns.
    Returns:
        DataFrame: A DataFrame containing the extracted features.
    """
    # Create url_list as a standalone Series
    # logger.info("Extracting textual features...")
    url_list = df["urls"].apply(safe_parse_urls)

    features = pd.DataFrame()
    features["email_length"] = df["body"].str.len()
    features["num_words"] = df["body"].str.split().apply(len)
    features["num_exclamations"] = df["body"].str.count("!")
    features["num_uppercase_words"] = df["body"].apply(
        lambda x: sum(1 for w in x.split() if w.isupper()))
    features["contains_spammy_phrases"] = df["body"].apply(
        count_spammy_phrases)
    features["num_urls"] = url_list.str.len()
    features["subject_has_reply"] = df["subject"].str.lower(
    ).str.startswith(("re:", "fwd:")).astype(float)
    features["has_suspicious_tld"] = url_list.apply(
        has_suspicious_tld)  # Use url_list directly

    html_features = df["body"].apply(parse_html_tags).apply(pd.Series)
    all_features = pd.concat([features, html_features], axis=1)

    # Add label from 'df' to 'features' DataFrame
    # features["label"] = df["label"]
    return all_features
