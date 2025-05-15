import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import logging
from spam_detection.textual_features import extract_textual_features
from sklearn.preprocessing import StandardScaler

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
logger = logging.getLogger(__name__)


def clean_text(text):
    """
    Clean input text: lowercase, remove non-alphabetic chars, remove stopwords, stem words.
    """
    text = re.sub(r"[^a-zA-Z]", " ", text)  # remove non-alphabetic characters
    text = text.lower()  # convert to lowercase
    word_tokens = word_tokenize(text)  # tokenize
    # remove stopwords
    text = [token for token in word_tokens if token not in stop_words]
    text = " ".join(text)
    return text


def stem_text(text):
    """Stem words in the text using Porter Stemmer."""
    words = text.split()
    text = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(text)


def clean_and_stem_text(text):
    cleaned_text = clean_text(text)
    stemmed_text = stem_text(cleaned_text)
    return stemmed_text


def extract_features(df, max_features=1000):
    """
    Clean text and extract TF-IDF features.
    Returns transformed features X, target labels y, and the vectorizer.

    Args:
        df (pd.DataFrame): DataFrame containing the email data with 'text' and 'label' columns.
        max_features (int): Maximum number of features to extract.
    """
    logger.info("Extracting Features text data...")
    # clEaning and stemming text
    df["clean_text"] = df["text"].apply(clean_text).apply(stem_text)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values
    scaler = StandardScaler(with_mean=False)
    X_struct = extract_textual_features(df)
    # logger.info("Standardizing features...")
    X_struct_scaled = scaler.fit_transform(X_struct)
    # logger.info("Combining TF-IDF and scaled custom features...")
    X_combined = hstack([X, X_struct_scaled])

    return X_combined, y, vectorizer
