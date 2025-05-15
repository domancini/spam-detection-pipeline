from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
from joblib import dump, load
import os
from spam_detection.config import MODELS_DIR
import logging
logger = logging.getLogger(__name__)


def save_model(model, filename=None):
    """Saves the trained model to a file in the models directory.

    Args:
        model: trained model to save
        filename (str): Optional: filename to save model to. If not provided, defaults to the model class name.
    """
    # set name to the model name if not provided (a unique name is generated), otherwise use the provided name
    # if filename is None:
    #     filename = f"{model.__class__.__name__}.joblib"
    if filename is None:
        #
        filename = os.path.join(
            MODELS_DIR, f"{model.__class__.__name__}.joblib")

    dump(model, filename, compress=True)


def train_models(X, y):
    """
    Trains and evaluates three classifiers: Naive Bayes, Logistic Regression, and SVM.
    Returns a dictionary with each model, its report, and the best model.
    """
    logger.info("Training Models..")
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    results = {}
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = report["weighted avg"]["f1-score"] #type: ignore
        results[name] = {
            "model": model,
            "f1_score": f1,
            "report_dict": report,
        }
        if f1 > best_f1: #type: ignore
            best_model = model
            best_f1 = f1
            best_model_name = name

    return {
        "all_models": results,
        "best_model": best_model,
        "best_model_name": best_model_name, #type: ignore
    }
