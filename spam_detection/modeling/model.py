import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import joblib
from spam_detection.config import MODELS_DIR
from spam_detection import plots


class Modeler:
    """Class to handle model training, evaluation, and saving/loading.
        Wraps sklearn models providing general functions for pipeline.

    Attributes:
        model: sklearn model
        model_name: (str) Optional name - uses class name otherwise
        random_state: (int) Optional random state for reproducibility
        y_pred: Predicted labels for the test set, set after 'evaluate'
        metrics: (dict) Dictionary of evaluation metrics
        model_path: Model path for saving/loading.
    """

    def __init__(self, model: MultinomialNB | LogisticRegression | SVC, model_name: str = "", random_state: int | None = None):
        self.model = model
        self.model_name = model_name if model_name else type(model).__name__
        self.y_pred = None
        self.metrics = None
        self.model_path = MODELS_DIR / f"{self.model_name}.joblib"
        self.random_state = random_state
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def save_model(self, filename):
        """
        Save the model to a file. If no filename is provided, it saves to the
        default model path in reports/<modelname>.joblib.
        Args:
            filename (optional): The name of the file to save the model to.
        """
        filename = filename if filename else (self.model_path)
        joblib.dump(self.model, filename)

    def save_model_vectorizer(self, vectorizer: TfidfVectorizer, vec_path=None):
        """Save the fitted vectorizer to a file. If no filename is provided, it saves to the
        default model path in reports/<modelname>_vectorizer.joblib.

        Args:
            vec_path (optional): string path name to save the vectorizer to.
            vectorizer: TfidfVectorizer or similar object to save.
        """
        vec_path = vec_path if vec_path else f"{self.model_name}"
        joblib.dump(self.model, self.model_path)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data and print the classification report.
        Args:
            X_test: Test features.
            y_test: Test labels.
        """
        y_pred = self.model.predict(X_test)
        self.y_pred = y_pred # initialize y_pred attribute
        metrics = classification_report(y_test, y_pred)
        # storing just the class attribute report as a dict
        self.metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
        }
        print(metrics)
        return metrics

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation on the model.
        Args:
            X: Features.
            y: Labels.
            cv: Number of folds for cross-validation.
        Returns:
        Cross-validation scores.
                """
        scores = cross_val_score(self.model, X, y, cv=cv)
        # print(f"Cross-validation scores: {scores}")
        return scores

    def train(self, X_train, y_train):
        """
        Train the model on the training data.
        Args:
            X_train: Training features.
            y_train: Training labels.
        """
        self.model.fit(X_train, y_train)
        return self.model

    def grid_search(self, X_train, y_train, param_grid, cv=5):
        """
        Perform grid search to find the best hyperparameters for the model.
        Args:
            X_train: Training features.
            y_train: Training labels.
            param_grid: Dictionary of hyperparameters to search.
            cv: Number of folds for cross-validation.
        Returns:
            Best model from grid search.
        """
        grid_search = GridSearchCV(self.model, param_grid, cv=cv)
        grid_search.fit(X_train, y_train)
        print(f"Best parameters: {grid_search.best_params_}")
        return grid_search.best_estimator_

    def load_model(self, filename: str):
        """
        Load a model from a file.
        Args:
            filename: The name of the file to load the model from.
        """
        self.model = joblib.load(filename)
        return self.model

    def confusion_matrix(self, y_true):
        """Plot the confusion matrix. Internally uses the 'plots' module to
        create plot. Must use 'evaluate' to set predictions first.

        Args:
            y_true: True labels for the test set.

        Returns:
            figure: Confusion Matrix plot.

        Raises:
            ValueError: If the model has not been evaluated yet.
        """
        if self.y_pred is None:
            raise ValueError(
                "Model has not been evaluated yet. Please call evaluate() first.")
        return plots.plot_confusion_matrix(y_true, self.y_pred, self.model_name)

    def roc_curve(self, y_test):
        """Plot the ROC curve. Internally uses the 'plots' module to create plot.
        Must use 'evaluate' to set predictions first.
        Args:
            y_test: True labels for the test set.
        Returns:
            figure: ROC Curve plot.
        Raises:
            ValueError: If the model has not been evaluated yet.
        """
        if self.y_pred is None:
            raise ValueError(
                "Model has not been evaluated yet. Please call evaluate() first.")
        return plots.plot_roc_curves(y_test, self.y_pred, self.model_name)


def evaluate_model(model, X_test, y_test):
    """
    Static function to evaluate a model on the test data, print classification
        report.
    Args:
        model: The trained model to evaluate.
        X_test: Test features.
        y_test: Test labels.
    """
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    return y_pred


def train_models(X_train, y_train, models):
    """
    Trains multiple models on the training data.

    Args:
        X_train: Training features.
        y_train: Training labels.
        models: Dictionary of model names and model objects.

    Returns:
        A dictionary of trained models.
    """
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models
