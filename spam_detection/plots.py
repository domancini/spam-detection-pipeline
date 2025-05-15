import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, auc, roc_auc_score, roc_curve
from spam_detection.config import PROCESSED_DATA_DIR, FIGURES_DIR
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import warnings
# from spam_detection.config import FIGURES_DIR, PROCESSED_DATA_DIR

color_map = {'ham': 'C1', 'spam': 'C0'}
warnings.filterwarnings("ignore")
plt.style.use("ggplot")


def save_plot(fig, filename):
    """Save the plot figure to filename in 'reports/figures' directory"""
    fig.savefig(FIGURES_DIR / filename)


def plot_roc_curves(models: dict, X_test, y_test):
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(layout="tight")
    for name, model in models.items():
        y_scores = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    # fig, ax = plt.subplots(layout="tight")
    # ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison')
    ax.legend(loc='lower right')
    return fig


def create_word_cloud(email_col, type="Spam"):
    """Generate a word cloud of the given text in column.

    Args:
        email_col (array-like): An array of text data.
        type (str): The value label (default 'Spam')

    Returns:
        Figure
    """
    emails = " ".join(email_col.astype(str))
    wc = WordCloud(
        max_words=100,
        width=1600,
        height=800,
        min_word_length=3
    ).generate(emails)
    fig, ax = plt.subplots(facecolor='k')
    ax.imshow(wc, interpolation='bilinear')
    # image so turn off axis
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    return fig


def aggregate_by_label(df: pd.DataFrame, column_name: str, agg: str, column_label: str = ""):
    """Return a plot figure comparing an `agg` for `column_name` by email type

    Args:
        df: Dataframe
        column_name: Column to aggregate
        agg: Aggregate to use as a string (e.g. "mean", "sum", "count")
        column_label: Display name of the column

    Returns:
        Figure: A plot figure
    """
    fig, ax = plt.subplots(layout="tight")
    column_label = column_label if column_label else column_name
    agg_values = df.groupby("email_type")[column_name].agg(agg)
    # palette arguments sets ham = blue, spam = red
    sns.barplot(x=agg_values.index, y=agg_values.values,
                hue=agg_values.index, palette=color_map, ax=ax)
    ax.set_title(f"{agg.title()} {column_label.title()} for Spam and Ham")
    ax.set_ylabel(f"{agg.title()} {column_label}")
    ax.set_xlabel("Email Type")
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Return a labeled confusion matrix using the provided model and data.
    """
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    fig, ax = plt.subplots(layout="tight")
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(cm, annot=True, fmt='d', cmap="viridis_r")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.xaxis.set_ticklabels(["Negative", "Positive"])
    ax.yaxis.set_ticklabels(["Negative", "Positive"])
    ax.set_title("Confusion Matrix")
    return fig
