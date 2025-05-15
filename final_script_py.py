# %%
#| echo: false
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from spam_detection.modeling import train, model
from spam_detection.modeling.model import Modeler
from spam_detection import config, features, textual_features, plots, dataset

# ### Data Loading and Preprocessing



labmap = {0: "ham", 1: "spam"}
df: pd.DataFrame = dataset.load_and_clean_data(config.NAZARIO_DATASET)
df["email_type"] = df["label"].map(labmap)  # type: ignore


# Merging features
all_feats = pd.merge(df, textual_features.extract_textual_features(
    df), left_index=True, right_index=True)


# This happens internally when using 
all_feats["clean_text"] = all_feats["text"].apply(features.clean_and_stem_text)


# ### TF-IDF
tfidf = TfidfVectorizer(max_features=1000)
scaler = StandardScaler(with_mean=False)


# %%
data = all_feats[["label", "clean_text", "num_urls",
                  "email_length", "subject_has_reply"]]
X = tfidf.fit_transform(data["clean_text"])
y = data["label"].values #type: ignore
X_struct = data[["num_urls", "email_length", "subject_has_reply"]]
X_struct_scaled = scaler.fit_transform(X_struct) # pyright: ignore
# combining the features
X_combined = np.hstack((X.toarray(), X_struct_scaled)) # pyright: ignore
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42
)


# Model training

nb_model = Modeler(MultinomialNB(), model_name="Naive Bayes", random_state=42)
nb_model.train(X_train, y_train)
nb_model.evaluate(X_test, y_test)
nb_model.confusion_matrix(y_test);

lr_model = Modeler(LogisticRegression(random_state=42),
                   model_name="Logistic Regression", random_state=42)
lr_model.train(X_train, y_train)
lr_model.evaluate(X_test, y_test)
lr_model.confusion_matrix(y_test);

# Save model, vectorizer, and scaler to disk
joblib.dump(lr_model.model, config.MODELS_DIR / "LogisticRegression.joblib")
joblib.dump(tfidf, config.MODELS_DIR / "tfidf_vectorizer.joblib")
joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
print("Logistic Regression model saved as LogisticRegression.joblib")



svm_model = Modeler(SVC(kernel="linear", probability=True, random_state=42), model_name="SVM", random_state=42)
svm_model.train(X_train, y_train)
svm_model.evaluate(X_test, y_test)
svm_model.confusion_matrix(y_test);



# ## Evaluation and Results

# Model Metrics Comparison
metrics = {}

for m in [nb_model, lr_model, svm_model]:
    metrics[m.model_name] = m.metrics

metric_df = pd.DataFrame.from_dict(metrics, orient="index").round(3)
print("\nComparison of Model Metrics:")
print(metric_df)


bmpm = {}
for metric in metric_df.columns:
    # worst_value = metric_df[metric].min()
    # worst_model = metric_df[metric].idxmin()
    best_model = metric_df[metric].idxmax()
    best_value = float(metric_df[metric].max())
    metric = metric.lower()
    bmpm[metric] = {
        'model': best_model,
        'value': best_value,
    }

print("\nBest Model Performance Metrics:")
pd.DataFrame.from_dict(bmpm, orient="index").round(3)

# ### Results and Visualizations


mods = [nb_model, lr_model, svm_model]
fig = plt.figure(figsize=(10, 10), dpi=80)
ax = plt.gca()
for m in mods:
    PrecisionRecallDisplay.from_estimator(
        m.model, X_test, y_test, name=m.model_name, ax=ax)
plt.title("Precision-Recall Curve")
plt.show()



fig = plt.figure(figsize=(10, 10), dpi=80)
ax = plt.gca()
for m in mods:
    RocCurveDisplay.from_estimator(
        m.model, X_test, y_test, name=m.model_name, ax=ax)
plt.show()
