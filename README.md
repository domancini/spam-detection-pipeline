# Spam Detection

A spam detection project using machine learning techniques to classify emails
as spam or ham. The project uses the Nazario dataset, which contains a
collection of spam and ham emails.

The project source code lives in the `spam_detection` directory.

The `Final.qmd` is the source file for the final report, which is generated using Quarto.

`final_script_py.py` is the main script that runs the entire pipeline.

## Dependencies


```requirements
beautifulsoup4
joblib
matplotlib
nltk
numpy
pandas
scikit_learn
scipy
seaborn
wordcloud
```
note: The `pyproject.toml` file is used to manage dependencies and can be used with `pip` or `poetry`.



## Usage

See the `final_script_py.py` file for the main script that runs the entire pipeline.



```bash
-- Final.qmd <- Quarto file for the final report
|-- final_script_py.py  <- Python script of the final report
|-- README.md <- You are here
|-- data <- data folder
|   |-- interim
|   |-- processed
|   |   `-- enron_spam_data_half.csv
|   `-- raw
|       |-- Nazario_5.csv
|-- models
|   |-- LogisticRegression.joblib
|   |-- scaler.joblib
|   `-- tfidf_vectorizer.joblib
|-- pyproject.toml
|-- references
|   |-- apa-numeric-superscript.csl
|   `-- refs.bib
|-- reports
|   |-- Final.pdf
|   |-- figures
`-- spam_detection <- source code folder
    |-- __init__.py
    |-- config.py <- Configuration file for the project
    |-- dataset.py <- Dataset class for loading and processing data
    |-- features.py <- Feature engineering functions
    |-- modeling <- Modeling subpackage
    |   |-- __init__.py
    |   |-- model.py <- 'Modeler' class for training and evaluating models
    |   `-- train.py <-  Training functions for the model
    |-- plots.py  <- Model plotting functions
    `-- textual_features.py <- Textual feature extraction functions
```




