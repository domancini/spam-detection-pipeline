# Email Spam Detection Pipeline


- [Email Spam Detection Pipeline](#email-spam-detection-pipeline)
  - [Project Overview](#project-overview)
    - [Directory structure](#directory-structure)
      - [`spam_detection` package:](#spam_detection-package)
      - [`spam_detection.modeling` subpackage:](#spam_detectionmodeling-subpackage)
      - [Tree](#tree)
    - [Key Features](#key-features)
  - [Dependencies](#dependencies)
  - [Installation](#installation)
  - [Usage](#usage)
  - [References](#references)
  - [License](#license)


This project implements a modular pipeline for classifying emails as spam or ham
 using supervised machine learning models and natural language processing
techniques. 
A version of the Nazario dataset is used, which can be found at
[Phishing Email Curated Datasets](https://zenodo.org/records/8339691)
 (Nazario_5.csv file). However, the modules can be used with any dataset with a
 similar structure.

The main entry point for the pipeline is the `final_script_py.py` file, which
runs the entire pipeline. The `Final.qmd` file similarly runs the pipeline and
includes the source code for the final report (rendered to
[PDF](reports/Final.pdf)). 

The best performing model was a Logistic Regression model (compared to Naive Bayes and SVM). 
- This model, including the TF-IDF vectorizer and scaler were saved to the `models` directory. See the [Report](reports/Final.pdf) for more details on the model performance.

## Project Overview

### Directory structure
The project is organized into several directories and files, each serving a specific
purposes:
- `models`: Contains the trained models and any necessary artifacts (e.g., vectorizers, scalers) used in the pipeline.
- `reports`: Contains the final report in PDF format, this is rendered from the `Final.qmd` file using Quarto. 
- `references`: Contains the references used in the report, including a BibTeX file and a CSL file for citation formatting (APA numeric superscript).

#### `spam_detection` package:

The `spam_detection` package contains the main source code for the project.
- `config.py`: Configuration file, sets paths to other locations in the project.
- `dataset.py`: Module for loading and processing the dataset. Includes function for loading and cleaning data, or loading raw data.
- `features.py`: Module for feature engineering, including functions for applying cleaning, stemming, and vectorization.
- `textual_features.py`: Module for extracting textual features from the emails, including TF-IDF vectorization and other text-based features
- `plots.py`: Module for generating plots and visualizations, including word clouds, class distributions, and model performance metrics.

#### `spam_detection.modeling` subpackage:
- `modeling`: Subpackage for model training and evaluation.
  - `model.py`: Contains the `Modeler` class, which handles the training and evaluation of models.
  - `train.py`: Contains functions for training the models, including hyperparameter tuning and cross-validation.

#### Tree
```bash
├─ Final.qmd <─ Quarto file for the final report
├─ final_script_py.py  <─ Python script of the final report
├─ README.md <─ You are here
├─ data <─ data folder
│   ├─ interim
│   ├─ processed
│   │   └─ enron_spam_data_half.csv
│   └─ raw
│       ├─ Nazario_5.csv
├─ models
│   ├─ LogisticRegression.joblib
│   ├─ scaler.joblib
│   └─ tfidf_vectorizer.joblib
├─ pyproject.toml <─ Dependency management files
├─ references
│   ├─ apa-numeric-superscript.csl
│   └─ refs.bib
├─ reports
│   ├─ Final.pdf
│   ├─ figures
└─ spam_detection <─ source code folder
    ├─ __init__.py
    ├─ config.py <─ Configuration file for the project
    ├─ dataset.py <─ Dataset class for loading and processing data
    ├─ features.py <─ Feature engineering functions
    ├─ modeling <─ Modeling subpackage
    │   ├─ __init__.py
    │   ├─ model.py <─ 'Modeler' class for training and evaluating models
    │   └─ train.py <─  Training functions for the model
    ├─ plots.py  <─ Model plotting functions
    └─ textual_features.py <─ Textual feature extraction functions
```

### Key Features
- **Unified Modeling API:** The [`spam_detection.modeling.model.Modeler`](spam_detection/modeling/model.py) class wraps scikit-learn estimators for easy training, evaluation, and persistence. 
  - Made with text classification in mind, but can be used for any supervised learning task.
  - Common interface for training, evaluating and saving models.
  - Stores various class attributes for easy access to model metrics and parameters.
  - Intialize with a random state for reproducibility.
- **Feature Engineering:** The [`spam_detection.features`](spam_detection/features.py) module provides functions for cleaning, stemming, and vectorizing text data:

## Dependencies

> [!NOTE]
> This project uses the `nltk` library. This library requires additional data files for certain features. <br>
> **When importing from `spam_detection`, it will automatically check if these data files are present, and download them if not**

For more control, you can also manually download them by running the following commands in a python shell or script if desired:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Or to use the `nltk` downloader GUI:
nltk.download() # then select the packages you want to download
```
The `stopwords` and `punkt` data files are used for text processing and tokenization.
See the [nltk documentation](https://www.nltk.org/data.html) for more information on downloading data files.

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

## Installation

1. **Clone the repository** and navigate to the project directory.
2. **Install dependencies** (see `pyproject.toml` or below):

```bash
pip install -r requirements.txt
```

To resolve the package imports, it is recommended to run:

```bash
pip install -e .
```
Where '.' is the pacth to the `spam_detection` directory. This will install the
package in editable mode, allowing you to run the scripts from anywhere in the
project directory.

## Usage


1. **Prepare Data:** Place `Nazario_5.csv` (or other) in `data/raw/` directory.
2. **Run the pipeline:**  
   Execute the main script to run the full pipeline:

    ```sh
    python final_script_py.py
    ```

    This will:
    - Load and preprocess data
    - Engineer features
    - Train and evaluate models
    - Output metrics and plots
    - Save the best model and preprocessing objects to [models](models) directory.

## References

- Nazario Spam Dataset: See refs.bib for citation.
- For methodology and results, see [Final.pdf](reports/Final.pdf).

## License

This project is for educational and research purposes only.



