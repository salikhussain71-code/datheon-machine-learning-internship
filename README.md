# Datheon Machine Learning & Data Science Internship

![Python](https://img.shields.io/badge/Python-3.14.3-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.9.1-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-red)
![Tests](https://img.shields.io/badge/Tests-22%20Passed-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**Three end-to-end machine learning projects developed for the Datheon Machine Learning & Data Science Internship (Standard Internship, 4 weeks).**

This repository contains three supervised machine learning projects covering multi-class classification, customer churn prediction, and natural language processing (NLP) sentiment classification.

Each selected task includes a Jupyter notebook, data preprocessing, model training, evaluation, visualizations, and supporting project documentation. The repository also includes reusable Python modules, saved model artifacts, automated tests, and a continuous integration workflow.

**Author:** Salik Hussain  
**GitHub:** [salikhussain71-code](https://github.com/salikhussain71-code)  
**Repository:** [datheon-machine-learning-internship](https://github.com/salikhussain71-code/datheon-machine-learning-internship)  
**Program:** [Datheon Machine Learning & Data Science Internship](https://www.datheon.in/training/catalog/machine-learning-and-data-science-internship)

---

## 1. Project Overview

The objective of this work is to implement practical machine learning workflows using Python and commonly used data science libraries.

Three tasks were selected from the official Datheon task pack:

| Task | Project | Models | Main evaluation |
|---|---|---|---|
| Task 1 | Iris Multi-Class Classification | Logistic Regression, Decision Tree | Accuracy, precision, recall, F1 |
| Task 3 | Telecom Customer Churn Prediction | Random Forest, XGBoost | ROC-AUC, recall, F1 |
| Task 4 | NLP Sentiment Classification | Multinomial Naive Bayes, Logistic Regression | Accuracy, precision, recall, F1 |

Task 2, California Housing Price Prediction, is not included. The official program asks participants to complete at least two or three predictive modelling tasks. This repository contains three selected tasks.

## 2. Results at a Glance

The following results are taken from the project's reported evaluation outputs.

| Project | Model | Evaluation result |
|---|---|---|
| Iris Classification | Logistic Regression | 93.3% test accuracy |
| Iris Classification | Decision Tree | 96.7% test accuracy |
| Customer Churn | XGBoost | 0.8448 test ROC-AUC |
| Customer Churn | Random Forest | 0.8433 test ROC-AUC |
| NLP Sentiment | Logistic Regression | 84.3% test accuracy |
| NLP Sentiment | Multinomial Naive Bayes | 81.5% test accuracy |

### Task 1: Iris Classification

- Test set: 30 flowers.
- Logistic Regression accuracy: 93.3%.
- Decision Tree accuracy: 96.7%.
- Five-fold cross-validation accuracy:
  - Logistic Regression: 96.6%.
  - Decision Tree: 93.2%.

### Task 3: Customer Churn Prediction

- Records after cleaning: 7,021.
- Training records: 5,616.
- Test records: 1,405.
- Churn rate: approximately 26.45%.
- Best model selected using training-set cross-validation: XGBoost.
- XGBoost cross-validation ROC-AUC: 0.8500.
- XGBoost test ROC-AUC: 0.8448.
- XGBoost test churn recall: 0.7876.

**Datheon's target:** ROC-AUC greater than 0.80.

The reported XGBoost test ROC-AUC of 0.8448 exceeds this target.

### Task 4: NLP Sentiment Classification

- Training and evaluation corpus: NLTK Movie Reviews.
- Total corpus size: 2,000 labelled reviews.
- Test set: 400 reviews.
- Logistic Regression accuracy: 84.3%.
- Multinomial Naive Bayes accuracy: 81.5%.

The Datheon-provided starter dataset is also evaluated separately. Its nine non-neutral labelled examples produced 6 correct predictions out of 9.

These external-check results are not the same as the 84.3% test accuracy reported for the NLTK corpus.

---

## 3. Project Details

### Task 1: Multi-Class Benchmark Classification Pipeline

**Objective:** Build and evaluate supervised classification models using the Iris dataset.

**Dataset:** Iris flower dataset.

Dataset source: [Seaborn Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)

**Implementation:**

- Load and inspect the Iris dataset using Pandas.
- Perform exploratory data analysis.
- Visualize feature relationships using pairplots.
- Split the data into training and testing sets using an 80/20 split.
- Normalize numerical features using `StandardScaler`.
- Build a Logistic Regression classifier.
- Build a Decision Tree classifier.
- Evaluate both models using classification metrics.
- Generate a confusion matrix heatmap.
- Visualize model comparisons and decision boundaries.

**Models:**

- Logistic Regression
- Decision Tree Classifier

**Evaluation:**

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Five-fold cross-validation

**Main result:** Decision Tree achieved 96.7% test accuracy, while Logistic Regression achieved 93.3%.

**Deliverable:** `task-1-classification/classification.ipynb`

### Task 3: Predictive Customer Churn Classifier

**Objective:** Predict customer churn using ensemble learning while addressing class imbalance and evaluating model discrimination.

**Dataset:** IBM Telco Customer Churn.

Dataset source: [IBM Telco Customer Churn Dataset](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv)

**Implementation:**

- Load and inspect telecom customer records.
- Convert `TotalCharges` into a numerical feature.
- Handle missing and invalid values.
- Separate training and testing data.
- Encode categorical features using `OneHotEncoder`.
- Handle class imbalance through class weighting and a separate SMOTE-based pipeline.
- Train Random Forest and XGBoost classifiers.
- Tune model hyperparameters using cross-validation.
- Compare model performance.
- Plot ROC curves and confusion matrices.
- Extract and visualize the ten most influential features.

**Models:**

- Random Forest
- XGBoost
- Random Forest with SMOTE

**Evaluation:**

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cross-validation ROC-AUC
- Confusion matrix
- Feature importance

**Best model:** Tuned XGBoost.

| Metric | XGBoost test result |
|---|---:|
| Accuracy | 75.73% |
| Precision | 52.79% |
| Recall | 78.76% |
| F1-score | 63.21% |
| ROC-AUC | 84.48% |

The XGBoost model achieved a test ROC-AUC above Datheon's stated 80% target.

**Important implementation detail:** Preprocessing and resampling are incorporated into pipelines to avoid fitting transformations or applying SMOTE before the training/test separation.

**Deliverable:** `task-3-customer-churn/customer_churn.ipynb`

### Task 4: NLP Sentiment Classification on Customer Reviews

**Objective:** Build a text classification pipeline that predicts positive or negative sentiment from review text.

**Datasets:**

1. NLTK Movie Reviews corpus.
2. Datheon-provided sentiment review starter dataset.

**Dataset sources:**

- [NLTK Movie Reviews](https://www.nltk.org/nltk_data/)
- [Datheon Sentiment Reviews](https://www.datheon.in/datasets/sentiment_reviews.csv)

**Dataset consideration:**

The Datheon starter file contains only ten records, including one neutral example. This is insufficient for a meaningful train/test evaluation of a binary sentiment classifier.

Therefore, the main model training and evaluation use the NLTK Movie Reviews corpus, containing 2,000 labelled movie reviews. The Datheon starter file is treated as a separate external prediction check.

**Implementation:**

- Load and inspect the text corpus.
- Clean review text.
- Remove HTML tags and punctuation.
- Normalize text and remove stopwords.
- Apply text preprocessing.
- Convert reviews into numerical feature vectors using TF-IDF.
- Configure `TfidfVectorizer` with a maximum of 5,000 features.
- Train Multinomial Naive Bayes.
- Train Logistic Regression.
- Compare model performance.
- Generate confusion matrices and evaluation charts.
- Implement a reusable `predict_sentiment()` function.
- Test custom review inputs.

**Models:**

- Multinomial Naive Bayes
- Logistic Regression

**Evaluation:**

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Model comparison

**Main results:**

- Logistic Regression: 84.3% test accuracy.
- Multinomial Naive Bayes: 81.5% test accuracy.
- Datheon external starter check: 6 correct predictions out of 9 non-neutral examples.

**Deliverable:** `task-4-nlp-sentiment/sentiment_analysis.ipynb`

---

## 4. Repository Structure

```text
datheon-machine-learning-internship/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── docs/
│   ├── GITHUB_SETUP.md
│   └── SUBMISSION_CHECKLIST.md
│
├── linkedin/
│   ├── project_walkthrough.md
│   └── linkedin_post.md
│
├── reports/
│   └── final_internship_report.md
│
├── scripts/
│   ├── download_data.py
│   ├── run_all_notebooks.py
│   └── predict_sentiment_cli.py
│
├── src/
│   └── datheon_ml/
│       ├── iris.py
│       ├── churn.py
│       └── sentiment.py
│
├── task-1-classification/
│   ├── README.md
│   ├── classification.ipynb
│   ├── data/
│   ├── models/
│   └── results/
│
├── task-3-customer-churn/
│   ├── README.md
│   ├── customer_churn.ipynb
│   ├── data/
│   ├── models/
│   └── results/
│
├── task-4-nlp-sentiment/
│   ├── README.md
│   ├── sentiment_analysis.ipynb
│   ├── data/
│   ├── models/
│   └── results/
│
├── tests/
│   ├── test_churn.py
│   ├── test_classification.py
│   ├── test_notebooks.py
│   └── test_sentiment.py
│
├── .gitignore
├── LICENSE
├── README.md
├── pytest.ini
└── requirements.txt
```

### Main Components

| Directory | Purpose |
|---|---|
| `src/datheon_ml/` | Reusable machine learning and preprocessing code |
| `task-1-classification/` | Iris classification notebook, models, data and results |
| `task-3-customer-churn/` | Customer churn notebook, models, data and results |
| `task-4-nlp-sentiment/` | NLP notebook, models, data and results |
| `tests/` | Automated project tests |
| `scripts/` | Data download, notebook execution and prediction utilities |
| `reports/` | Consolidated project report |
| `docs/` | Setup and submission documentation |
| `linkedin/` | Project walkthrough and demo post drafts |
| `.github/workflows/` | Continuous integration configuration |

---

## 5. Technologies and Libraries

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations |
| Pandas | Data manipulation and analysis |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning, preprocessing and evaluation |
| XGBoost | Gradient-boosted decision trees |
| imbalanced-learn | Class imbalance handling and SMOTE |
| NLTK | Natural language processing and movie review corpus |
| SciPy | Scientific computing |
| Joblib | Model serialization |
| Jupyter Notebook | Interactive development and documentation |
| Pytest | Automated testing |
| Git and GitHub | Version control and project hosting |

---

## 6. Installation and Setup

### Requirements

Python and the dependencies listed in `requirements.txt` are required.

The following environment was used for the successful full test run:

| Component | Verified version |
|---|---:|
| Python | 3.14.3 |
| NumPy | 2.5.4 |
| Pandas | 3.1.1 |
| Scikit-learn | 1.9.1 |
| XGBoost | 3.3.0 |
| imbalanced-learn | 0.14.2 |
| Pytest | 9.2.0 |

Other dependency versions are specified in `requirements.txt`.

### Step 1: Clone the repository

```bash
git clone https://github.com/salikhussain71-code/datheon-machine-learning-internship.git
```

### Step 2: Open the project directory

```bash
cd datheon-machine-learning-internship
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Install dependencies

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS/Linux:**

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### Step 5: Download the datasets and required NLTK resources

```bash
python scripts/download_data.py
```

This script is intended to download missing project data and required NLTK resources.

### Step 6: Run the automated tests

**Windows PowerShell:**

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

**macOS/Linux:**

```bash
.venv/bin/python -m pytest -v
```

### Step 7: Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the relevant task directory and run its notebook.

---

## 7. Automated Testing

The repository includes a Pytest test suite covering the shared machine learning code, saved models, notebook implementation, and prediction functionality.

**Latest verified test result:**

```text
22 passed, 0 failed
```

The full test suite was run using:

- Python 3.14.3
- Scikit-learn 1.9.1
- XGBoost 3.3.0
- Pytest 9.2.0

One XGBoost model-serialization warning appeared when loading the saved churn model. The associated test passed, but the warning indicates that the serialized model was created with an older XGBoost version or serialization format. This should be considered when reproducing or migrating the saved model.

The project also includes a GitHub Actions workflow for automated testing.

---

## 8. Reproducibility and Machine Learning Practices

The notebooks and shared code are organized around reproducible machine learning workflows.

Practices used include:

- Fixed random states where configured.
- Stratified splitting for classification tasks where appropriate.
- Separation of training and testing data.
- Feature preprocessing within machine learning pipelines.
- Training-set cross-validation for model selection.
- Evaluation using held-out test data.
- Separate evaluation of the Datheon NLP starter examples.
- Saved models and generated evaluation artifacts.
- Automated tests for important project functionality.

For the customer churn task, the final model selection is based on cross-validation using training data rather than choosing a model based on test-set performance.

---

## 9. Generated Results and Artifacts

The task directories contain saved models and evaluation outputs.

### Classification results

Examples include:

- Confusion matrix.
- Pairplot.
- Correlation heatmap.
- Model comparison.
- Decision tree visualization.
- Decision boundary visualization.
- Evaluation metrics.

### Customer churn results

Examples include:

- ROC curve.
- Confusion matrix.
- Feature importance chart.
- Model comparison.
- Additional churn analysis plots.
- Evaluation metrics.

### NLP sentiment results

Examples include:

- Confusion matrix.
- Sentiment distribution.
- Model comparison.
- Top-word visualization.
- Evaluation metrics.

The corresponding model artifacts are stored in each task's `models/` directory.

---

## 10. Command-Line Sentiment Prediction

The repository includes a command-line utility for testing the saved sentiment classifier.

Example:

```bash
python scripts/predict_sentiment_cli.py "This product is amazing. I really love it."
```

The project also provides a reusable `predict_sentiment()` helper for custom text predictions.

---

## 11. Limitations and Scope

These projects demonstrate supervised machine learning workflows and model evaluation. They should be considered educational and portfolio projects, not production-deployed systems.

Current limitations include:

- No production model monitoring or automated drift detection.
- No comprehensive fairness audit.
- No production deployment or live customer integration.
- No guarantee that model performance will generalize to new datasets.
- The Datheon sentiment starter dataset is too small for reliable standalone model evaluation.
- The saved XGBoost model produces a serialization compatibility warning in the tested environment.

Model performance depends on dataset quality, preprocessing, sampling, feature selection, and evaluation methodology.

---

## 12. Official Internship Requirement Mapping

The following table maps the repository to the requirements published on the Datheon program page.

| Official requirement | Project evidence |
|---|---|
| Complete at least 2 or 3 predictive modelling tasks | Tasks 1, 3 and 4 |
| Classification notebook and evaluation charts | `task-1-classification/` |
| Customer churn notebook with ROC-AUC above 80% | `task-3-customer-churn/` |
| NLP sentiment classification notebook | `task-4-nlp-sentiment/` |
| Public GitHub repository | This repository |
| Documented README and installation instructions | This file |
| LinkedIn project walkthrough tagging @Datheon | Draft materials in `linkedin/`; public publication is a separate step |

The repository provides the technical work and documentation. A public LinkedIn post tagging @Datheon must be published separately to satisfy that stated completion criterion.

The repository's tests, reusable modules, CI configuration and additional documentation are independent project improvements, not claims that Datheon specifically requires those additions.

---

## 13. License and Dataset Attribution

This project is released under the MIT License. See [LICENSE](LICENSE).

Dataset sources:

- Iris: [Seaborn Data Repository](https://github.com/mwaskom/seaborn-data)
- Telecom Customer Churn: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d)
- Movie Reviews: [NLTK Movie Reviews Corpus](https://www.nltk.org/nltk_data/)
- Sentiment Starter Dataset: [Datheon](https://www.datheon.in/datasets/sentiment_reviews.csv)

Dataset ownership and licensing remain with their respective original providers. Refer to the original dataset sources for applicable terms.

---

## 14. Acknowledgement

This repository was developed as part of the learning and practical project work associated with the **Datheon Machine Learning & Data Science Internship**.

The work focuses on building a practical foundation in data preprocessing, supervised learning, model evaluation, NLP, reproducibility, and software development practices.

**Project Author:** Salik Hussain

**GitHub:** https://github.com/salikhussain71-code

**Project Repository:** https://github.com/salikhussain71-code/datheon-machine-learning-internship

**Program:** https://www.datheon.in/training/catalog/machine-learning-and-data-science-internship

---

*Three machine learning projects. Reproducible notebooks. Evaluated models. Documented results.*
