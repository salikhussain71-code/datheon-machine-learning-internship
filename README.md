# Datheon Machine Learning & Data Science Internship — Portfolio

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-pipelines-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-boosting-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

Three end-to-end supervised machine-learning projects completed for the **Datheon Machine Learning & Data Science Internship** (Standard Internship, 4 weeks). Every notebook runs from the first cell to the last, every number below is produced by the notebooks, and a test suite checks the code and the saved results.

**Author:** Salik Hussain · **GitHub:** [salikhussain71-code](https://github.com/salikhussain71-code)

## Results at a glance

| Task | Problem | Models | Main result (held-out test set) |
|---|---|---|---|
| [**Task 1** — Classification](task-1-classification) | Iris species (3 classes) | Logistic Regression, Decision Tree | Accuracy **93.3 %** (LR) / **96.7 %** (DT) on 30 test flowers; 5-fold CV favours LR (96.6 % vs 93.2 %) |
| [**Task 3** — Customer churn](task-3-customer-churn) | Telco churn (imbalanced, 26.5 % churn) | Random Forest, XGBoost (+ SMOTE comparison) | **ROC-AUC 0.845** (XGBoost) — target > 0.80 met; churn recall 78.8 % |
| [**Task 4** — NLP sentiment](task-4-nlp-sentiment) | Review polarity (pos / neg) | TF-IDF + Multinomial NB, TF-IDF + Logistic Regression | Accuracy **84.3 %** (LR) / 81.5 % (NB) on 400 test reviews |

Task 2 (housing regression) is not included: the official rule is *"complete at least 2 or 3 tasks"*, and Tasks 1, 3 and 4 give three.

## Important honesty notes

* **Task 4 dataset.** The official `sentiment_reviews.csv` from Datheon contains only **10 rows** (1 neutral), which is too small to train or evaluate a model. The notebook therefore trains on the public **NLTK `movie_reviews`** corpus (2,000 labelled reviews; the task allows "customer product *or movie* review records") and uses Datheon's file as a separate 9-row external check (6/9 correct). Nothing is hidden — see [`task-4-nlp-sentiment/README.md`](task-4-nlp-sentiment/README.md).
* The churn ROC-AUC target is reported as measured; the test set was never used for tuning or model selection (selection uses cross-validation on the training data).
* These are notebook-grade models. They are **not** production-ready (no drift monitoring, no fairness audit, default decision threshold).

## Repository structure

```
datheon-machine-learning-internship/
├── README.md                     <- you are here
├── requirements.txt
├── LICENSE
├── pytest.ini
├── .gitignore
├── .github/workflows/ci.yml      <- runs the tests on every push
├── src/datheon_ml/               <- shared, tested helper code
│   ├── iris.py  churn.py  sentiment.py
├── task-1-classification/
│   ├── README.md
│   ├── classification.ipynb
│   ├── data/iris.csv
│   ├── models/                   <- saved pipelines (.joblib)
│   └── results/                  <- confusion_matrix.png, pairplot.png, correlation_heatmap.png,
│                                    model_comparison.png, decision_tree.png, decision_boundary.png, metrics.json
├── task-3-customer-churn/
│   ├── README.md
│   ├── customer_churn.ipynb
│   ├── data/Telco-Customer-Churn.csv
│   ├── models/
│   └── results/                  <- roc_curve.png, confusion_matrix.png, feature_importance.png,
│                                    model_comparison.png, churn_*.png, metrics.json
├── task-4-nlp-sentiment/
│   ├── README.md
│   ├── sentiment_analysis.ipynb
│   ├── data/sentiment_reviews.csv
│   ├── models/
│   └── results/                  <- confusion_matrix.png, sentiment_distribution.png,
│                                    model_comparison.png, top_words.png, metrics.json
├── scripts/                      <- download_data.py, run_all_notebooks.py, predict_sentiment_cli.py
├── tests/                        <- 22 pytest tests (code, saved models, notebooks)
├── reports/final_internship_report.md
├── linkedin/                     <- project_walkthrough.md, linkedin_post.md
└── docs/                         <- GITHUB_SETUP.md, SUBMISSION_CHECKLIST.md
```

## Quick start (Windows PowerShell; on macOS/Linux use `source .venv/bin/activate`)

```powershell
git clone https://github.com/salikhussain71-code/datheon-machine-learning-internship.git
cd datheon-machine-learning-internship
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name python3
python scripts/download_data.py          # only downloads files that are missing + NLTK data
pytest                                   # 22 tests should pass
jupyter notebook                         # open a task folder and "Run All"
```

To rebuild every figure, model and metrics file from scratch: `python scripts/run_all_notebooks.py` (about 5–10 minutes; Task 3 tuning is the slow step).

## Test the sentiment model from the command line

```powershell
python scripts/predict_sentiment_cli.py "This product is amazing. I really love it."
```

## How the work maps to the official task page

| Official requirement | Where it is met |
|---|---|
| Complete 2–3 predictive modelling tasks | Tasks 1, 3 and 4 |
| Jupyter notebooks (.ipynb) with evaluation charts | the three notebooks + `results/*.png` |
| Task 1: StandardScaler, 80/20 split, LR + Decision Tree, accuracy/precision/recall/F1, confusion-matrix heatmap, pairplot, decision-boundary plot | `classification.ipynb` |
| Task 3: `TotalCharges` numeric fix, OneHotEncoder, SMOTE or class weights, RF + XGBoost with cross-validation, ROC-AUC > 0.80, top-10 feature importance, confusion matrix | `customer_churn.ipynb` |
| Task 4: HTML/punctuation/stopword/lemma cleaning, `TfidfVectorizer(max_features=5000)`, MultinomialNB + Logistic Regression, `predict_sentiment()`, custom-input tester | `sentiment_analysis.ipynb` |
| Public GitHub repository with README (installation and architecture) | this repository |
| LinkedIn project walkthrough tagging @Datheon | [`linkedin/`](linkedin) |

Source of requirements: the official [Datheon task pack](https://www.datheon.in/training/catalog/machine-learning-and-data-science-internship). The folder layout, tests and CI file are this project's own professional additions, not Datheon requirements.

## Reproducibility

Fixed `random_state=42` everywhere; stratified splits; preprocessing (scaling, encoding, TF-IDF, SMOTE) lives inside pipelines so it is fitted on training data only; model choice is made with cross-validation, never with the test set. Tested with Python 3.12, scikit-learn 1.9, XGBoost 3.4, imbalanced-learn 0.14, pandas 3.0.

## Licence

MIT — see [LICENSE](LICENSE). Datasets keep their own licences: Iris (seaborn-data), IBM Telco Customer Churn (IBM sample data), NLTK movie_reviews (Pang & Lee), Datheon starter file (Datheon).
