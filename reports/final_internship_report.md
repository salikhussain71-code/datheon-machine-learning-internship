# Final Internship Report — Datheon Machine Learning & Data Science Internship

**Intern:** Salik Hussain  **Programme:** Standard Internship (1 month, 4 weeks), Computer Science & IT, intermediate
**Repository:** https://github.com/salikhussain71-code/datheon-machine-learning-internship
**LinkedIn walkthrough:** `<ADD YOUR LINKEDIN POST LINK HERE AFTER PUBLISHING>`
**Tasks completed:** Task 1, Task 3, Task 4 (the rule is "at least 2 or 3 tasks")

All figures below are copied from `results/metrics.json` files produced by the notebooks.

## 1. Internship overview
The internship asks for supervised predictive-modelling work: preprocessing datasets, training classification models and benchmarking them with clear metrics, delivered as clean Jupyter notebooks in a public GitHub repository, plus a public LinkedIn walkthrough tagging @Datheon.

## 2. Objectives
1. Build three complete ML pipelines (classification, imbalanced classification, NLP).
2. Prevent data leakage by keeping all preprocessing inside scikit-learn pipelines.
3. Report honest metrics, choose models with cross-validation, and document limitations.
4. Package the work professionally (README files, tests, saved models, figures).

## 3. Task 1 — methodology (Iris classification)
Loaded the seaborn-data Iris file, removed one duplicate row (149 rows), explored distributions with pairplots and a correlation heatmap, split 80 / 20 stratified (119 / 30), scaled with `StandardScaler` inside a `Pipeline`, trained Logistic Regression and a depth-3 Decision Tree, ran 5-fold CV on the training set, and evaluated once on the test set.

## 4. Task 1 — results
| Model | Accuracy | Precision | Recall | F1 | CV accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0.9333 | 0.9333 | 0.9333 | 0.9333 | 0.9656 |
| Decision Tree | 0.9667 | 0.9697 | 0.9667 | 0.9666 | 0.9319 |

Setosa was classified perfectly by both models; all errors were versicolor/virginica confusions (2 flowers for Logistic Regression, 1 for the tree). The test set is only 30 flowers, so the ranking of the two models is not statistically reliable.

## 5. Task 3 — methodology (customer churn)
IBM Telco data (7,043 rows). `TotalCharges` was converted from text to numbers; the 11 blanks all had `tenure = 0` and were set to 0.0. Customer id and 22 duplicate rows were dropped (7,021 rows, 26.45 % churn). Added `AvgMonthlySpend` and `NumServices`. One-hot encoding and scaling were placed in a `ColumnTransformer` inside the pipeline. Imbalance was handled with class weights (Random Forest), `scale_pos_weight` (XGBoost) and, for comparison, SMOTE inside an `imblearn` pipeline. Models were tuned with `RandomizedSearchCV` (ROC-AUC, 5-fold) and selected by cross-validated ROC-AUC.

## 6. Task 3 — results
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Random Forest (tuned) | 0.7601 | 0.5325 | 0.7715 | 0.6301 | 0.8433 | 0.8485 |
| **XGBoost (tuned)** | 0.7573 | 0.5279 | 0.7876 | 0.6321 | **0.8448** | 0.8500 |
| Random Forest + SMOTE | 0.7758 | 0.5831 | 0.5376 | 0.5594 | 0.8154 | 0.8244 |

The task-pack target of ROC-AUC > 0.80 was met on the untouched test set. XGBoost's confusion matrix: 771 true negatives, 262 false positives, 79 false negatives, 293 true positives. Strongest drivers: month-to-month contract, fibre-optic internet, missing tech support and online security, short tenure.

## 7. Task 4 — methodology (sentiment)
The Datheon starter file has only 10 rows, so it cannot support training or a meaningful test split. The NLTK `movie_reviews` corpus (2,000 labelled reviews) was used for training and the 80 / 20 benchmark (1,600 / 400), and the 9 positive/negative Datheon rows were kept as an external check. Cleaning: HTML removal, lowercasing, punctuation removal, tokenisation, stopword removal (negations kept), WordNet lemmatisation. Features: `TfidfVectorizer(max_features=5000)` inside a pipeline. Models: Multinomial Naive Bayes and Logistic Regression.

## 8. Task 4 — results
| Model | Accuracy | Precision | Recall | F1 | CV accuracy |
|---|---|---|---|---|---|
| Multinomial Naive Bayes | 0.8150 | 0.8316 | 0.7900 | 0.8103 | 0.8269 |
| **Logistic Regression** | 0.8425 | 0.8186 | 0.8800 | 0.8482 | 0.8431 |

On the 9 Datheon product reviews the selected model was right 6 times (66.7 %), which is anecdotal. A custom negated sentence ("The film was not good at all.") was misclassified, illustrating a known limitation of unigram TF-IDF.

## 9. Comparison of models
* Task 1: both models are near-ceiling; cross-validation favours Logistic Regression, the single test split favours the tree — a reminder that 30 samples are too few to separate them.
* Task 3: boosted trees and random forest are within 0.002 ROC-AUC of each other; class weighting clearly beat SMOTE here on recall (0.77–0.79 vs 0.54) and AUC.
* Task 4: Logistic Regression beats Naive Bayes by about 2.8 accuracy points and finds more positives (recall 0.88 vs 0.79).

## 10. Technical challenges
* Detecting that `TotalCharges` was text and deciding the correct fill value.
* Avoiding leakage (scaler, encoder, TF-IDF and SMOTE all inside pipelines).
* The official sentiment dataset being far too small — resolved transparently with a documented public corpus.
* Keeping negation words so "not good" is not read as "good".

## 11. What I learned
Pipelines and leakage prevention; stratified splitting and cross-validation; imbalance strategies and why accuracy misleads; ROC-AUC vs precision/recall trade-offs; TF-IDF; model selection without touching the test set; professional repository and testing practice.

## 12. Limitations
Small benchmarks (Iris, 30-sample test); churn precision about 53 % at the default 0.5 threshold; movie-review training data for a product-review use case; no calibration, drift, fairness or latency testing. The models are **not production-ready**.

## 13. GitHub links
* Repository: https://github.com/salikhussain71-code/datheon-machine-learning-internship
* Notebooks: `task-1-classification/classification.ipynb`, `task-3-customer-churn/customer_churn.ipynb`, `task-4-nlp-sentiment/sentiment_analysis.ipynb`

## 14. Conclusion
Three reproducible, tested pipelines were delivered with honest evaluation: Iris (93.3 % / 96.7 % accuracy), churn (ROC-AUC 0.845, above the 0.80 target) and sentiment (84.3 % accuracy). The main next steps are cost-based thresholds and SHAP for churn, and a larger in-domain review dataset with bigrams or a transformer for sentiment.
