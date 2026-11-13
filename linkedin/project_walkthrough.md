# Project walkthrough script (2–3 minute screen recording, or a 6-slide LinkedIn carousel)

**Goal:** show proof of work so Datheon can verify it publicly. Record your screen with the notebooks open, or export the slides below as images.

| # | Slide / scene | What to show | What to say (one sentence) |
|---|---|---|---|
| 1 | Title | Repo name, your name, "@Datheon ML & Data Science Internship" | "Three ML projects, one reproducible repository." |
| 2 | Task 1 — Iris | `results/pairplot.png` + `confusion_matrix.png` | "Scaled features in a pipeline, compared Logistic Regression and a Decision Tree: 93.3% and 96.7% accuracy on 30 test flowers." |
| 3 | Task 3 — Churn | `results/roc_curve.png` + `feature_importance.png` | "Handled 26% churn imbalance with class weights and SMOTE; tuned XGBoost reached ROC-AUC 0.845 and month-to-month contracts are the top driver." |
| 4 | Task 4 — Sentiment | Run `predict_sentiment()` on 2–3 live sentences in the notebook | "TF-IDF plus Logistic Regression reached 84.3% accuracy; here it is predicting my own sentences." |
| 5 | Engineering quality | Terminal: `pytest` → "22 passed"; repo tree | "No leakage, models picked by cross-validation, 22 automated tests." |
| 6 | Honesty + next steps | Limitations bullets | "The official sentiment file had only 10 rows, so I documented a public corpus; next: SHAP, threshold tuning, transformers." |

## Checklist before posting
- [ ] Repository is **public** and the README renders the images
- [ ] Notebooks open on GitHub with outputs visible
- [ ] Post tags **@Datheon** and includes the repo link
- [ ] Numbers in the post match `results/metrics.json`
- [ ] Copy the post URL into `reports/final_internship_report.md` and submit it to Datheon
