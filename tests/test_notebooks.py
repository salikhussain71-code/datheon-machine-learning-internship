import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    "task-1-classification/classification.ipynb": ["StandardScaler", "LogisticRegression", "DecisionTree", "classification_report", "predict_species"],
    "task-3-customer-churn/customer_churn.ipynb": ["TotalCharges", "OneHotEncoder", "SMOTE", "XGBClassifier", "roc_auc_score", "feature_importances_", "predict_churn"],
    "task-4-nlp-sentiment/sentiment_analysis.ipynb": ["TfidfVectorizer", "MultinomialNB", "LogisticRegression", "predict_sentiment", "max_features=5000"],
}


@pytest.mark.parametrize("rel,keywords", NOTEBOOKS.items())
def test_notebook_executed_cleanly_and_has_required_parts(rel, keywords):
    nb = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 15
    assert all(c.get("execution_count") for c in code_cells), "a code cell was never executed"
    errors = [o for c in code_cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, f"error output found in {rel}"
    text = "\n".join("".join(c["source"]) for c in nb["cells"]) + json.dumps(nb["cells"])
    for kw in keywords:
        assert kw in text, f"{kw} missing from {rel}"
