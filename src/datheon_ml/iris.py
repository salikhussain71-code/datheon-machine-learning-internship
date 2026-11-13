"""Task 1 helpers: Iris multi-class classification."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"


def load_iris(path: str | Path) -> pd.DataFrame:
    """Load the Iris CSV and validate the expected columns."""
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Iris file is missing columns: {sorted(missing)}")
    return df


def build_pipelines(random_state: int = 42) -> dict[str, Pipeline]:
    """Return the two required models, each wrapped with StandardScaler.

    Using a Pipeline means the scaler is fitted on training data only, so no
    information from the test set leaks into preprocessing.
    """
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(max_depth=3, random_state=random_state)),
            ]
        ),
    }


def predict_species(model: Pipeline, sepal_length: float, sepal_width: float,
                    petal_length: float, petal_width: float) -> dict:
    """Predict the species for one flower and return label + class probabilities."""
    row = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=FEATURES)
    label = model.predict(row)[0]
    probs = dict(zip(model.classes_, model.predict_proba(row)[0].round(4)))
    return {"prediction": str(label), "probabilities": {k: float(v) for k, v in probs.items()}}
