import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from datheon_ml.iris import FEATURES, TARGET, build_pipelines, load_iris, predict_species

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "task-1-classification"


@pytest.fixture(scope="module")
def data():
    df = load_iris(TASK / "data/iris.csv").drop_duplicates().reset_index(drop=True)
    return train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42, stratify=df[TARGET])


def test_dataset_shape_and_classes(data):
    X_train, X_test, y_train, y_test = data
    assert len(X_train) + len(X_test) == 149
    assert set(y_train) == {"setosa", "versicolor", "virginica"}


def test_pipelines_have_scaler_first_and_reach_good_accuracy(data):
    X_train, X_test, y_train, y_test = data
    for name, pipe in build_pipelines(42).items():
        assert pipe.steps[0][0] == "scaler"
        pipe.fit(X_train, y_train)
        assert pipe.score(X_test, y_test) >= 0.90, name


def test_scaler_is_fitted_on_training_data_only(data):
    X_train, X_test, y_train, _ = data
    pipe = build_pipelines(42)["Logistic Regression"].fit(X_train, y_train)
    assert np.allclose(pipe.named_steps["scaler"].mean_, X_train.mean().to_numpy())


def test_predict_species_output_format(data):
    X_train, _, y_train, _ = data
    pipe = build_pipelines(42)["Logistic Regression"].fit(X_train, y_train)
    out = predict_species(pipe, 5.0, 3.4, 1.5, 0.2)
    assert out["prediction"] == "setosa"
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-3


def test_saved_artifacts_exist_and_work():
    model = joblib.load(TASK / "models/best_model.joblib")
    assert predict_species(model, 6.9, 3.1, 5.8, 2.1)["prediction"] == "virginica"
    metrics = json.loads((TASK / "results/metrics.json").read_text())
    for m in metrics["models"].values():
        assert 0.85 <= m["Accuracy"] <= 1.0
    for png in ["confusion_matrix", "pairplot", "correlation_heatmap", "model_comparison"]:
        assert (TASK / f"results/{png}.png").stat().st_size > 5_000
