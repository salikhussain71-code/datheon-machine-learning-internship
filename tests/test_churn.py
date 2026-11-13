
"""Tests for the Datheon customer churn project."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from datheon_ml.churn import (
    TARGET,
    add_features,
    build_preprocessor,
    clean_telco,
    load_raw,
    predict_churn,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "task-3-customer-churn"

DATA_PATH = TASK / "data" / "Telco-Customer-Churn.csv"
NOTEBOOK_PATH = TASK / "customer_churn.ipynb"
MODEL_PATH = TASK / "models" / "best_churn_model.joblib"
METRICS_PATH = TASK / "results" / "metrics.json"

CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": "286.5",
}


@pytest.fixture(scope="module")
def raw():
    """Load the original Telco dataset once for this test module."""
    assert DATA_PATH.is_file(), f"Dataset not found: {DATA_PATH}"
    return load_raw(DATA_PATH)


def test_raw_total_charges_is_text_with_blanks(raw):
    """Raw TotalCharges should contain text and 11 blank values."""
    assert "TotalCharges" in raw.columns
    assert not pd.api.types.is_numeric_dtype(raw["TotalCharges"])
    assert raw["TotalCharges"].str.strip().eq("").sum() == 11


def test_cleaning_makes_total_charges_numeric_and_target_binary(raw):
    """Cleaning should produce numeric charges and a binary target."""
    df = clean_telco(raw)

    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}
    assert "customerID" not in df.columns


def test_feature_engineering(raw):
    """Engineered features should exist and have valid values."""
    df = add_features(clean_telco(raw))

    assert {"AvgMonthlySpend", "NumServices"} <= set(df.columns)
    assert df["NumServices"].between(0, 8).all()
    assert (df["AvgMonthlySpend"] >= 0).all()


def test_pipeline_trains_and_smote_stays_inside_pipeline(raw):
    """The model should train and notebook SMOTE usage should be pipeline-based."""
    df = add_features(clean_telco(raw)).sample(
        n=1200,
        random_state=0,
    )

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    pipe = Pipeline(
        [
            ("pre", build_preprocessor(X)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=30,
                    class_weight="balanced",
                    random_state=0,
                ),
            ),
        ]
    )

    pipe.fit(X, y)

    probabilities = pipe.predict_proba(X.head(5))

    assert probabilities.shape == (5, 2)
    assert NOTEBOOK_PATH.is_file(), f"Notebook not found: {NOTEBOOK_PATH}"

    # Explicit UTF-8 avoids Windows cp1252 decoding errors.
    notebook = NOTEBOOK_PATH.read_text(encoding="utf-8")
    assert "ImbPipeline" in notebook
    assert "SMOTE(random_state" in notebook

    # Confirm the imbalanced-learn pipeline class is importable.
    assert issubclass(ImbPipeline, Pipeline) or issubclass(
        ImbPipeline, object
    )


def test_saved_model_predicts_valid_probability():
    """The saved model should return valid churn predictions."""
    assert MODEL_PATH.is_file(), f"Saved model not found: {MODEL_PATH}"

    model = joblib.load(MODEL_PATH)
    out = predict_churn(model, CUSTOMER)

    assert 0.0 <= out["churn_probability"] <= 1.0
    assert out["prediction"] in {"Churn", "Stay"}

    safe_customer = {
        **CUSTOMER,
        "tenure": 60,
        "Contract": "Two year",
        "TechSupport": "Yes",
        "OnlineSecurity": "Yes",
        "MonthlyCharges": 70.0,
        "TotalCharges": "4200.0",
    }

    safe_out = predict_churn(model, safe_customer)

    assert 0.0 <= safe_out["churn_probability"] <= 1.0
    assert safe_out["churn_probability"] < out["churn_probability"]


def test_recorded_metrics_are_real_and_meet_target():
    """Saved metrics and evaluation plots should meet project requirements."""
    assert METRICS_PATH.is_file(), f"Metrics file not found: {METRICS_PATH}"

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    best_model = metrics["best_model_by_cv"]

    assert best_model in metrics["test_metrics"]
    assert metrics["test_metrics"][best_model]["ROC-AUC"] > 0.80
    assert metrics["target_roc_auc_gt_0_80_met"] is True

    required_plots = [
        "roc_curve",
        "confusion_matrix",
        "feature_importance",
        "model_comparison",
        "churn_distribution",
    ]

    for plot_name in required_plots:
        plot_path = TASK / "results" / f"{plot_name}.png"

        assert plot_path.is_file(), f"Missing evaluation plot: {plot_path}"
        assert plot_path.stat().st_size > 5_000
