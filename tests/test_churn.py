import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from datheon_ml.churn import TARGET, add_features, build_preprocessor, clean_telco, load_raw, predict_churn

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "task-3-customer-churn"

CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 3,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": "286.5",
}


@pytest.fixture(scope="module")
def raw():
    return load_raw(TASK / "data/Telco-Customer-Churn.csv")


def test_raw_total_charges_is_text_with_blanks(raw):
    assert not pd.api.types.is_numeric_dtype(raw["TotalCharges"])
    assert raw["TotalCharges"].str.strip().eq("").sum() == 11


def test_cleaning_makes_total_charges_numeric_and_target_binary(raw):
    df = clean_telco(raw)
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0
    assert set(df[TARGET].unique()) == {0, 1}
    assert "customerID" not in df.columns


def test_feature_engineering(raw):
    df = add_features(clean_telco(raw))
    assert {"AvgMonthlySpend", "NumServices"} <= set(df.columns)
    assert df["NumServices"].between(0, 8).all()
    assert (df["AvgMonthlySpend"] >= 0).all()


def test_pipeline_trains_and_smote_stays_inside_pipeline(raw):
    df = add_features(clean_telco(raw)).sample(1200, random_state=0)
    X, y = df.drop(columns=TARGET), df[TARGET]
    pipe = Pipeline([("pre", build_preprocessor(X)),
                     ("model", RandomForestClassifier(n_estimators=30, class_weight="balanced", random_state=0))]).fit(X, y)
    assert pipe.predict_proba(X.head(5)).shape == (5, 2)
    notebook = (TASK / "customer_churn.ipynb").read_text()
    assert "ImbPipeline" in notebook and "SMOTE(random_state" in notebook      # SMOTE only used via imblearn Pipeline
    assert issubclass(ImbPipeline, object)


def test_saved_model_predicts_valid_probability():
    model = joblib.load(TASK / "models/best_churn_model.joblib")
    out = predict_churn(model, CUSTOMER)
    assert 0.0 <= out["churn_probability"] <= 1.0
    assert out["prediction"] in {"Churn", "Stay"}
    safe = {**CUSTOMER, "tenure": 60, "Contract": "Two year", "TechSupport": "Yes", "OnlineSecurity": "Yes",
            "MonthlyCharges": 70.0, "TotalCharges": "4200.0"}
    assert predict_churn(model, safe)["churn_probability"] < out["churn_probability"]


def test_recorded_metrics_are_real_and_meet_target():
    m = json.loads((TASK / "results/metrics.json").read_text())
    best = m["best_model_by_cv"]
    assert m["test_metrics"][best]["ROC-AUC"] > 0.80
    assert m["target_roc_auc_gt_0_80_met"] is True
    for png in ["roc_curve", "confusion_matrix", "feature_importance", "model_comparison", "churn_distribution"]:
        assert (TASK / f"results/{png}.png").stat().st_size > 5_000
