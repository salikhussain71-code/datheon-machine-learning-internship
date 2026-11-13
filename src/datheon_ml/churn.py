"""Task 3 helpers: Telco customer churn prediction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "AvgMonthlySpend", "NumServices"]


def load_raw(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw IBM Telco table.

    * drops the customer id (no predictive meaning)
    * converts TotalCharges (stored as text) to numeric; the 11 blank values all
      belong to customers with tenure == 0 who have not been billed yet, so they
      are set to 0.0
    * maps the target Churn to 1 (Yes) / 0 (No)
    * removes exact duplicate rows
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].replace(" ", np.nan), errors="coerce")
    blank_mask = out["TotalCharges"].isna()
    out.loc[blank_mask & (out["tenure"] == 0), "TotalCharges"] = 0.0
    out = out.dropna(subset=["TotalCharges"])
    out[TARGET] = out[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    out = out.drop(columns=[ID_COL], errors="ignore").drop_duplicates().reset_index(drop=True)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: average monthly spend and number of optional services."""
    out = df.copy()
    out["AvgMonthlySpend"] = out["TotalCharges"] / out["tenure"].clip(lower=1)
    out["NumServices"] = (out[SERVICE_COLS] == "Yes").sum(axis=1)
    return out


def split_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in NUMERIC if c in X.columns]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """OneHotEncoder for categoricals, StandardScaler for numeric columns."""
    numeric, categorical = split_columns(X)
    num_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [
            ("num", num_step, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def predict_churn(model, customer: dict) -> dict:
    """Predict churn for a single customer given as a dict of raw Telco columns."""
    row = pd.DataFrame([customer])
    row["TotalCharges"] = pd.to_numeric(row["TotalCharges"], errors="coerce").fillna(0.0)
    row = add_features(row)
    proba = float(model.predict_proba(row)[0, 1])
    return {"churn_probability": round(proba, 4), "prediction": "Churn" if proba >= 0.5 else "Stay"}
