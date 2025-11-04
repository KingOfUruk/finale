"""
Staffing / attendance forecasting utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import MODELS_DIR
from .data_loader import DataLoader


MODEL_NAME = "staffing_forecast.pkl"


def _feature_engineering(attendance_df: pd.DataFrame) -> pd.DataFrame:
    df = attendance_df.copy()
    df = df.dropna(subset=["service_libelle", "date_jour"])
    df["date_jour"] = pd.to_datetime(df["date_jour"])
    df["annee"] = df["date_jour"].dt.year
    df["semaine"] = df["date_jour"].dt.isocalendar().week.astype(int)
    df["jour"] = df["date_jour"].dt.dayofweek
    df["is_weekend"] = df["jour"].isin([5, 6]).astype(int)
    df["duree_heure"] = df["duree_minutes"].fillna(0) / 60.0
    # aggregate per service + semaine
    weekly = (
        df.groupby(["service_libelle", "annee", "semaine"])
        .agg(
            employes_uniques=("mat_pers", "nunique"),
            total_pointages=("nbr_pointages", "sum"),
            total_heures=("duree_heure", "sum"),
            jours_pointes=("date_jour", "nunique"),
        )
        .reset_index()
    )
    weekly["semaine_sin"] = np.sin(2 * np.pi * weekly["semaine"] / 52)
    weekly["semaine_cos"] = np.cos(2 * np.pi * weekly["semaine"] / 52)
    return weekly


def get_dataset(loader: Optional[DataLoader] = None, start_year: Optional[int] = None) -> pd.DataFrame:
    loader = loader or DataLoader()
    attendance = loader.fetch_attendance(start_year=start_year)
    if attendance.empty:
        return pd.DataFrame()
    return _feature_engineering(attendance)


def train_model(
    data: pd.DataFrame, model_path: Optional[Path] = None
) -> Tuple[Pipeline, float]:
    if data.empty:
        raise ValueError("Attendance dataset is empty, cannot train staffing model.")

    features = data.drop(columns=["employes_uniques"])
    target = data["employes_uniques"]

    categorical = ["service_libelle"]
    numeric = [col for col in features.columns if col not in categorical]

    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )

    regressor = GradientBoostingRegressor(random_state=42)
    pipeline = Pipeline(steps=[("preprocessor", preproc), ("model", regressor)])

    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    mae = float(mean_absolute_error(y_val, preds))

    output_path = model_path or MODELS_DIR / MODEL_NAME
    joblib.dump({"model": pipeline, "mae": mae}, output_path)

    return pipeline, mae


def load_model(model_path: Optional[Path] = None) -> Pipeline:
    path = model_path or MODELS_DIR / MODEL_NAME
    payload = joblib.load(path)
    return payload["model"]


def forecast_staffing(
    forecast_df: pd.DataFrame, model: Optional[Pipeline] = None
) -> pd.DataFrame:
    if forecast_df.empty:
        return pd.DataFrame(columns=["service_libelle", "annee", "semaine", "employes_predits"])

    if model is None:
        model = load_model()

    features = forecast_df.copy()
    key_cols = ["service_libelle", "annee", "semaine"]
    preds = model.predict(features)
    result = features[key_cols].copy()
    result["employes_predits"] = preds.clip(min=0).round(2)
    return result

