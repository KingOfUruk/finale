"""
Mass payroll forecasting utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import MODELS_DIR
from .data_loader import DataLoader

MODEL_NAME = "payroll_forecast.pkl"


def prepare_dataset(loader: Optional[DataLoader] = None) -> pd.DataFrame:
    loader = loader or DataLoader()
    payroll = loader.fetch_payroll()
    if payroll.empty:
        return pd.DataFrame()

    payroll["montant"] = pd.to_numeric(payroll["montant"], errors="coerce").fillna(0)
    df = payroll.groupby(
        ["annee", "mois", "service_libelle", "lib_cat", "source"]
    )["montant"].sum().reset_index()
    df["date_ref"] = pd.to_datetime(
        df["annee"].astype(str) + "-" + df["mois"].astype(str) + "-01"
    )
    df = df.sort_values("date_ref")
    pivot = df.pivot_table(
        index=["date_ref", "annee", "mois", "service_libelle"],
        columns="source",
        values="montant",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    pivot["salaire_total"] = pivot.drop(
        columns=["date_ref", "annee", "mois", "service_libelle"]
    ).sum(axis=1)
    pivot["lag1"] = pivot.groupby("service_libelle")["salaire_total"].shift(1).fillna(0)
    pivot["lag2"] = pivot.groupby("service_libelle")["salaire_total"].shift(2).fillna(0)
    pivot["lag3"] = pivot.groupby("service_libelle")["salaire_total"].shift(3).fillna(0)
    pivot = pivot.dropna(subset=["salaire_total"])
    return pivot


def train_model(
    data: pd.DataFrame, model_path: Optional[Path] = None
) -> Tuple[Pipeline, float]:
    if data.empty:
        raise ValueError("Payroll dataset is empty, cannot train payroll forecast model.")

    features = data.drop(columns=["salaire_total", "date_ref"])
    target = data["salaire_total"]

    categorical = ["service_libelle"]
    numeric = [col for col in features.columns if col not in categorical]

    preproc = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )

    regressor = ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000)
    pipeline = Pipeline([("preprocessor", preproc), ("model", regressor)])

    tscv = TimeSeriesSplit(n_splits=5)
    maes = []
    for train_idx, val_idx in tscv.split(features):
        X_train, X_val = features.iloc[train_idx], features.iloc[val_idx]
        y_train, y_val = target.iloc[train_idx], target.iloc[val_idx]
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        maes.append(mean_absolute_error(y_val, preds))

    final_mae = float(sum(maes) / len(maes))
    pipeline.fit(features, target)

    output_path = model_path or MODELS_DIR / MODEL_NAME
    joblib.dump({"model": pipeline, "mae": final_mae}, output_path)

    return pipeline, final_mae


def load_model(model_path: Optional[Path] = None) -> Pipeline:
    payload = joblib.load(model_path or MODELS_DIR / MODEL_NAME)
    return payload["model"]


def forecast(
    future_df: pd.DataFrame, model: Optional[Pipeline] = None
) -> pd.DataFrame:
    if future_df.empty:
        return pd.DataFrame()
    model = model or load_model()
    features = future_df.copy()
    preds = model.predict(features)
    result = features[["service_libelle", "annee", "mois"]].copy()
    result["salaire_prevision"] = preds.clip(min=0).round(2)
    return result

