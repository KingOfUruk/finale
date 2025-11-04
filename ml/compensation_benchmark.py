"""
Compensation benchmarking with explainable models.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    SHAP_AVAILABLE = False

from . import MODELS_DIR
from .data_loader import DataLoader

MODEL_NAME = "compensation_benchmark.pkl"


def prepare_dataset(
    loader: Optional[DataLoader] = None, start_year: Optional[int] = None
) -> pd.DataFrame:
    loader = loader or DataLoader()
    payroll = loader.fetch_payroll(start_year=start_year)
    if payroll.empty:
        return pd.DataFrame()

    # Focus on salaire net + primes principales
    salary_sources = {"SNET", "SBRUT", "SBASE", "SDU", "SCNG"}
    df = payroll.copy()
    df["montant"] = pd.to_numeric(df["montant"], errors="coerce").fillna(0)
    df = df[df["montant"] > 0]
    df = df[df["source"].str.upper().isin(salary_sources)]

    agg = (
        df.groupby("mat_pers")
        .agg(
            salaire_annuel=("montant", "sum"),
            salaire_moyen_mensuel=("montant", "mean"),
            operations=("id_fact", "count"),
            lib_cat=("lib_cat", "last"),
            service_libelle=("service_libelle", "last"),
            age=("age", "last"),
            anciennete=("anciennete", "last"),
        )
        .reset_index()
    )
    agg["age"] = agg["age"].fillna(agg["age"].median())
    agg["anciennete"] = agg["anciennete"].fillna(agg["anciennete"].median())
    agg = agg.fillna({"lib_cat": "Non classé", "service_libelle": "Service inconnu"})
    return agg


def train_model(
    data: pd.DataFrame, model_path: Optional[Path] = None
) -> Tuple[Pipeline, dict]:
    if data.empty:
        raise ValueError("Payroll dataset is empty, cannot train compensation benchmark model.")

    features = data.drop(columns=["salaire_annuel", "mat_pers"])
    target = data["salaire_annuel"]

    categorical = ["lib_cat", "service_libelle"]
    numeric = [col for col in features.columns if col not in categorical]

    preproc = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ],
        remainder="drop",
    )

    regressor = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, max_depth=12
    )
    pipeline = Pipeline([("preprocessor", preproc), ("model", regressor)])

    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)

    metrics = {
        "mae": float(mean_absolute_error(y_val, preds)),
        "r2": float(r2_score(y_val, preds)),
    }

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        shap_values = explainer.shap_values(
            pipeline.named_steps["preprocessor"].transform(X_val)
        )
        metrics["shap_values_sample"] = shap_values[: min(100, len(shap_values))]
        metrics["feature_names"] = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    else:
        perm = permutation_importance(
            pipeline, X_val, y_val, n_repeats=10, random_state=42, n_jobs=-1
        )
        metrics["feature_importances"] = {
            name: float(importance)
            for name, importance in zip(
                pipeline.named_steps["preprocessor"].get_feature_names_out(),
                perm.importances_mean,
            )
        }

    output_path = model_path or MODELS_DIR / MODEL_NAME
    joblib.dump({"model": pipeline, "metrics": metrics}, output_path)
    return pipeline, metrics


def load_model(model_path: Optional[Path] = None) -> Pipeline:
    payload = joblib.load(model_path or MODELS_DIR / MODEL_NAME)
    return payload["model"]


def benchmark_services(
    data: pd.DataFrame, model: Optional[Pipeline] = None
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    model = model or load_model()
    features = data.drop(columns=["salaire_annuel", "mat_pers"])
    preds = model.predict(features)

    result = data[["mat_pers", "service_libelle", "lib_cat", "salaire_annuel"]].copy()
    result["salaire_attendu"] = preds
    result["ecart"] = result["salaire_annuel"] - result["salaire_attendu"]
    return result.sort_values(by="ecart", ascending=False)

