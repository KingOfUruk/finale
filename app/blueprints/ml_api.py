from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Blueprint, jsonify, request

from ml.compensation_benchmark import benchmark_services, prepare_dataset as prepare_compensation_data, train_model as train_compensation_model
from ml.payroll_forecast import forecast as forecast_payroll, prepare_dataset as prepare_payroll_data, train_model as train_payroll_model
from ml.scenario_simulator import ScenarioInput, simulate_compensation_adjustment, simulate_payroll_impact
from ml.staffing_forecast import forecast_staffing, get_dataset as prepare_staffing_data, train_model as train_staffing_model
from ml import MODELS_DIR
from ml.data_loader import DataLoader

ml_bp = Blueprint("ml", __name__, url_prefix="/api/ml")


def _get_models_dir() -> Path:
    path = Path(MODELS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


@ml_bp.route("/train", methods=["POST"])
def trigger_training():
    """
    Lance l'entraînement de l'ensemble des modèles.
    Optionnellement, accepter un champ JSON {"start_year": 2020}.
    """
    payload = request.get_json(silent=True) or {}
    start_year = payload.get("start_year")
    loader = DataLoader()
    models_dir = _get_models_dir()

    summary = {}

    staffing_df = prepare_staffing_data(loader, start_year=start_year)
    if staffing_df.empty:
        summary["staffing"] = "aucune donnée"
    else:
        _, mae = train_staffing_model(staffing_df, model_path=models_dir / "staffing_forecast.pkl")
        summary["staffing"] = {"mae": mae, "rows": int(len(staffing_df))}

    compensation_df = prepare_compensation_data(loader, start_year=start_year)
    if compensation_df.empty:
        summary["compensation"] = "aucune donnée"
    else:
        _, metrics = train_compensation_model(compensation_df, model_path=models_dir / "compensation_benchmark.pkl")
        summary["compensation"] = {"mae": metrics["mae"], "r2": metrics["r2"], "rows": int(len(compensation_df))}

    payroll_df = prepare_payroll_data(loader)
    if payroll_df.empty:
        summary["payroll"] = "aucune série paie"
    else:
        _, mae = train_payroll_model(payroll_df, model_path=models_dir / "payroll_forecast.pkl")
        summary["payroll"] = {"mae": mae, "rows": int(len(payroll_df))}

    return jsonify(summary)


@ml_bp.route("/staffing_forecast", methods=["GET"])
def get_staffing_forecast():
    """
    Retourne les prévisions d'effectif par service/semaine pour l'année donnée.
    Paramètres query :
        year (int, optionnel)
    """
    year = request.args.get("year", type=int)
    loader = DataLoader()
    df = prepare_staffing_data(loader, start_year=year)
    if df.empty:
        return jsonify([])
    forecast_df = df.drop(columns=["employes_uniques"])
    predictions = forecast_staffing(forecast_df)
    return jsonify(predictions.to_dict(orient="records"))


@ml_bp.route("/compensation_benchmark", methods=["GET"])
def get_compensation_benchmark():
    """
    Retourne les écarts salaire réel vs attendu (top 200 par défaut).
    Paramètres query :
        start_year (int, optionnel)
        limit (int, optionnel)
    """
    start_year = request.args.get("start_year", type=int)
    limit = request.args.get("limit", default=200, type=int)
    loader = DataLoader()
    df = prepare_compensation_data(loader, start_year=start_year)
    if df.empty:
        return jsonify([])
    results = benchmark_services(df)
    return jsonify(results.head(limit).to_dict(orient="records"))


@ml_bp.route("/payroll_forecast", methods=["GET"])
def get_payroll_forecast():
    """
    Prévision masse salariale par service/mois pour l'année cible.
    Paramètres :
        year (int) -> obligatoire pour générer la grille future.
    """
    target_year = request.args.get("year", type=int)
    if not target_year:
        return jsonify({"error": "Paramètre 'year' requis."}), 400

    loader = DataLoader()
    historical = prepare_payroll_data(loader)
    if historical.empty:
        return jsonify([])

    services = historical["service_libelle"].unique()
    future_rows = []
    for service in services:
        for month in range(1, 13):
            row = historical.iloc[-1:].copy()
            row["annee"] = target_year
            row["mois"] = month
            row["service_libelle"] = service
            future_rows.append(row.drop(columns=["salaire_total"]))
    future_df = pd.concat(future_rows, ignore_index=True)
    predictions = forecast_payroll(future_df)
    return jsonify(predictions.to_dict(orient="records"))


@ml_bp.route("/scenarios/payroll", methods=["POST"])
def simulate_payroll():
    """
    Simulation d'impact paie : payload JSON avec
    {
        "service": "RESTAURANT",
        "variation_headcount": 5,
        "variation_average_salary": 120.0,
        "variation_overtime_hours": 80
    }
    """
    payload = request.get_json(force=True)
    required = {"service"}
    if not required.issubset(payload):
        return jsonify({"error": "Champ 'service' requis."}), 400

    loader = DataLoader()
    historical = prepare_payroll_data(loader)
    if historical.empty:
        return jsonify({"error": "Aucune donnée paie disponible"}), 400

    scenario = ScenarioInput(
        service=payload["service"],
        variation_headcount=payload.get("variation_headcount", 0),
        variation_average_salary=payload.get("variation_average_salary", 0.0),
        variation_overtime_hours=payload.get("variation_overtime_hours", 0.0),
        months=payload.get("months", 12),
    )

    # Build current snapshot aggregated per service
    snapshot = (
        historical.groupby("service_libelle")
        .agg(
            salaire_total=("salaire_total", "mean"),
            effectif=("service_libelle", "count"),
        )
        .reset_index()
    )

    result = simulate_payroll_impact(snapshot, scenario)
    return jsonify(result)
