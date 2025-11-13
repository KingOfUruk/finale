from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from celery import states
from celery.result import AsyncResult
from flask import Blueprint, jsonify, request

from celery_app import celery
from ml.compensation_benchmark import benchmark_services, prepare_dataset as prepare_compensation_data
from ml.payroll_forecast import forecast as forecast_payroll, prepare_dataset as prepare_payroll_data
from ml.scenario_simulator import ScenarioInput, simulate_compensation_adjustment, simulate_payroll_impact
from ml.staffing_forecast import forecast_staffing, get_dataset as prepare_staffing_data
from ml.data_loader import DataLoader
from ml.tasks import train_all_models

ml_bp = Blueprint("ml", __name__, url_prefix="/api/ml")


@ml_bp.route("/train", methods=["POST"])
def trigger_training():
    """
    Lance l'entraînement de l'ensemble des modèles en tâche de fond.
    Optionnellement, accepter un champ JSON {"start_year": 2020}.
    """
    payload = request.get_json(silent=True) or {}
    start_year = payload.get("start_year")
    async_result = train_all_models.delay(start_year=start_year)
    logging.info("Training job %s scheduled (start_year=%s)", async_result.id, start_year)
    return jsonify({"job_id": async_result.id}), 202


@ml_bp.route("/train/<job_id>", methods=["GET"])
def get_training_status(job_id: str):
    """Retourne l'état du job d'entraînement."""
    result = AsyncResult(job_id, app=celery)
    if result.state == states.PENDING:
        payload = {"job_id": job_id, "status": "queued"}
    elif result.state == states.STARTED:
        payload = {"job_id": job_id, "status": "running"}
    elif result.state == states.SUCCESS:
        payload = {
            "job_id": job_id,
            "status": "finished",
            "result": result.result,
        }
    elif result.state == states.FAILURE:
        payload = {
            "job_id": job_id,
            "status": "failed",
            "error": str(result.info),
        }
    else:
        payload = {"job_id": job_id, "status": result.state.lower()}
    return jsonify(payload)


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
