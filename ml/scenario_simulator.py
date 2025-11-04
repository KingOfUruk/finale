"""
Simple HR scenario simulator leveraging trained models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .payroll_forecast import load_model as load_payroll_model
from .compensation_benchmark import load_model as load_compensation_model


@dataclass
class ScenarioInput:
    service: str
    variation_headcount: int = 0
    variation_average_salary: float = 0.0
    variation_overtime_hours: float = 0.0
    months: int = 12


def simulate_payroll_impact(
    current_snapshot: pd.DataFrame,
    scenario: ScenarioInput,
    payroll_model=None,
) -> Dict[str, float]:
    """
    Estimate the payroll impact of a scenario by adjusting headcount/salary
    and querying the trained payroll forecast model.
    """
    payroll_model = payroll_model or load_payroll_model()

    service_data = current_snapshot[current_snapshot["service_libelle"] == scenario.service].copy()
    if service_data.empty:
        return {"service": scenario.service, "impact_tnd": 0.0}

    avg_salary = service_data["salaire_total"].mean()
    base_headcount = service_data["effectif"].mean()

    new_headcount = base_headcount + scenario.variation_headcount
    new_avg_salary = avg_salary + scenario.variation_average_salary

    incremental_cost = (new_headcount * new_avg_salary - base_headcount * avg_salary) * scenario.months

    overtime_cost = scenario.variation_overtime_hours * service_data["cout_horaire_moyen"].mean() if "cout_horaire_moyen" in service_data else 0.0

    return {
        "service": scenario.service,
        "impact_tnd": float(incremental_cost + overtime_cost),
        "new_headcount": float(new_headcount),
        "new_avg_salary": float(new_avg_salary),
    }


def simulate_compensation_adjustment(
    benchmark_dataset: pd.DataFrame,
    delta_salary: float,
    compensation_model=None,
) -> Dict[str, float]:
    """
    Evaluate how increasing/decreasing salaries by delta affects alignment with model.
    """
    compensation_model = compensation_model or load_compensation_model()
    dataset = benchmark_dataset.copy()
    dataset["salaire_annuel_adj"] = dataset["salaire_annuel"] + delta_salary
    dataset["delta"] = dataset["salaire_annuel_adj"] - dataset["salaire_annuel"]

    features = dataset.drop(columns=["salaire_annuel_adj", "delta", "mat_pers", "salaire_annuel"])
    predicted = compensation_model.predict(features)
    ecart_initial = dataset["salaire_annuel"] - predicted
    ecart_apres = dataset["salaire_annuel_adj"] - predicted

    return {
        "delta_median": float(np.median(dataset["delta"])),
        "ecart_initial_moyen": float(np.mean(ecart_initial)),
        "ecart_post_moyen": float(np.mean(ecart_apres)),
    }

