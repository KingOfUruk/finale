"""Shared helpers to train ML models end-to-end."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ml import MODELS_DIR
from ml.compensation_benchmark import (
    benchmark_services,
    prepare_dataset as prepare_compensation_data,
    train_model as train_compensation_model,
)
from ml.data_loader import DataLoader
from ml.payroll_forecast import (
    forecast as forecast_payroll,
    prepare_dataset as prepare_payroll_data,
    train_model as train_payroll_model,
)
from ml.staffing_forecast import (
    forecast_staffing,
    get_dataset as prepare_staffing_data,
    train_model as train_staffing_model,
)


def _get_models_dir() -> Path:
    path = Path(MODELS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_training_pipeline(start_year: Optional[int] = None) -> Dict[str, object]:
    """Train staffing, compensation, and payroll models and return metrics."""
    loader = DataLoader()
    models_dir = _get_models_dir()

    summary: Dict[str, object] = {}

    staffing_df = prepare_staffing_data(loader, start_year=start_year)
    if staffing_df.empty:
        summary["staffing"] = "aucune donnée"
    else:
        _, mae = train_staffing_model(
            staffing_df, model_path=models_dir / "staffing_forecast.pkl"
        )
        summary["staffing"] = {"mae": mae, "rows": int(len(staffing_df))}

    compensation_df = prepare_compensation_data(loader, start_year=start_year)
    if compensation_df.empty:
        summary["compensation"] = "aucune donnée"
    else:
        _, metrics = train_compensation_model(
            compensation_df, model_path=models_dir / "compensation_benchmark.pkl"
        )
        summary["compensation"] = {
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "rows": int(len(compensation_df)),
        }

    payroll_df = prepare_payroll_data(loader)
    if payroll_df.empty:
        summary["payroll"] = "aucune série paie"
    else:
        _, mae = train_payroll_model(
            payroll_df, model_path=models_dir / "payroll_forecast.pkl"
        )
        summary["payroll"] = {"mae": mae, "rows": int(len(payroll_df))}

    return summary
