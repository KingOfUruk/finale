"""
CLI helper to train all ML models.

Usage:
    python -m ml.train_all --start-year 2020
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .compensation_benchmark import prepare_dataset as prepare_compensation, train_model as train_compensation
from .data_loader import DataLoader
from .payroll_forecast import prepare_dataset as prepare_payroll, train_model as train_payroll
from .staffing_forecast import get_dataset as prepare_staffing, train_model as train_staffing


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML models for HR analytics.")
    parser.add_argument("--start-year", type=int, default=None, help="Filter data from this year forward.")
    parser.add_argument("--models-dir", type=str, default=None, help="Custom directory for saving models.")
    args = parser.parse_args()

    loader = DataLoader()
    models_dir = Path(args.models_dir) if args.models_dir else None

    # Staffing / attendance forecast
    staffing_df = prepare_staffing(loader, start_year=args.start_year)
    if staffing_df.empty:
        print("Aucune donnée de pointage trouvée pour l'entraînement staffing.")
    else:
        _, mae = train_staffing(staffing_df, model_path=models_dir / "staffing_forecast.pkl" if models_dir else None)
        print(f"Modèle staffing entraîné (MAE={mae:.2f}).")

    # Compensation benchmarking
    compensation_df = prepare_compensation(loader, start_year=args.start_year)
    if compensation_df.empty:
        print("Aucune donnée de paie disponible pour le benchmarking.")
    else:
        _, metrics = train_compensation(compensation_df, model_path=models_dir / "compensation_benchmark.pkl" if models_dir else None)
        print(f"Modèle compensation entraîné (MAE={metrics['mae']:.2f}, R2={metrics['r2']:.3f}).")

    # Payroll forecast
    payroll_df = prepare_payroll(loader)
    if payroll_df.empty:
        print("Aucune série paie disponible pour la prévision.")
    else:
        _, mae = train_payroll(payroll_df, model_path=models_dir / "payroll_forecast.pkl" if models_dir else None)
        print(f"Modèle de prévision paie entraîné (MAE={mae:.2f}).")


if __name__ == "__main__":
    main()
