from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sqlalchemy
from flask import Blueprint, jsonify, render_template, request, send_file
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

from app.config.database import get_oracle_credentials, build_sqlalchemy_url

# Blueprint configuration
payroll_bp = Blueprint("payroll", __name__, template_folder="templates")
CORS(payroll_bp)

logger = logging.getLogger(__name__)

NUMERIC_COLUMNS = [
    "salaire_brut",
    "salaire_net",
    "montant_primes",
    "montant_cnss",
    "montant_impots",
    "montant_avances",
    "montant_rappels",
    "montant_conges",
    "montant_assurance",
    "jours_absence",
    "nombre_enfants",
]

MIN_YEAR = 2018
FORECAST_HORIZON_YEARS = 5


def _safe_float(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    try:
        number = float(value)
        if np.isnan(number):
            return 0.0
        return number
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Optional[float]) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _format_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _month_label(year: int, month: int, name: Optional[str]) -> str:
    if name:
        return f"{name.title()} {year}"
    return f"{year}-{str(month).zfill(2)}"


def _percent_change(current: float, previous: float) -> float:
    if previous == 0:
        if current == 0:
            return 0.0
        return 100.0 if current > 0 else -100.0
    return round(((current - previous) / abs(previous)) * 100, 2)


def _format_currency(value: Optional[float]) -> str:
    return f"{_safe_float(value):,.2f}".replace(",", " ")


def _format_percent(value: Optional[float]) -> str:
    return f"{_safe_float(value) * 100:.2f}%"


@dataclass
class PayrollDataService:
    config: Optional[Dict[str, str]] = None
    initialization_error: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        creds = self.config or get_oracle_credentials()
        self.connection_string = build_sqlalchemy_url(creds)
        self.engine: Optional[sqlalchemy.Engine] = None
        self.connect()

    def connect(self) -> bool:
        if self.initialization_error:
            logger.error("Connexion Oracle impossible : %s", self.initialization_error)
            return False
        try:
            self.engine = sqlalchemy.create_engine(self.connection_string)
            with self.engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1 FROM dual"))
            logger.info("Connexion Oracle établie")
            return True
        except Exception as exc:  # pragma: no cover - logging only
            logger.error("Connexion Oracle échouée: %s", exc)
            self.engine = None
            return False

    def ensure_connection(self) -> bool:
        if self.engine is not None:
            return True
        return self.connect()

    def extract_payroll_data(self, year: Optional[int] = None) -> pd.DataFrame:
        if not self.ensure_connection():
            logger.error("Impossible d'extraire les données: connexion absente")
            return pd.DataFrame()

        filter_clause = ""
        params: Dict[str, int] = {}
        if year:
            filter_clause = "AND dd.annee = :annee"
            params["annee"] = int(year)
        else:
            filter_clause = "AND dd.annee >= :min_year"
            params["min_year"] = MIN_YEAR

        query_template = """
            SELECT
                fp.id_paie AS id_paie,
                fp.mat_pers AS mat_pers,
                fp.id_temps AS id_temps,
                fp.cod_serv AS code_service,
                fp.cod_cat AS code_categorie,
                fp.salaire_brut AS salaire_brut,
                fp.salaire_net AS salaire_net,
                fp.montant_primes AS montant_primes,
                fp.montant_cnss AS montant_cnss,
                fp.montant_impots AS montant_impots,
                fp.montant_avances AS montant_avances,
                fp.montant_rappels AS montant_rappels,
                fp.jours_absence AS jours_absence,
                fp.montant_conges AS montant_conges,
                fp.montant_assurance AS montant_assurance,
                ds.libelle AS service_libelle,
                ds.type AS service_type,
                dc.lib_cat AS categorie_libelle,
                dp.sexe AS sexe,
                dp.dat_nais AS date_naissance,
                dp.dat_emb AS date_embauche,
                dp.etat_civil AS etat_civil,
                dp.nbre_enf AS nombre_enfants,
                dp.age AS age,
                dp.anciennete AS anciennete,
                dp.band_age AS tranche_age,
                dd.annee AS annee,
                dd.mois AS mois,
                dd.nom_mois AS nom_mois,
                dd.trimestre AS trimestre
            FROM fait_paie fp
            LEFT JOIN dim_service ds ON fp.cod_serv = ds.code_serv
            LEFT JOIN dim_categorie dc ON fp.cod_cat = dc.{categorie_key}
            LEFT JOIN dim_personnel dp ON fp.mat_pers = dp.mat_pers
            LEFT JOIN dim_date dd ON fp.id_temps = dd.id_temps
            WHERE dd.annee IS NOT NULL {filter_clause}
        """

        last_error: Optional[Exception] = None
        for categorie_key in ("cod_cat", "cod_c"):
            try:
                query = sqlalchemy.text(query_template.format(categorie_key=categorie_key, filter_clause=filter_clause))
                df = pd.read_sql(query, self.engine, params=params)
                break
            except Exception as exc:  # pragma: no cover - logging only
                logger.warning("Échec extraction avec jointure sur dim_categorie.%s: %s", categorie_key, exc)
                last_error = exc
                df = pd.DataFrame()
        else:
            logger.error("Erreur lors de la lecture SQL: %s", last_error)
            return pd.DataFrame()

        if df.empty:
            return df

        for column in NUMERIC_COLUMNS:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

        if "annee" in df.columns:
            df["annee"] = pd.to_numeric(df["annee"], errors="coerce").fillna(0).astype(int)
        if "mois" in df.columns:
            df["mois"] = pd.to_numeric(df["mois"], errors="coerce").fillna(0).astype(int)
        if "date_comptable" in df.columns:
            df["date_comptable"] = pd.to_datetime(df["date_comptable"], errors="coerce")
        if "anciennete" in df.columns:
            df["anciennete"] = pd.to_numeric(df["anciennete"], errors="coerce")
        if "age" in df.columns:
            df["age"] = pd.to_numeric(df["age"], errors="coerce")

        return df

    def get_available_years(self) -> List[int]:
        if not self.ensure_connection():
            return []
        query = sqlalchemy.text(
            """
            SELECT DISTINCT dd.annee
            FROM fait_paie fp
            JOIN dim_date dd ON fp.id_temps = dd.id_temps
            WHERE dd.annee IS NOT NULL
            ORDER BY dd.annee DESC
            """
        )
        try:
            df = pd.read_sql(query, self.engine)
            return [int(year) for year in df["ANNEE" if "ANNEE" in df.columns else "annee"].tolist()]
        except Exception as exc:  # pragma: no cover - logging only
            logger.error("Erreur récupération années: %s", exc)
            return []


def _monthly_summary(df: pd.DataFrame) -> List[Dict[str, float]]:
    if df.empty:
        return []
    monthly = (
        df.groupby(["annee", "mois", "nom_mois"], dropna=False)
        .agg(
            salaire_net_sum=("salaire_net", "sum"),
            salaire_brut_sum=("salaire_brut", "sum"),
            primes_sum=("montant_primes", "sum"),
            charges_sum=("montant_cnss", "sum"),
            impots_sum=("montant_impots", "sum"),
            assurance_sum=("montant_assurance", "sum"),
            avances_sum=("montant_avances", "sum"),
            employee_count=("mat_pers", pd.Series.nunique),
        )
        .reset_index()
    )

    monthly = monthly.sort_values(["annee", "mois"])
    payload: List[Dict[str, float]] = []
    for _, row in monthly.iterrows():
        label = _month_label(_safe_int(row["annee"]), _safe_int(row["mois"]), row.get("nom_mois"))
        employee_count = int(row.get("employee_count", 0) or 0)
        avg_net = _safe_float(row["salaire_net_sum"]) / employee_count if employee_count else 0.0
        charges = _safe_float(row["charges_sum"]) + _safe_float(row["impots_sum"]) + _safe_float(row["assurance_sum"])
        payload.append(
            {
                "label": label,
                "salaire_net": round(_safe_float(row["salaire_net_sum"]), 2),
                "salaire_brut": round(_safe_float(row["salaire_brut_sum"]), 2),
                "primes": round(_safe_float(row["primes_sum"]), 2),
                "charges": round(charges, 2),
                "avances": round(_safe_float(row["avances_sum"]), 2),
                "employee_count": employee_count,
                "salaire_net_moyen": round(avg_net, 2),
            }
        )
    return payload


def _seasonal_summary(df: pd.DataFrame) -> List[Dict[str, float]]:
    if df.empty or not {"annee", "trimestre"}.issubset(df.columns):
        return []

    seasonal = (
        df.groupby(["annee", "trimestre"], dropna=False)
        .agg(
            salaire_net_sum=("salaire_net", "sum"),
            employee_count=("mat_pers", pd.Series.nunique),
        )
        .reset_index()
    )

    seasonal = seasonal.sort_values(["annee", "trimestre"])
    payload: List[Dict[str, float]] = []
    for _, row in seasonal.iterrows():
        label = f"T{_safe_int(row['trimestre'])} {_safe_int(row['annee'])}"
        employee_count = int(row.get("employee_count", 0) or 0)
        avg_net = _safe_float(row["salaire_net_sum"]) / employee_count if employee_count else 0.0
        payload.append(
            {
                "label": label,
                "salaire_net": round(_safe_float(row["salaire_net_sum"]), 2),
                "employee_count": employee_count,
                "salaire_net_moyen": round(avg_net, 2),
            }
        )
    return payload


def _employee_monthly_breakdown(df: pd.DataFrame, top_n: int = 10) -> Dict[str, List[Dict[str, object]]]:
    required_cols = {"mat_pers", "annee", "mois", "salaire_net"}
    if df.empty or not required_cols.issubset(df.columns):
        return {"months": [], "per_service": [], "per_categorie": []}

    subset_cols = [
        "mat_pers",
        "annee",
        "mois",
        "salaire_net",
        "service_libelle",
        "categorie_libelle",
    ]
    if "nom_mois" in df.columns:
        subset_cols.append("nom_mois")
    base = df[subset_cols].dropna(subset=["mat_pers", "annee", "mois"]).copy()
    if base.empty:
        return {"months": [], "per_service": [], "per_categorie": []}

    base["annee"] = base["annee"].astype(int)
    base["mois"] = base["mois"].astype(int)
    base["month_key"] = base["annee"].astype(str) + "-" + base["mois"].astype(str).str.zfill(2)
    base["service_libelle"] = base["service_libelle"].fillna("Service inconnu")
    base["categorie_libelle"] = base["categorie_libelle"].fillna("Catégorie inconnue")

    month_meta = (
        base[["annee", "mois", "month_key"] + (["nom_mois"] if "nom_mois" in base.columns else [])]
        .drop_duplicates(subset=["month_key"])
        .sort_values(["annee", "mois"])
    )
    month_order: List[str] = []
    month_labels: List[Dict[str, str]] = []
    for _, row in month_meta.iterrows():
        key = str(row["month_key"])
        name = None
        if "nom_mois" in row.index:
            raw_name = row["nom_mois"]
            if pd.notna(raw_name):
                name = str(raw_name)
        label = _month_label(_safe_int(row["annee"]), _safe_int(row["mois"]), name)
        month_order.append(key)
        month_labels.append({"key": key, "label": label})

    aggregated = (
        base.groupby([
            "service_libelle",
            "categorie_libelle",
            "mat_pers",
            "month_key",
        ])
        ["salaire_net"]
        .sum()
        .reset_index()
    )

    def _build_group(group_level: str, other_label: str) -> List[Dict[str, object]]:
        results: List[Dict[str, object]] = []
        for value, group in aggregated.groupby(group_level):
            employees: List[Dict[str, object]] = []
            for mat, emp_df in group.groupby("mat_pers"):
                monthly_map = {
                    row["month_key"]: round(_safe_float(row["salaire_net"]), 2)
                    for _, row in emp_df.iterrows()
                }
                total = round(sum(monthly_map.values()), 2)
                other_value = str(emp_df[other_label].iloc[0])
                employees.append(
                    {
                        "mat_pers": str(mat),
                        "total": total,
                        "monthly": monthly_map,
                        "service": str(emp_df["service_libelle"].iloc[0]),
                        "categorie": str(emp_df["categorie_libelle"].iloc[0]),
                        "label": other_value,
                    }
                )
            employees.sort(key=lambda item: item["total"], reverse=True)
            limited = employees[:top_n]
            results.append(
                {
                    "label": str(value),
                    "employees": limited,
                }
            )
        results.sort(key=lambda item: item["label"])
        return results

    per_service = _build_group("service_libelle", "categorie_libelle")
    per_categorie = _build_group("categorie_libelle", "service_libelle")

    return {
        "months": month_labels,
        "per_service": per_service,
        "per_categorie": per_categorie,
    }


def _prepare_monthly_series(df: pd.DataFrame, service: Optional[str] = None, metric: str = "salaire_net") -> pd.Series:
    """
    Aggregate monthly net salaries for the whole scope or a specific service.
    """
    if df.empty:
        return pd.Series(dtype=float)

    if metric not in df.columns:
        return pd.Series(dtype=float)

    scoped = df if service is None else df[df["service_libelle"] == service]
    if scoped.empty:
        return pd.Series(dtype=float)

    scoped = scoped.copy()
    scoped[metric] = pd.to_numeric(scoped[metric], errors="coerce").fillna(0.0)

    monthly = (
        scoped.groupby(["annee", "mois"])
        .agg(metric_sum=(metric, "sum"))
        .reset_index()
    )
    monthly["date"] = pd.to_datetime(
        monthly["annee"].astype(int).astype(str) + "-" + monthly["mois"].astype(int).astype(str).str.zfill(2) + "-01",
        errors="coerce",
    )
    monthly = monthly.dropna(subset=["date"]).sort_values("date")
    if monthly.empty:
        return pd.Series(dtype=float)

    series = monthly.set_index("date")["metric_sum"].astype(float)
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    return series.reindex(full_index, fill_value=0.0)


def _build_time_features(index: pd.DatetimeIndex, start_offset: int = 0) -> pd.DataFrame:
    features = pd.DataFrame({"date": index})
    features["t"] = np.arange(start_offset, start_offset + len(index))
    features["t2"] = features["t"] ** 2
    month = features["date"].dt.month
    angle = 2 * np.pi * month / 12.0
    features["sin12"] = np.sin(angle)
    features["cos12"] = np.cos(angle)
    features["sin24"] = np.sin(2 * angle)
    features["cos24"] = np.cos(2 * angle)
    return features


def _period_highlights(df: pd.DataFrame, column: str) -> Dict[str, Dict[str, object]]:
    if df.empty or column not in df.columns:
        return {}

    subset_cols = ["annee", "mois", "nom_mois", "trimestre", column]
    available = [col for col in subset_cols if col in df.columns]
    if column not in available:
        return {}

    data = df[available].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0.0)
    if data.empty:
        return {}

    highlights: Dict[str, Dict[str, object]] = {}

    if {"annee", "mois"}.issubset(data.columns):
        monthly = (
            data.groupby(["annee", "mois", "nom_mois"], dropna=False)[column]
            .sum()
            .reset_index()
            .sort_values(column, ascending=False)
        )
        if not monthly.empty:
            top = monthly.iloc[0]
            label = _month_label(_safe_int(top["annee"]), _safe_int(top["mois"]), top.get("nom_mois"))
            highlights["top_month"] = {
                "label": label,
                "annee": _safe_int(top["annee"]),
                "mois": _safe_int(top["mois"]),
                "value": round(_safe_float(top[column]), 2),
            }

    if {"annee", "trimestre"}.issubset(data.columns):
        quarterly = (
            data.groupby(["annee", "trimestre"], dropna=False)[column]
            .sum()
            .reset_index()
            .dropna(subset=["trimestre"])
            .sort_values(column, ascending=False)
        )
        if not quarterly.empty:
            top_q = quarterly.iloc[0]
            quarter_label = f"T{_safe_int(top_q['trimestre'])} {_safe_int(top_q['annee'])}"
            highlights["top_quarter"] = {
                "label": quarter_label,
                "annee": _safe_int(top_q["annee"]),
                "trimestre": _safe_int(top_q["trimestre"]),
                "value": round(_safe_float(top_q[column]), 2),
            }

    if "annee" in data.columns:
        yearly = (
            data.groupby("annee", dropna=False)[column]
            .sum()
            .reset_index()
            .sort_values(column, ascending=False)
        )
        if not yearly.empty:
            top_year = yearly.iloc[0]
            highlights["top_year"] = {
                "label": str(_safe_int(top_year["annee"])),
                "annee": _safe_int(top_year["annee"]),
                "value": round(_safe_float(top_year[column]), 2),
            }

    return highlights


def _fit_seasonal_regression(series: pd.Series, horizon_months: int) -> Optional[Dict[str, object]]:
    """
    Fit a simple linear regression enriched with seasonal harmonics and project over the requested horizon.
    Returns None if there is not enough history to build a robust projection.
    """
    if series.empty:
        return None

    valid = series.dropna()
    if len(valid) < 12:
        return None

    history_index = valid.index
    history_values = valid.values.astype(float)

    train_features = _build_time_features(history_index, start_offset=0)
    X_train = train_features[["t", "t2", "sin12", "cos12", "sin24", "cos24"]]
    y_train = history_values

    model = LinearRegression()
    model.fit(X_train, y_train)
    fitted = model.predict(X_train)
    residuals = y_train - fitted

    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.abs(residuals).mean() if len(residuals) else 0.0)
    mae = float(np.mean(np.abs(residuals))) if len(residuals) else 0.0

    future_index = pd.date_range(history_index[-1] + pd.offsets.MonthBegin(1), periods=horizon_months, freq="MS")
    future_features = _build_time_features(future_index, start_offset=len(history_index))
    X_future = future_features[["t", "t2", "sin12", "cos12", "sin24", "cos24"]]
    preds = model.predict(X_future)
    preds = np.clip(preds, a_min=0.0, a_max=None)

    ci = 1.96 * residual_std if residual_std else 0.0
    lower_preds = np.clip(preds - ci, a_min=0.0, a_max=None)
    upper_preds = np.clip(preds + ci, a_min=0.0, a_max=None)

    history_points = [
        {"date": idx.strftime("%Y-%m"), "value": round(float(val), 2)}
        for idx, val in zip(history_index[-60:], history_values[-60:])
    ]
    forecast_points = [
        {
            "date": idx.strftime("%Y-%m"),
            "value": round(float(pred), 2),
            "lower": round(float(max(pred - ci, 0.0)), 2),
            "upper": round(float(pred + ci), 2),
        }
        for idx, pred in zip(future_index, preds)
    ]

    next_year_sum = float(np.sum(preds[:12])) if len(preds) >= 12 else float(np.sum(preds))
    next_year_sum_lower = float(np.sum(lower_preds[:12])) if len(lower_preds) >= 12 else float(np.sum(lower_preds))
    next_year_sum_upper = float(np.sum(upper_preds[:12])) if len(upper_preds) >= 12 else float(np.sum(upper_preds))
    five_year_sum = float(np.sum(preds))
    five_year_sum_lower = float(np.sum(lower_preds))
    five_year_sum_upper = float(np.sum(upper_preds))
    start_value = float(history_values[-1])
    end_value = float(preds[-1]) if len(preds) else start_value
    years = horizon_months / 12 if horizon_months else 0
    cagr = ((end_value / start_value) ** (1 / years) - 1) if years and start_value > 0 and end_value > 0 else 0.0

    return {
        "history": history_points,
        "history_months": len(history_index),
        "full_history_sum": float(np.sum(history_values[-12:])) if len(history_values) >= 12 else float(np.sum(history_values)),
        "forecast": forecast_points,
        "mae": round(mae, 2),
        "residual_std": round(residual_std, 2),
        "next_year_sum": round(next_year_sum, 2),
        "five_year_sum": round(five_year_sum, 2),
        "next_year_sum_lower": round(next_year_sum_lower, 2),
        "next_year_sum_upper": round(next_year_sum_upper, 2),
        "five_year_sum_lower": round(five_year_sum_lower, 2),
        "five_year_sum_upper": round(five_year_sum_upper, 2),
        "start_value": round(start_value, 2),
        "end_value": round(end_value, 2),
        "cagr": round(float(cagr), 4) if not np.isnan(cagr) else 0.0,
        "last_actual": {
            "date": history_index[-1].strftime("%Y-%m"),
            "value": round(float(history_values[-1]), 2),
        },
    }


def _compute_forecast_projection(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {"error": "Aucune donnée disponible pour construire la projection."}

    horizon_months = FORECAST_HORIZON_YEARS * 12
    df = df.copy()
    if "service_libelle" in df.columns:
        df["service_libelle"] = df["service_libelle"].fillna("Service inconnu")
    global_series = _prepare_monthly_series(df, metric="salaire_net")
    global_result = _fit_seasonal_regression(global_series, horizon_months)
    if not global_result:
        return {"error": "Historique insuffisant pour calculer une projection fiable."}

    actual_last_year_sum = float(global_series.tail(12).sum()) if not global_series.empty else 0.0

    latest_year = int(df["annee"].max())
    recent_mask = df["annee"] == latest_year
    recent_totals = (
        df.loc[recent_mask]
        .groupby("service_libelle")["salaire_net"]
        .sum()
        .sort_values(ascending=False)
    )
    top_services = recent_totals.index.tolist()

    global_recent_sum = global_result.get("full_history_sum", 0.0) or 0.0
    services_payload: List[Dict[str, object]] = []
    for service in top_services:
        service_name = str(service or "Service inconnu")
        service_series = _prepare_monthly_series(df, service_name, metric="salaire_net")
        service_result = _fit_seasonal_regression(service_series, horizon_months)
        if not service_result:
            continue
        share = 0.0
        if global_recent_sum > 0:
            share = min(1.0, service_result.get("full_history_sum", 0.0) / global_recent_sum)
        last_year_service = float(service_series.tail(12).sum()) if not service_series.empty else 0.0
        delta_next_year = service_result["next_year_sum"] - last_year_service
        delta_pct = (delta_next_year / last_year_service) if last_year_service else None
        services_payload.append(
            {
                "service": service_name,
                "share": round(float(share), 4),
                "history": service_result["history"],
                "forecast": service_result["forecast"],
                "next_year_sum": service_result["next_year_sum"],
                "five_year_sum": service_result["five_year_sum"],
                "next_year_sum_lower": service_result["next_year_sum_lower"],
                "next_year_sum_upper": service_result["next_year_sum_upper"],
                "five_year_sum_lower": service_result["five_year_sum_lower"],
                "five_year_sum_upper": service_result["five_year_sum_upper"],
                "cagr": service_result["cagr"],
                "mae": service_result["mae"],
                "residual_std": service_result["residual_std"],
                "last_actual": service_result["last_actual"],
                "actual_last_year_sum": round(last_year_service, 2),
                "delta_next_year": round(delta_next_year, 2),
                "delta_next_year_pct": round(delta_pct, 4) if delta_pct is not None else None,
            }
        )

    global_payload = {
        "history": global_result["history"],
        "forecast": global_result["forecast"],
        "next_year_sum": global_result["next_year_sum"],
        "five_year_sum": global_result["five_year_sum"],
        "next_year_sum_lower": global_result["next_year_sum_lower"],
        "next_year_sum_upper": global_result["next_year_sum_upper"],
        "five_year_sum_lower": global_result["five_year_sum_lower"],
        "five_year_sum_upper": global_result["five_year_sum_upper"],
        "cagr": global_result["cagr"],
        "mae": global_result["mae"],
        "residual_std": global_result["residual_std"],
        "last_actual": global_result["last_actual"],
        "actual_last_year_sum": round(actual_last_year_sum, 2),
        "delta_next_year": round(global_result["next_year_sum"] - actual_last_year_sum, 2),
        "delta_next_year_pct": round((global_result["next_year_sum"] - actual_last_year_sum) / actual_last_year_sum, 4)
        if actual_last_year_sum
        else None,
    }

    metrics_payload: Dict[str, Dict[str, object]] = {}
    metrics_payload["net"] = {
        "label": "Masse salariale nette",
        "forecast": global_result,
        "highlights": _period_highlights(df, "salaire_net"),
        "actual_last_year_sum": round(actual_last_year_sum, 2),
        "delta_next_year": round(global_result["next_year_sum"] - actual_last_year_sum, 2),
        "delta_next_year_pct": round((global_result["next_year_sum"] - actual_last_year_sum) / actual_last_year_sum, 4)
        if actual_last_year_sum
        else None,
    }

    for column, key, label in [
        ("montant_avances", "avances", "Avances"),
        ("montant_primes", "primes", "Primes"),
        ("montant_rappels", "rappels", "Rappels"),
        ("montant_cnss", "charges", "Charges sociales"),
    ]:
        series = _prepare_monthly_series(df, metric=column)
        result = _fit_seasonal_regression(series, horizon_months) if not series.empty else None
        actual_sum = float(series.tail(12).sum()) if not series.empty else 0.0
        delta = result["next_year_sum"] - actual_sum if result else None
        delta_pct = (delta / actual_sum) if (result and actual_sum) else None
        metrics_payload[key] = {
            "label": label,
            "forecast": result,
            "highlights": _period_highlights(df, column),
            "actual_last_year_sum": round(actual_sum, 2),
            "delta_next_year": round(delta, 2) if delta is not None else None,
            "delta_next_year_pct": round(delta_pct, 4) if delta_pct is not None else None,
        }

    global_payload["highlights"] = metrics_payload["net"]["highlights"]
    global_payload["sensitivity"] = {
        "next_year": {
            "lower": round(global_result["next_year_sum_lower"], 2),
            "baseline": round(global_result["next_year_sum"], 2),
            "upper": round(global_result["next_year_sum_upper"], 2),
        },
        "five_year": {
            "lower": round(global_result["five_year_sum_lower"], 2),
            "baseline": round(global_result["five_year_sum"], 2),
            "upper": round(global_result["five_year_sum_upper"], 2),
        },
    }

    model_details = {
        "name": "Régression linéaire saisonnière",
        "description": "Modèle interne combinant une tendance quadratique et des composantes sinusoïdales (12 mois) pour capturer la saisonnalité paie.",
        "horizon_months": horizon_months,
        "horizon_years": FORECAST_HORIZON_YEARS,
        "history_months": global_result["history_months"],
        "residual_std": global_result["residual_std"],
        "mae": global_result["mae"],
        "generated_at": datetime.utcnow().isoformat(),
        "assumptions": [
            "Projection réalisée à partir de la masse salariale nette mensuelle.",
            "Les tendances observées sur l'historique sont prolongées sans choc externe.",
            "Les composantes saisonnières restent stables sur la période projetée.",
        ],
        "features": [
            "Tendance (t, t²)",
            "Saisonnalité sin/cos 12 mois",
            "Harmoniques secondaires sin/cos 24 mois",
        ],
    }

    return {
        "model": model_details,
        "global": global_payload,
        "services": services_payload,
        "metrics": metrics_payload,
    }


def _build_salary_distribution(series: pd.Series) -> Dict[str, int]:
    if series.empty:
        return {}
    bins = [0, 500, 1000, 1500, 2000, 2500, 4000, 6000, np.inf]
    labels = [
        "0-500",
        "500-1 000",
        "1 000-1 500",
        "1 500-2 000",
        "2 000-2 500",
        "2 500-4 000",
        "4 000-6 000",
        "6 000+",
    ]
    counts = pd.cut(series, bins=bins, labels=labels, include_lowest=True, right=False)
    distribution = counts.value_counts().sort_index()
    return {str(idx): int(value) for idx, value in distribution.items() if value > 0}


def _bucket_age(series: pd.Series) -> Dict[str, int]:
    if series.empty:
        return {}
    bins = [0, 25, 35, 45, 55, 65, np.inf]
    labels = ["<25", "25-34", "35-44", "45-54", "55-64", "65+"]
    categorized = pd.cut(series.fillna(0), bins=bins, labels=labels, right=False, include_lowest=True)
    counts = categorized.value_counts().sort_index()
    return {label: int(count) for label, count in counts.items() if count > 0}


def _bucket_anciennete(values: pd.Series) -> Dict[str, int]:
    if values.empty:
        return {}
    bins = [-np.inf, 1, 3, 5, 10, 15, np.inf]
    labels = ["< 1 an", "1-3 ans", "3-5 ans", "5-10 ans", "10-15 ans", "15+ ans"]
    categorized = pd.cut(values.fillna(0), bins=bins, labels=labels)
    counts = categorized.value_counts().sort_index()
    return {label: int(count) for label, count in counts.items() if count > 0}


def _compute_overview(df: pd.DataFrame, year: Optional[int]) -> Dict[str, float]:
    if df.empty:
        return {}

    total_employees = df["mat_pers"].nunique()
    masse_brut = _safe_float(df["salaire_brut"].sum())
    masse_net = _safe_float(df["salaire_net"].sum())
    primes = _safe_float(df["montant_primes"].sum())
    total_cnss = _safe_float(df["montant_cnss"].sum())
    total_impots = _safe_float(df["montant_impots"].sum())
    total_assurance = _safe_float(df["montant_assurance"].sum())
    charges = total_cnss + total_impots + total_assurance
    avances = _safe_float(df["montant_avances"].sum())
    rappels = _safe_float(df["montant_rappels"].sum())
    conges = _safe_float(df["montant_conges"].sum())
    absences = _safe_float(df["jours_absence"].sum())
    deduction_total = max(masse_brut - masse_net, 0.0)

    moyenne_nette = _safe_float(masse_net / total_employees) if total_employees else 0.0
    ratio_charges = _format_ratio(charges, masse_brut)
    ratio_primes = _format_ratio(primes, masse_brut)
    deduction_rate = _format_ratio(deduction_total, masse_brut)
    cnss_rate = _format_ratio(total_cnss, masse_brut)
    tax_rate = _format_ratio(total_impots, masse_brut)
    average_absence = _safe_float(absences / total_employees) if total_employees else 0.0
    months_covered = 1
    if {"annee", "mois"}.issubset(df.columns):
        months_covered = max(1, df[["annee", "mois"]].dropna().drop_duplicates().shape[0])

    employee_months_total = 0
    adjusted_net_mean = moyenne_nette
    if {"mat_pers", "annee", "mois", "salaire_net"}.issubset(df.columns):
        month_df = df[["mat_pers", "annee", "mois"]].dropna().copy()
        if not month_df.empty:
            month_df["annee"] = month_df["annee"].astype(int)
            month_df["mois"] = month_df["mois"].astype(int)
            month_df["mois_key"] = month_df["annee"].astype(str) + '-' + month_df["mois"].astype(str).str.zfill(2)
            months_per_employee = month_df.groupby("mat_pers")["mois_key"].nunique()
            employee_months_total = int(months_per_employee.sum())
            net_per_employee = df.groupby("mat_pers")["salaire_net"].sum().reindex(months_per_employee.index, fill_value=0.0)
            valid_months = months_per_employee.replace(0, np.nan)
            per_employee_monthly = net_per_employee / valid_months
            per_employee_monthly = per_employee_monthly.dropna()
            if not per_employee_monthly.empty:
                adjusted_net_mean = _safe_float(per_employee_monthly.mean())

    moyenne_nette = adjusted_net_mean

    daily_cost_base = 0.0
    denominator_employees = total_employees if total_employees else 1
    if denominator_employees > 0:
        assumed_working_days = months_covered * 22
        daily_cost_base = (masse_brut / denominator_employees) / max(assumed_working_days, 1)
    absence_cost_estimate = round(daily_cost_base * absences, 2)

    cost_per_employee_brut = _safe_float(masse_brut / denominator_employees)
    cost_per_employee_net = _safe_float(masse_net / denominator_employees)

    monthly_summary = _monthly_summary(df)
    seasonal_summary: List[Dict[str, float]] = []
    try:
        seasonal_summary = _seasonal_summary(df)
    except Exception as exc:  # pragma: no cover - defensive log
        logger.warning("Échec agrégat trimestriel: %s", exc)
    anomalies: List[Dict[str, float]] = []
    if monthly_summary:
        net_values = [entry.get("salaire_net", 0.0) for entry in monthly_summary]
        if net_values:
            series = pd.Series(net_values)
            mean = series.mean()
            std = series.std(ddof=0)
            threshold = mean + 2 * std
            for entry in monthly_summary:
                montant = entry.get("salaire_net", 0.0)
                if std > 0 and montant > threshold:
                    anomalies.append({
                        "label": entry.get("label"),
                        "valeur": round(montant, 2),
                        "ecart_percent": round((montant - mean) / mean * 100, 2) if mean else 0.0,
                    })

    annee_courante = int(year or df["annee"].dropna().max()) if "annee" in df.columns and not df["annee"].dropna().empty else datetime.now().year

    return {
        "annee_courante": annee_courante,
        "total_employes": int(total_employees),
        "nombre_services": int(df["service_libelle"].nunique(dropna=True)),
        "nombre_categories": int(df["categorie_libelle"].nunique(dropna=True)),
        "masse_salariale_brute": round(masse_brut, 2),
        "masse_salariale_nette": round(masse_net, 2),
        "salaire_net_moyen": round(moyenne_nette, 2),
        "total_primes": round(primes, 2),
        "total_charges": round(charges, 2),
        "total_avances": round(avances, 2),
        "total_rappels": round(rappels, 2),
        "total_conges": round(conges, 2),
        "total_absence_jours": round(absences, 2),
        "ratio_charges_brut": ratio_charges,
        "ratio_primes_brut": ratio_primes,
        "deduction_rate": deduction_rate,
        "bonus_ratio": ratio_primes,
        "total_cnss": round(total_cnss, 2),
        "cnss_rate": cnss_rate,
        "total_impots": round(total_impots, 2),
        "tax_rate": tax_rate,
        "total_assurance": round(total_assurance, 2),
        "average_absence_par_employe": round(average_absence, 2),
        "absence_cost_estime": round(absence_cost_estimate, 2),
        "cout_par_employe_brut": round(cost_per_employee_brut, 2),
        "cout_par_employe_net": round(cost_per_employee_net, 2),
        "total_mois_remuneres": employee_months_total,
        "tendance_mensuelle": monthly_summary,
        "moyennes_mensuelles": [
            {
                "label": entry.get("label"),
                "salaire_net_moyen": entry.get("salaire_net_moyen", 0.0),
                "employee_count": entry.get("employee_count", 0),
            }
            for entry in monthly_summary
        ],
        "moyennes_trimestrielles": seasonal_summary,
        "anomalies_mensuelles": anomalies,
    }


def _compute_salary_insights(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}

    payload: Dict[str, Dict[str, float]] = {}
    total_brut = _safe_float(df["salaire_brut"].sum())
    total_net = _safe_float(df["salaire_net"].sum())
    total_primes = _safe_float(df["montant_primes"].sum())
    total_rappels = _safe_float(df["montant_rappels"].sum())
    bonus_ratio = _format_ratio(total_primes, total_brut)
    deduction_rate = _format_ratio(max(total_brut - total_net, 0.0), total_brut)
    gross_to_net_ratio = _format_ratio(total_net, total_brut)

    payload["compensation_summary"] = {
        "total_brut": round(total_brut, 2),
        "total_net": round(total_net, 2),
        "total_primes": round(total_primes, 2),
        "total_rappels": round(total_rappels, 2),
        "bonus_ratio": bonus_ratio,
        "deduction_rate": deduction_rate,
        "gross_to_net_ratio": gross_to_net_ratio,
    }

    if "categorie_libelle" in df.columns:
        by_cat = (
            df.groupby("categorie_libelle")["salaire_net"]
            .agg(["mean", "median", "min", "max", "count"])
            .round(2)
        )
        payload["salaire_par_categorie"] = {
            cat: {
                "moyenne": float(row["mean"]),
                "mediane": float(row["median"]),
                "minimum": float(row["min"]),
                "maximum": float(row["max"]),
                "effectif": int(row["count"]),
            }
            for cat, row in by_cat.iterrows()
        }
        top_categories = by_cat.sort_values("mean", ascending=False).head(5)
        payload["categorie_rankings"] = [
            {
                "categorie": str(index),
                "salaire_moyen": float(values["mean"]),
                "effectif": int(values["count"]),
            }
            for index, values in top_categories.iterrows()
        ]

    if "service_libelle" in df.columns:
        by_service = (
            df.groupby("service_libelle")
            .agg(
                salaire_net_mean=("salaire_net", "mean"),
                salaire_net_sum=("salaire_net", "sum"),
                salaire_brut_mean=("salaire_brut", "mean"),
                primes_sum=("montant_primes", "sum"),
                effectif=("mat_pers", "nunique"),
            )
            .round(2)
        )
        payload["salaire_par_service"] = {
            service or "Service inconnu": {
                "moyenne_nette": float(row["salaire_net_mean"]),
                "moyenne_brute": float(row["salaire_brut_mean"]),
                "masse_salariale": float(row["salaire_net_sum"]),
                "total_primes": float(row["primes_sum"]),
                "effectif": int(row["effectif"]),
            }
            for service, row in by_service.iterrows()
        }

        top_services = by_service.sort_values("salaire_net_mean", ascending=False).head(5)
        payload["service_rankings"] = [
            {
                "service": str(index or "Service inconnu"),
                "salaire_moyen": float(values["salaire_net_mean"]),
                "masse": float(values["salaire_net_sum"]),
                "effectif": int(values["effectif"]),
            }
            for index, values in top_services.iterrows()
        ]

    payload["distribution_salaire"] = _build_salary_distribution(df["salaire_net"])

    if "sexe" in df.columns:
        by_gender = (
            df.groupby("sexe")["salaire_net"]
            .agg(["mean", "median", "count"])
            .round(2)
        )
        payload["ecart_par_genre"] = {
            str(genre or "Non renseigné"): {
                "moyenne": float(row["mean"]),
                "mediane": float(row["median"]),
                "effectif": int(row["count"]),
            }
            for genre, row in by_gender.iterrows()
        }

    top_primes = (
        df.groupby(["mat_pers", "service_libelle"], dropna=False)["montant_primes"]
        .sum()
        .reset_index()
    )
    top_primes = top_primes[top_primes["montant_primes"] > 0].sort_values("montant_primes", ascending=False).head(10)
    payload["top_primes"] = [
        {
            "mat_pers": str(row["mat_pers"]),
            "service": str(row["service_libelle"] or "Service inconnu"),
            "montant": round(_safe_float(row["montant_primes"]), 2),
        }
        for _, row in top_primes.iterrows()
    ]

    payload["employee_monthly_breakdown"] = _employee_monthly_breakdown(df)

    return payload


def _compute_workforce(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}

    demographics: Dict[str, Dict[str, float]] = {}
    if "tranche_age" in df.columns and df["tranche_age"].notna().any():
        demographics["repartition_age"] = (
            df["tranche_age"].fillna("Inconnu").value_counts().sort_index().astype(int).to_dict()
        )
    elif "age" in df.columns:
        demographics["repartition_age"] = _bucket_age(df["age"]) or {}

    if "sexe" in df.columns:
        demographics["repartition_genre"] = (
            df["sexe"].fillna("Non renseigné").value_counts().astype(int).to_dict()
        )

    if "anciennete" in df.columns:
        demographics["repartition_anciennete"] = _bucket_anciennete(df["anciennete"]) or {}

    effectifs_service = (
        df.groupby("service_libelle", dropna=False)["mat_pers"].nunique().sort_values(ascending=False)
    )
    effectifs_service = effectifs_service.head(15)
    effectif_total = int(df["mat_pers"].nunique())
    effectifs_categorie = (
        df.groupby("categorie_libelle", dropna=False)["mat_pers"].nunique().sort_values(ascending=False)
    )
    effectifs_categorie = effectifs_categorie.head(15)

    return {
        "analyse_demographique": demographics,
        "effectifs_par_service": {
            str(service or "Service inconnu"): int(count)
            for service, count in effectifs_service.items()
        },
        "effectifs_par_categorie": {
            str(cat or "Non classé"): int(count)
            for cat, count in effectifs_categorie.items()
        },
        "effectif_total": effectif_total,
        "age_moyen": round(_safe_float(df["age"].mean()), 1) if "age" in df.columns else 0.0,
        "anciennete_moyenne": round(_safe_float(df["anciennete"].mean()), 1)
        if "anciennete" in df.columns
        else 0.0,
        "nombre_enfants_total": int(df.get("nombre_enfants", pd.Series(dtype=float)).fillna(0).sum()),
    }


def _compute_financial_health(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if df.empty:
        return {}

    total_cnss = _safe_float(df["montant_cnss"].sum())
    total_impots = _safe_float(df["montant_impots"].sum())
    total_assurance = _safe_float(df["montant_assurance"].sum())
    charges = df["montant_cnss"] + df["montant_impots"] + df["montant_assurance"]
    charges_total = _safe_float(charges.sum())
    salaire_total = _safe_float(df["salaire_brut"].sum())
    avances = df[df["montant_avances"] > 0]
    primes = df[df["montant_primes"] > 0]

    total_employes = df["mat_pers"].nunique()
    employes_avance = avances["mat_pers"].nunique()
    employes_primes = primes["mat_pers"].nunique()

    avance_pct = _format_ratio(employes_avance, total_employes)
    charges_ratio = _format_ratio(charges_total, salaire_total)
    primes_ratio = _format_ratio(primes["montant_primes"].sum(), salaire_total)
    cnss_ratio = _format_ratio(total_cnss, salaire_total)
    tax_ratio = _format_ratio(total_impots, salaire_total)
    deduction_rate = _format_ratio(max(salaire_total - _safe_float(df["salaire_net"].sum()), 0.0), salaire_total)

    score = 100
    if avance_pct > 0.3:
        score -= 30
    elif avance_pct > 0.15:
        score -= 15
    elif avance_pct > 0.05:
        score -= 5

    if charges_ratio > 0.35:
        score -= 25
    elif charges_ratio > 0.25:
        score -= 15
    elif charges_ratio > 0.18:
        score -= 5

    if primes_ratio > 0.25:
        score += 8
    elif primes_ratio > 0.15:
        score += 4

    score = max(0, min(100, score))

    monthly = (
        df.assign(charges=charges)
        .groupby(["annee", "mois", "nom_mois"], dropna=False)
        .agg(
            avances_sum=("montant_avances", "sum"),
            charges_sum=("charges", "sum"),
            primes_sum=("montant_primes", "sum"),
            salaire_net_sum=("salaire_net", "sum"),
        )
        .reset_index()
        .sort_values(["annee", "mois"])
    )

    tendance = [
        {
            "label": _month_label(_safe_int(row["annee"]), _safe_int(row["mois"]), row.get("nom_mois")),
            "avances": round(_safe_float(row["avances_sum"]), 2),
            "charges": round(_safe_float(row["charges_sum"]), 2),
            "primes": round(_safe_float(row["primes_sum"]), 2),
            "salaire_net": round(_safe_float(row["salaire_net_sum"]), 2),
        }
        for _, row in monthly.iterrows()
    ]

    return {
        "indicateurs_avances": {
            "employes": int(employes_avance),
            "pourcentage": round(avance_pct * 100, 2),
            "montant_total": round(_safe_float(avances["montant_avances"].sum()), 2),
            "montant_moyen": round(_safe_float(avances["montant_avances"].mean()), 2)
            if not avances.empty
            else 0.0,
        },
        "indicateurs_charges": {
            "montant_total": round(charges_total, 2),
            "ratio_brut": round(charges_ratio * 100, 2),
            "par_employe": round(_safe_float(charges_total / total_employes), 2)
            if total_employes
            else 0.0,
            "total_cnss": round(total_cnss, 2),
            "taux_cnss": round(cnss_ratio * 100, 2),
            "total_impots": round(total_impots, 2),
            "taux_impots": round(tax_ratio * 100, 2),
            "total_assurance": round(total_assurance, 2),
            "deduction_rate": round(deduction_rate * 100, 2),
        },
        "indicateurs_primes": {
            "employes": int(employes_primes),
            "pourcentage": round(primes_ratio * 100, 2),
            "montant_total": round(_safe_float(primes["montant_primes"].sum()), 2),
            "montant_moyen": round(_safe_float(primes["montant_primes"].mean()), 2)
            if not primes.empty
            else 0.0,
        },
        "score_sante_financiere": score,
        "tendance_mensuelle": tendance,
    }


def _compute_top_earners(df: pd.DataFrame) -> Dict[str, List[Dict[str, float]]]:
    if df.empty:
        return {"top_earners_by_service": {}}

    net_by_employee = (
        df.groupby(["mat_pers", "service_libelle", "categorie_libelle", "sexe"], dropna=False)[
            "salaire_net"
        ]
        .sum()
        .reset_index()
    )
    response: Dict[str, List[Dict[str, float]]] = {}
    for service, group in net_by_employee.groupby("service_libelle"):
        top = group.sort_values("salaire_net", ascending=False).head(5)
        response[str(service or "Service inconnu")] = [
            {
                "mat_pers": str(row["mat_pers"]),
                "montant": round(_safe_float(row["salaire_net"]), 2),
                "categorie": str(row["categorie_libelle"] or "N/A"),
                "sexe": str(row["sexe"] or "N/A"),
            }
            for _, row in top.iterrows()
        ]
    return {"top_earners_by_service": response}


def _compute_historical_trends(df: pd.DataFrame) -> Dict[str, List[float]]:
    if df.empty:
        return {"years": [], "masse_nette": [], "masse_brute": [], "charges": []}

    charges = df["montant_cnss"] + df["montant_impots"] + df["montant_assurance"]
    per_year = (
        df.assign(charges=charges)
        .groupby("annee", dropna=True)
        .agg(
            salaire_net_sum=("salaire_net", "sum"),
            salaire_brut_sum=("salaire_brut", "sum"),
            charges_sum=("charges", "sum"),
        )
        .reset_index()
        .sort_values("annee")
    )

    years = per_year["annee"].astype(int).tolist()
    return {
        "years": years,
        "masse_nette": [round(_safe_float(val), 2) for val in per_year["salaire_net_sum"].tolist()],
        "masse_brute": [round(_safe_float(val), 2) for val in per_year["salaire_brut_sum"].tolist()],
        "charges": [round(_safe_float(val), 2) for val in per_year["charges_sum"].tolist()],
    }


def _build_pdf_report(year: Optional[int], overview: Dict[str, float], salary: Dict[str, Dict[str, float]], financial: Dict[str, Dict[str, float]]) -> io.BytesIO:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=14, leading=18, spaceAfter=12, textColor="#1f2937", fontName="Helvetica-Bold"))
    styles.add(ParagraphStyle(name="BodySmall", fontSize=10, leading=14, spaceAfter=8))

    story: List = []
    title = f"Rapport Paie {year}" if year else "Rapport Paie - Toutes années"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.25 * inch))

    highlights: List[ListItem] = []
    if overview:
        highlights.append(ListItem(Paragraph(
            f"Masse salariale brute : <b>{_format_currency(overview.get('masse_salariale_brute'))} TND</b>", styles["BodySmall"]), leftIndent=12))
        highlights.append(ListItem(Paragraph(
            f"Effectif couvert : <b>{overview.get('total_employes', 0)}</b> collaborateurs", styles["BodySmall"]), leftIndent=12))
        highlights.append(ListItem(Paragraph(
            f"Salaire net moyen : <b>{_format_currency(overview.get('salaire_net_moyen'))} TND</b>", styles["BodySmall"]), leftIndent=12))
        if overview.get("variation_brut_pct") is not None:
            trend = overview.get("variation_brut_pct", 0.0)
            direction = "hausse" if trend > 0 else "baisse" if trend < 0 else "stabilité"
            highlights.append(ListItem(Paragraph(
                f"Variation annuelle masse brute : <b>{_format_percent(trend / 100)}</b> ({direction})", styles["BodySmall"]), leftIndent=12))
        highlights.append(ListItem(Paragraph(
            f"Ratio primes : <b>{_format_percent(overview.get('bonus_ratio'))}</b>", styles["BodySmall"]), leftIndent=12))
        highlights.append(ListItem(Paragraph(
            f"Taux de déductions : <b>{_format_percent(overview.get('deduction_rate'))}</b>", styles["BodySmall"]), leftIndent=12))

    story.append(Paragraph("Résumé exécutif", styles["SectionHeader"]))
    if highlights:
        story.append(Paragraph("Les points marquants de la période étudiée sont résumés ci-dessous.", styles["BodySmall"]))
        story.append(ListFlowable(highlights, bulletType="bullet"))
    else:
        story.append(Paragraph("Aucune donnée disponible pour établir le résumé exécutif.", styles["BodySmall"]))
    story.append(Spacer(1, 0.2 * inch))

    if overview:
        story.append(Paragraph("Synthèse chiffrée", styles["SectionHeader"]))
        overview_table = [
            ["Indicateur", "Valeur"],
            ["Total employés", overview.get("total_employes", 0)],
            ["Services / Catégories", f"{overview.get('nombre_services', 0)} / {overview.get('nombre_categories', 0)}"],
            ["Masse salariale brute", f"{_format_currency(overview.get('masse_salariale_brute'))} TND"],
            ["Masse salariale nette", f"{_format_currency(overview.get('masse_salariale_nette'))} TND"],
            ["Salaire net moyen", f"{_format_currency(overview.get('salaire_net_moyen'))} TND"],
            ["Primes", f"{_format_currency(overview.get('total_primes'))} TND ({_format_percent(overview.get('bonus_ratio'))})"],
            ["Charges totales", f"{_format_currency(overview.get('total_charges'))} TND ({_format_percent(overview.get('ratio_charges_brut'))})"],
            ["CNSS", f"{_format_currency(overview.get('total_cnss'))} TND ({_format_percent(overview.get('cnss_rate'))})"],
            ["Impôts", f"{_format_currency(overview.get('total_impots'))} TND ({_format_percent(overview.get('tax_rate'))})"],
            ["Absence moyenne", f"{overview.get('average_absence_par_employe', 0):.2f} j / collaborateur"],
            ["Coût estimé des absences", f"{_format_currency(overview.get('absence_cost_estime'))} TND"],
        ]
        table = Table(overview_table, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Analyse", styles["SectionHeader"]))
        narrative: List[str] = []
        if overview.get("bonus_ratio", 0) > 0.2:
            narrative.append("La part des primes dépasse 20% de la masse brute, traduisant une politique de rémunération variable dynamique.")
        elif overview.get("bonus_ratio", 0) < 0.08:
            narrative.append("Le poids des primes demeure limité (< 8%), ce qui pourrait restreindre les leviers de reconnaissance individuelle.")

        if overview.get("deduction_rate", 0) > 0.35:
            narrative.append("Le taux de déductions dépasse 35% : un suivi fin des retenues est recommandé pour préserver le pouvoir d'achat.")
        else:
            narrative.append("Les retenues restent contenues et soutiennent un net-to-gross compétitif.")

        if overview.get("average_absence_par_employe", 0) > 2:
            narrative.append("Les absences moyennes dépassent deux jours par collaborateur ; un plan de prévention ciblé peut être envisagé.")

        if not narrative:
            narrative.append("Les indicateurs globaux demeurent dans les fourchettes cibles et confirment la maîtrise de la masse salariale.")

        story.append(ListFlowable([ListItem(Paragraph(text, styles["BodySmall"]), leftIndent=12) for text in narrative], bulletType="bullet"))
        story.append(Spacer(1, 0.25 * inch))

    salaire_service = salary.get("salaire_par_service") if salary else None
    if salaire_service:
        story.append(Paragraph("Salaires par service", styles["SectionHeader"]))
        rows = [["Service", "Effectif", "Salaire net moyen", "Primes totales"]]
        for service, values in list(salaire_service.items())[:10]:
            rows.append([
                service,
                values.get("effectif", 0),
                f"{_format_currency(values.get('moyenne_nette'))} TND",
                f"{_format_currency(values.get('total_primes'))} TND",
            ])
        table = Table(rows, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

    service_rankings = salary.get("service_rankings") if salary else None
    if service_rankings:
        story.append(Paragraph("Top 5 services (salaire net moyen)", styles["SectionHeader"]))
        rows = [["Service", "Salaire net moyen", "Masse nette", "Effectif"]]
        for entry in service_rankings[:5]:
            rows.append([
                entry.get("service", "Service"),
                f"{_format_currency(entry.get('salaire_moyen'))} TND",
                f"{_format_currency(entry.get('masse'))} TND",
                entry.get("effectif", 0),
            ])
        table = Table(rows, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

    categorie_rankings = salary.get("categorie_rankings") if salary else None
    if categorie_rankings:
        story.append(Paragraph("Top 5 catégories (salaire net moyen)", styles["SectionHeader"]))
        rows = [["Catégorie", "Salaire net moyen", "Effectif"]]
        for entry in categorie_rankings[:5]:
            rows.append([
                entry.get("categorie", "Catégorie"),
                f"{_format_currency(entry.get('salaire_moyen'))} TND",
                entry.get("effectif", 0),
            ])
        table = Table(rows, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.2 * inch))

    if financial:
        story.append(Paragraph("Santé financière", styles["SectionHeader"]))
        fin_table = [
            ["Indicateur", "Valeur"],
            ["Score", financial.get("score_sante_financiere", 0)],
            ["Avances - % employés", f"{financial.get('indicateurs_avances', {}).get('pourcentage', 0):.2f}%"],
            ["Avances - montant total", f"{_format_currency(financial.get('indicateurs_avances', {}).get('montant_total'))} TND"],
            ["Charges - ratio brut", f"{financial.get('indicateurs_charges', {}).get('ratio_brut', 0):.2f}%"],
            ["Charges - montant total", f"{_format_currency(financial.get('indicateurs_charges', {}).get('montant_total'))} TND"],
            ["CNSS", f"{_format_currency(financial.get('indicateurs_charges', {}).get('total_cnss'))} TND ({financial.get('indicateurs_charges', {}).get('taux_cnss', 0):.2f}%)"],
            ["Impôts", f"{_format_currency(financial.get('indicateurs_charges', {}).get('total_impots'))} TND ({financial.get('indicateurs_charges', {}).get('taux_impots', 0):.2f}%)"],
            ["Primes - % employés", f"{financial.get('indicateurs_primes', {}).get('pourcentage', 0):.2f}%"],
        ]
        table = Table(fin_table, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.lightgrey]),
        ]))
        story.append(table)

        recommandations: List[str] = []
        if financial.get("indicateurs_avances", {}).get("pourcentage", 0) > 25:
            recommandations.append("Réduire la pression de trésorerie des collaborateurs via des actions d'éducation financière ou des avances ciblées.")
        if financial.get("indicateurs_charges", {}).get("taux_cnss", 0) > 18:
            recommandations.append("Étudier les leviers d'optimisation des charges sociales (classification, exonérations, équilibre brut/net).")
        if financial.get("indicateurs_charges", {}).get("ratio_brut", 0) > 35:
            recommandations.append("Mettre en place un suivi mensuel des charges pour prévenir les dépassements budgétaires.")
        if not recommandations:
            recommandations.append("Les indicateurs financiers demeurent maîtrisés ; poursuivre la trajectoire actuelle.")

        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph("Recommandations", styles["SectionHeader"]))
        story.append(ListFlowable([ListItem(Paragraph(text, styles["BodySmall"]), leftIndent=12) for text in recommandations], bulletType="bullet"))

    doc.build(story)
    buffer.seek(0)
    return buffer


hr_service = PayrollDataService()


@payroll_bp.route("/payroll")
def payroll_dashboard() -> str:
    return render_template("payroll.html")


@payroll_bp.route("/api/connect", methods=["POST"])
def connect_database():
    try:
        if hr_service.connect():
            return jsonify({"success": True, "message": "Connexion établie"})
        return jsonify({"success": False, "message": "Connexion échouée"}), 500
    except Exception as exc:  # pragma: no cover - logging only
        return jsonify({"success": False, "message": str(exc)}), 500


@payroll_bp.route("/api/years", methods=["GET"])
def get_years():
    years = hr_service.get_available_years()
    return jsonify({"years": years, "current_year": max(years) if years else None})


@payroll_bp.route("/api/kpis/overview", methods=["GET"])
def overview_kpis():
    year = request.args.get("year", type=int)
    df = hr_service.extract_payroll_data(year)
    if df.empty:
        return jsonify({"error": "Aucune donnée disponible"}), 404
    overview = _compute_overview(df, year)
    if year:
        previous_df = hr_service.extract_payroll_data(year - 1)
        if not previous_df.empty:
            previous = _compute_overview(previous_df, year - 1)
            overview["variation_brut_pct"] = _percent_change(
                overview.get("masse_salariale_brute", 0.0), previous.get("masse_salariale_brute", 0.0)
            )
            overview["variation_net_pct"] = _percent_change(
                overview.get("masse_salariale_nette", 0.0), previous.get("masse_salariale_nette", 0.0)
            )
        else:
            overview["variation_brut_pct"] = None
            overview["variation_net_pct"] = None
    else:
        overview["variation_brut_pct"] = None
        overview["variation_net_pct"] = None
    return jsonify(overview)


@payroll_bp.route("/api/kpis/salaires", methods=["GET"])
def salary_kpis():
    year = request.args.get("year", type=int)
    df = hr_service.extract_payroll_data(year)
    if df.empty:
        return jsonify({"error": "Aucune donnée disponible"}), 404
    salary = _compute_salary_insights(df)
    return jsonify(salary)


@payroll_bp.route("/api/kpis/workforce-analytics", methods=["GET"])
def workforce_kpis():
    year = request.args.get("year", type=int)
    df = hr_service.extract_payroll_data(year)
    if df.empty:
        return jsonify({"error": "Aucune donnée disponible"}), 404
    workforce = _compute_workforce(df)
    return jsonify(workforce)


@payroll_bp.route("/api/kpis/financial-health", methods=["GET"])
def financial_kpis():
    year = request.args.get("year", type=int)
    df = hr_service.extract_payroll_data(year)
    if df.empty:
        return jsonify({"error": "Aucune donnée disponible"}), 404
    financial = _compute_financial_health(df)
    return jsonify(financial)


@payroll_bp.route("/api/kpis/top-earners", methods=["GET"])
def top_earners_kpis():
    year = request.args.get("year", type=int)
    df = hr_service.extract_payroll_data(year)
    if df.empty:
        return jsonify({"error": "Aucune donnée disponible"}), 404
    top = _compute_top_earners(df)
    return jsonify(top)


@payroll_bp.route("/api/kpis/historical-trends", methods=["GET"])
def historical_trends():
    df = hr_service.extract_payroll_data()
    if df.empty:
        return jsonify({"years": [], "masse_nette": [], "masse_brute": [], "charges": []})
    trends = _compute_historical_trends(df)
    return jsonify(trends)


@payroll_bp.route("/api/kpis/forecast", methods=["GET"])
def payroll_forecast_projection():
    df = hr_service.extract_payroll_data()
    if df.empty:
        return jsonify({"error": "Aucune donnée disponible"}), 404
    result = _compute_forecast_projection(df)
    if "error" in result:
        return jsonify(result), 422
    return jsonify(result)


@payroll_bp.route("/api/export/pdf", methods=["GET"])
def export_pdf():
    year = request.args.get("year", type=int)
    df = hr_service.extract_payroll_data(year)
    if df.empty:
        return jsonify({"error": "Aucune donnée disponible"}), 404
    overview = _compute_overview(df, year)
    salary = _compute_salary_insights(df)
    financial = _compute_financial_health(df)
    pdf_buffer = _build_pdf_report(year, overview, salary, financial)
    filename = f"rapport_paie_{year}.pdf" if year else "rapport_paie.pdf"
    return send_file(pdf_buffer, mimetype="application/pdf", as_attachment=True, download_name=filename)
