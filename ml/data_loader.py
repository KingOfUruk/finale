"""
Common data loading utilities for ML modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from app.config.database import get_oracle_credentials, build_sqlalchemy_url


@dataclass
class DatabaseConfig:
    """Configuration container for Oracle connectivity."""

    username: str
    password: str
    host: Optional[str]
    port: Optional[int]
    service_name: Optional[str]
    driver: str = "oracle+oracledb"
    dsn: Optional[str] = None
    tns_admin: Optional[str] = None
    wallet_password: Optional[str] = None

    @classmethod
    def from_settings(cls) -> "DatabaseConfig":
        creds = get_oracle_credentials()
        port_val = creds.get("port")
        if isinstance(port_val, str) and port_val.isdigit():
            port_val = int(port_val)
        return cls(
            username=creds["username"],
            password=creds["password"],
            host=creds.get("host"),
            port=port_val,
            service_name=creds.get("service_name"),
            driver=creds.get("driver", "oracle+oracledb"),
            dsn=creds.get("dsn"),
            tns_admin=creds.get("tns_admin"),
            wallet_password=creds.get("wallet_password"),
        )

    def sqlalchemy_url(self) -> str:
        credentials = {
            "username": self.username,
            "password": self.password,
            "host": self.host,
            "port": self.port,
            "service_name": self.service_name,
            "driver": self.driver,
            "dsn": self.dsn,
            "tns_admin": self.tns_admin,
            "wallet_password": self.wallet_password,
        }
        return build_sqlalchemy_url(credentials)


class DataLoader:
    """Helper to pull paie / pointage datasets for ML training."""

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig.from_settings()
        self._engine = create_engine(self.config.sqlalchemy_url())

    def fetch_attendance(self, start_year: Optional[int] = None) -> pd.DataFrame:
        """Return aggregated pointage data."""
        where_clause = ""
        if start_year:
            where_clause = "WHERE dt.annee >= :start_year"

        query = text(
            f"""
            SELECT fp.mat_pers,
                   dt.date_jour,
                   dt.annee,
                   dt.mois,
                   dt.num_semaine,
                   dt.jour_semaine,
                   fp.nbr_pointages,
                   fp.duree_minutes,
                   serv.libelle AS service_libelle,
                   emp.lib_cat
            FROM FAIT_POINTAGE fp
            JOIN DIM_TEMPS_nouveau dt ON fp.id_temps = dt.id_temps
            JOIN DIM_EMPLOYEe_nouveau emp ON fp.mat_pers = emp.mat_pers
            LEFT JOIN DIM_SERVICE serv ON emp.code_serv = serv.code_serv
            {where_clause}
            """
        )
        params = {"start_year": start_year} if start_year else {}
        return pd.read_sql(query, self._engine, params=params)

    def fetch_payroll(self, start_year: Optional[int] = None) -> pd.DataFrame:
        """Return payroll facts enriched with employee + time dimensions."""
        where_clause = ""
        if start_year:
            where_clause = "WHERE dt.annee >= :start_year"

        query = text(
            f"""
            SELECT fr.id_fact,
                   fr.mat_pers,
                   fr.montant,
                   fr.source,
                   dt.annee,
                   dt.mois,
                   dt.trimestre,
                   emp.lib_cat,
                   emp.code_serv,
                   serv.libelle AS service_libelle,
                   emp.age,
                   emp.anciennete
            FROM FAIT_remuneration fr
            JOIN DIM_TEMPS_nouveau dt ON fr.id_temps = dt.id_temps
            LEFT JOIN DIM_EMPLOYEe_nouveau emp ON fr.mat_pers = emp.mat_pers
            LEFT JOIN DIM_SERVICE serv ON emp.code_serv = serv.code_serv
            {where_clause}
            """
        )
        params = {"start_year": start_year} if start_year else {}
        return pd.read_sql(query, self._engine, params=params)

    def fetch_headcount_snapshot(self) -> pd.DataFrame:
        """Return the most recent employee snapshot for benchmarking features."""
        query = text(
            """
            SELECT mat_pers,
                   sexe,
                   age,
                   anciennete,
                   lib_cat,
                   code_serv,
                   libelle AS service_libelle,
                   statut
            FROM DIM_EMPLOYEe_nouveau
            """
        )
        return pd.read_sql(query, self._engine)
