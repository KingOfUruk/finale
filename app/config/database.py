"""
Centralised database credential handling.

Credentials are loaded from environment variables with safe defaults for local
development. Import `get_oracle_credentials()` wherever you need Oracle
connection parameters instead of hard-coding them in each module.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict
from urllib.parse import urlencode

_ENV_FILE_LOADED = False


def _load_env_file_once() -> None:
    """Populate os.environ from a local .env file when present."""
    global _ENV_FILE_LOADED
    if _ENV_FILE_LOADED:
        return
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                value = value.strip().strip("'\"")
                os.environ.setdefault(key, value)
    _ENV_FILE_LOADED = True


def get_oracle_credentials() -> Dict[str, str]:
    """
    Return Oracle connection parameters pulled from environment variables.

    - ORACLE_USERNAME
    - ORACLE_PASSWORD
    - ORACLE_HOST
    - ORACLE_PORT
    - ORACLE_SERVICE_NAME
    """
    _load_env_file_once()

    def _read_required(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(
                f"Environment variable '{name}' is required to connect to Oracle. "
                "Set it in your shell or in the project .env file."
            )
        return value

    username = _read_required("ORACLE_USERNAME")
    password = _read_required("ORACLE_PASSWORD")
    dsn = os.getenv("ORACLE_DSN")

    connect_descriptor = os.getenv("ORACLE_CONNECT_DESCRIPTOR")

    host = os.getenv("ORACLE_HOST")
    port = os.getenv("ORACLE_PORT")
    service_name = os.getenv("ORACLE_SERVICE_NAME")

    if not dsn and not connect_descriptor:
        # When DSN alias or raw descriptor not provided, host/port/service remain mandatory
        host = host or _read_required("ORACLE_HOST")
        port = port or _read_required("ORACLE_PORT")
        service_name = service_name or _read_required("ORACLE_SERVICE_NAME")

    return {
        "username": username,
        "password": password,
        "host": host,
        "port": port,
        "service_name": service_name,
        "connect_descriptor": connect_descriptor,
        "driver": os.getenv("ORACLE_DRIVER", "oracle+oracledb"),
        "dsn": dsn,
        "tns_admin": os.getenv("ORACLE_TNS_ADMIN"),
        "wallet_password": os.getenv("ORACLE_WALLET_PASSWORD"),
    }


def build_sqlalchemy_url(credentials: Dict[str, str]) -> str:
    """Build an SQLAlchemy connection URL supporting Oracle wallets/DSN aliases."""

    driver = credentials.get("driver", "oracle+oracledb")
    username = credentials["username"]
    password = credentials["password"]

    dsn = credentials.get("dsn")
    if dsn:
        base = f"{driver}://{username}:{password}@{dsn}"
    else:
        host = credentials["host"]
        port = credentials["port"]
        service = credentials["service_name"]
        base = f"{driver}://{username}:{password}@{host}:{port}/?service_name={service}"

    query_params = []
    tns_admin = credentials.get("tns_admin")
    if tns_admin:
        query_params.append(("config_dir", tns_admin))
        query_params.append(("wallet_location", tns_admin))
    wallet_password = credentials.get("wallet_password")
    if wallet_password:
        query_params.append(("wallet_password", wallet_password))

    if query_params:
        separator = "&" if "?" in base else "?"
        base = f"{base}{separator}{urlencode(query_params)}"

    return base
