"""Utility helpers for Oracle connectivity via the thin drivers.

Centralises import handling so the rest of the codebase can provide
actionable error messages when the Oracle driver is missing.
"""

from __future__ import annotations

import os
from typing import Any, Dict

try:  # pragma: no cover - runtime environment dependent
    import oracledb as _oracle_driver  # type: ignore
except ImportError:  # pragma: no cover - executed when python-oracledb absent
    try:
        import cx_Oracle as _oracle_driver  # type: ignore
    except ImportError as exc:  # pragma: no cover - executed when both absent
        _oracle_driver = None  # type: ignore
        ORACLE_IMPORT_ERROR = exc

        class OracleError(Exception):
            """Fallback error type when no Oracle driver is available."""

    else:  # pragma: no cover - executed when cx_Oracle found
        ORACLE_IMPORT_ERROR = None
        OracleError = _oracle_driver.DatabaseError  # type: ignore

else:  # pragma: no cover - executed when python-oracledb found
    ORACLE_IMPORT_ERROR = None
    OracleError = _oracle_driver.DatabaseError  # type: ignore


def _require_driver() -> Any:
    """Return the loaded Oracle driver module or raise a helpful runtime error."""

    if _oracle_driver is None:
        message = (
            "Aucun pilote Oracle n'est installé.\n"
            "Installez `python-oracledb` (recommandé) ou `cx-Oracle`,\n"
            "puis vérifiez que les bibliothèques clientes Oracle sont accessibles si nécessaire."
        )
        raise RuntimeError(message) from ORACLE_IMPORT_ERROR
    return _oracle_driver


def create_connection_from_credentials(credentials: Dict[str, str]):
    """Create an Oracle connection using a credentials dictionary."""

    driver = _require_driver()
    connect_kwargs = {
        "user": credentials["username"],
        "password": credentials["password"],
    }

    tns_admin = credentials.get("tns_admin")
    if tns_admin:
        os.environ.setdefault("TNS_ADMIN", tns_admin)
        connect_kwargs.setdefault("config_dir", tns_admin)
        connect_kwargs.setdefault("wallet_location", tns_admin)

    connect_descriptor = credentials.get("connect_descriptor")
    dsn_alias = credentials.get("dsn")
    if connect_descriptor:
        connect_kwargs["dsn"] = connect_descriptor
    elif dsn_alias:
        connect_kwargs["dsn"] = dsn_alias
    else:
        dsn = f"{credentials['host']}:{credentials['port']}/{credentials['service_name']}"
        connect_kwargs["dsn"] = dsn

    wallet_password = credentials.get("wallet_password")
    if wallet_password:
        connect_kwargs["wallet_password"] = wallet_password

    return driver.connect(**connect_kwargs)


def make_dsn(host: str, port: str, service_name: str) -> str:
    """Return an Oracle DSN string, using makedsn when available."""

    driver = _require_driver()
    if hasattr(driver, "makedsn"):
        return driver.makedsn(host, port, service_name=service_name)
    return f"{host}:{port}/{service_name}"


__all__ = [
    "OracleError",
    "ORACLE_IMPORT_ERROR",
    "create_connection_from_credentials",
    "make_dsn",
]
