"""
Centralised database credential handling.

Credentials are loaded from environment variables with safe defaults for local
development. Import `get_oracle_credentials()` wherever you need Oracle
connection parameters instead of hard-coding them in each module.
"""
from __future__ import annotations

import os
from typing import Dict


def get_oracle_credentials() -> Dict[str, str]:
    """
    Return Oracle connection parameters pulled from environment variables.

    - ORACLE_USERNAME
    - ORACLE_PASSWORD
    - ORACLE_HOST
    - ORACLE_PORT
    - ORACLE_SERVICE_NAME
    """
    return {
        "username": os.getenv("ORACLE_USERNAME", "feldway"),
        "password": os.getenv("ORACLE_PASSWORD", "newpassword123"),
        "host": os.getenv("ORACLE_HOST", "localhost"),
        "port": os.getenv("ORACLE_PORT", "1521"),
        "service_name": os.getenv("ORACLE_SERVICE_NAME", "projetfinetude"),
        "driver": os.getenv("ORACLE_DRIVER", "oracle+oracledb"),
    }
