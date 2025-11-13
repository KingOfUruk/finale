"""Celery application setup for background jobs."""
from __future__ import annotations

import os
from pathlib import Path

from celery import Celery

_ENV_LOADED = False


def _load_env_file_once() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            value = value.strip().strip("'\"")
            os.environ.setdefault(key, value)
    _ENV_LOADED = True


def create_celery() -> Celery:
    _load_env_file_once()
    redis_url = (
        os.getenv("REDIS_URL")
        or os.getenv("LOGIN_STATE_REDIS_URL")
        or "redis://127.0.0.1:6379/0"
    )
    celery = Celery(
        "proj",
        broker=redis_url,
        backend=redis_url,
        include=["ml.tasks"],
    )
    celery.conf.update(
        task_track_started=True,
        result_expires=3600,
    )
    return celery


celery = create_celery()
