"""Celery tasks for ML pipelines."""
from __future__ import annotations

import logging
from typing import Optional

from celery import shared_task

from celery_app import celery
from ml.training_pipeline import run_training_pipeline


@celery.task(name="ml.train_all_models")
def train_all_models(start_year: Optional[int] = None):
    logging.info("Starting training job (start_year=%s)", start_year)
    summary = run_training_pipeline(start_year)
    logging.info("Training job completed")
    return summary
