import logging
import os
import sys
from typing import Any
from uuid import uuid4

from flask import Flask, jsonify, g, request
from prometheus_flask_exporter import PrometheusMetrics
from app.blueprints.login import login_bp
from app.blueprints.payroll import payroll_bp
from app.blueprints.hr_analytics import hr_analytics_bp
from app.blueprints.employee_performance import employee_performance_bp
from app.blueprints.ml_api import ml_bp
from app.config.database import get_oracle_credentials
from app.oracle_helpers import OracleError, create_connection_from_credentials
from celery_app import celery
from pythonjsonlogger import jsonlogger

try:  # redis is optional unless readiness checks include it
    import redis  # type: ignore
except ImportError:  # pragma: no cover - redis optional
    redis = None


def _require_secret_key() -> str:
    key = os.environ.get("SECRET_KEY")
    if not key:
        raise RuntimeError("SECRET_KEY must be provided via environment variable.")
    return key


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging helper
        try:
            record.request_id = getattr(g, "request_id", "-")
            record.path = getattr(request, "path", "-")
            record.method = getattr(request, "method", "-")
        except RuntimeError:
            record.request_id = record.path = record.method = "-"
        return True


def _configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(path)s %(method)s"
    )
    handler.setFormatter(formatter)
    handler.addFilter(RequestContextFilter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)


_configure_logging()


app = Flask(__name__)
app.secret_key = _require_secret_key()
metrics = PrometheusMetrics(app)
metrics.info("app_info", "Application info", version="1.0.0")


def _check_oracle() -> bool:
    try:
        credentials = get_oracle_credentials()
        connection = create_connection_from_credentials(credentials)
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM dual")
        cursor.close()
        connection.close()
        return True
    except (OracleError, RuntimeError):
        return False
    except Exception:
        return False


def _check_redis() -> bool:
    redis_url = os.getenv("REDIS_URL") or os.getenv("LOGIN_STATE_REDIS_URL")
    if not redis_url or redis is None:
        return False
    try:
        client = redis.Redis.from_url(redis_url)
        return bool(client.ping())
    except Exception:
        return False


def _check_celery() -> bool:
    try:
        responses = celery.control.ping(timeout=1)
        return bool(responses)
    except Exception:
        return False


@app.before_request
def _assign_request_id():  # pragma: no cover - request hook
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    g.request_id = request_id


@app.after_request
def _inject_request_id(response):  # pragma: no cover - response hook
    if hasattr(g, "request_id"):
        response.headers["X-Request-ID"] = g.request_id
    return response

# Register blueprints
app.register_blueprint(login_bp)
app.register_blueprint(payroll_bp)
app.register_blueprint(hr_analytics_bp)
app.register_blueprint(employee_performance_bp)
app.register_blueprint(ml_bp)


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/readyz")
def readyz():
    status = {
        "flask": True,
        "oracle": _check_oracle(),
        "redis": _check_redis(),
        "celery": _check_celery(),
    }
    http_code = 200 if all(status.values()) else 503
    return jsonify(status), http_code

if os.environ.get("ENABLE_PREDICTION_API") == "1":
    from app.blueprints.prediction import prediction_bp

    app.register_blueprint(prediction_bp)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
