# Setup & Configuration Guide

This project relies on an Oracle database. Follow these steps to get the
application running locally.

## 1. Install Python dependencies

Create a virtual environment and install the packages listed in
`requirements.txt` (see below). The Oracle driver is optional at install time
but required for database access.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Install the Oracle driver

The application expects `cx_Oracle` (or the new `oracledb` package) to be
available. Install it with pip:

```bash
pip install cx-Oracle
```

Oracle also requires the Oracle Instant Client libraries to be available on the
machine. Refer to the official guide:
<https://cx-oracle.readthedocs.io/en/latest/user_guide/installation.html>

## 3. Configure environment variables

Create a `.env` file (or supply the variables through your shell) with the
following keys:

```
ORACLE_USERNAME=your_username
ORACLE_PASSWORD=your_password
ORACLE_HOST=localhost
ORACLE_PORT=1521
ORACLE_SERVICE_NAME=ORCLCDB
# ORACLE_DSN=your_wallet_alias  # e.g. mydb_tp
# ORACLE_CONNECT_DESCRIPTOR=(description=...)  # raw descriptor from tnsnames.ora
# ORACLE_TNS_ADMIN=/absolute/path/to/wallet
# ORACLE_WALLET_PASSWORD=wallet_password
ADMIN_USERS=visitor
```

Only the `ADMIN_USERS` listed here can access the access-control dashboard.
Multiple administrators can be declared using a comma-separated list.

If you use an Autonomous Database wallet, either set `ORACLE_DSN` to the alias
declared inside `tnsnames.ora` (and point `ORACLE_TNS_ADMIN` to the wallet
folder) or copy the descriptor line verbatim into `ORACLE_CONNECT_DESCRIPTOR`.
The app will prioritise the raw descriptor, then the DSN alias, and finally the
host/port/service triplet.

## 4. Initialise users

Use `runuser.py` to rebuild the `users` table or to create standard accounts:

```bash
# Rebuild table and seed default admin (visitor / visitor123)
python runuser.py --rebuild

# Create a standard active user
python runuser.py --username alice --email alice@example.com --password MySecret123

# Create an inactive user
python runuser.py --username bob --email bob@example.com --password Temp123 --inactive
```

## 5. Start the application

```bash
flask --app main.py run
```

The dashboards will now connect to the Oracle instance described in the
environment variables. If the driver or database are unavailable, the
application will log a descriptive error instead of crashing.

## 6. Background worker & Redis

Long running ML jobs and login throttling both depend on Redis. After setting
`REDIS_URL` (Celery broker/backend) or `LOGIN_STATE_REDIS_URL` in `.env`, launch
Redis locally (for example via Docker):

```bash
docker run --name redis -p 6379:6379 redis:7
```

Then start at least one Celery worker so `/readyz` reports the service as up:

```bash
celery -A celery_app.celery worker --loglevel=info
```

Scaling the worker fleet is as simple as running the same command on additional
hosts/containers that can reach the broker URL.

## 7. Prometheus scraping

Metrics are exposed automatically on `/metrics` in Prometheus format. Add the
Flask service to your Prometheus configuration, for example:

```yaml
scrape_configs:
  - job_name: "hr-app"
    static_configs:
      - targets: ["app:5000"]  # replace with the Flask host:port
```

Reload Prometheus after editing `prometheus.yml`. You should then see default
HTTP metrics (request counts, latencies) plus the `app_info` gauge.
