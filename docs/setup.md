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
ADMIN_USERS=visitor
```

Only the `ADMIN_USERS` listed here can access the access-control dashboard.
Multiple administrators can be declared using a comma-separated list.

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
