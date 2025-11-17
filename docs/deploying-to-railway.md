# Deploying to Railway

Railway can build the Docker image shipped with this repository and run both the Flask web process and optional Celery workers. Follow these steps to deploy.

## 1. Prepare the repository

- Ensure the latest code (including `Dockerfile` and `Wallet_PFE/`) is pushed to GitHub.
- Confirm environment variables in `.env` are correct; you will recreate them in Railway.

## 2. Create a Railway project

1. Sign up / log in at <https://railway.app/>.
2. Click **New Project → Deploy from GitHub Repo** and grant Railway access to `KingOfUruk/finale` (or your fork).
3. Choose the repository and select the `main` branch. Railway will detect the `Dockerfile` automatically.

## 3. Configure the service

- After the first build, open the service settings and set the **Port** to `8080` (exposed by `gunicorn` in the Dockerfile).
- Optionally rename the service (e.g., `flask-api`) for clarity.

## 4. Set environment variables

Under **Variables**, add the following keys and values (copy them from your local `.env` but do *not* check them into Git):

| Variable | Description |
| --- | --- |
| `SECRET_KEY` | Flask session secret |
| `ORACLE_USERNAME` | Database username |
| `ORACLE_PASSWORD` | Database password |
| `ORACLE_DSN` *or* (`ORACLE_HOST`, `ORACLE_PORT`, `ORACLE_SERVICE_NAME`) | Oracle connection |
| `ORACLE_CONNECT_DESCRIPTOR` | Optional raw descriptor from `tnsnames.ora` |
| `ORACLE_TNS_ADMIN` | Wallet directory inside the container (default: `/app/Wallet_PFE`) |
| `ORACLE_WALLET_PASSWORD` | Wallet password if required |
| `REDIS_URL` / `LOGIN_STATE_REDIS_URL` | Redis broker for Celery + login throttling |
| Feature flags (`ENABLE_PREDICTION_API`, etc.) | Optional toggles |

Because the Docker image copies `Wallet_PFE/` into `/app/Wallet_PFE`, you can usually set `ORACLE_TNS_ADMIN=/app/Wallet_PFE`. If you prefer to keep the wallet outside the repo, store it as a Railway volume or inject it through a base64 environment variable decoded at container start-up.

## 5. (Optional) Provision Redis and Celery workers

1. Add a **Redis** plugin in the Railway project and save the connection URL as `REDIS_URL`.  
2. To run Celery workers on Railway, create a second service pointing to the same repository. In the **Deploy settings**, override the start command to:  
   ```
   celery -A celery_app.celery worker --loglevel=info
   ```  
   Ensure the worker service shares the same environment variables and Redis plugin.

> **Note:** If you opt for the Procfile build instead of Docker, Railway will use the `Procfile` command `sh -c 'gunicorn main:app --bind 0.0.0.0:${PORT:-8080}'`, which expands the `$PORT` environment variable correctly. Without the `sh -c` wrapper you would see repeated logs saying `'$PORT' is not a valid port number`.

## 6. Deploy

Each push to `main` triggers a new Docker build and deployment. You can also redeploy manually from the service page. Once the container is running, Railway exposes a public URL—visit `/healthz` to confirm the app is up.

---

For advanced setups (multiple environments, base64 wallet injection, or infrastructure-as-code), consider using the [Railway CLI](https://docs.railway.app/develop/cli) to version your workflow.
