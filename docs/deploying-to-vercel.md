# Deploying to Vercel

The project now ships with a Vercel-compatible serverless entrypoint (`api/index.py`) and a `vercel.json` configuration. Follow the steps below to produce a successful deployment.

1. **Install the Vercel CLI (optional but convenient)**  
   ```bash
   npm install -g vercel
   ```

2. **Create the required environment variables**  
   In the Vercel dashboard (or via `vercel env`), define each sensitive value:

   - `SECRET_KEY`
   - `ORACLE_USERNAME`
   - `ORACLE_PASSWORD`
   - `ORACLE_DSN` or the `ORACLE_HOST` / `ORACLE_PORT` / `ORACLE_SERVICE_NAME` trio
   - `ORACLE_CONNECT_DESCRIPTOR` (if you use the wallet connection descriptor)
   - `ORACLE_WALLET_PASSWORD`
   - `ORACLE_TNS_ADMIN` (optional – see below)
   - `REDIS_URL` / `LOGIN_STATE_REDIS_URL` if Redis is reachable from Vercel
   - Any other feature flags you rely on, such as `ENABLE_PREDICTION_API`

3. **Oracle wallet files (`Wallet_PFE`)**  
   The repository includes the wallet directory. The new database helper looks for `ORACLE_TNS_ADMIN` and, when missing, automatically points to `Wallet_PFE` in the repository root. Uploading the wallet to Git is convenient for testing but audit your security posture before deploying to production – you may prefer storing the wallet elsewhere and setting `ORACLE_TNS_ADMIN` to a secure object store mount or a Vercel environment file.

4. **Deploy**  
   ```bash
   vercel --prod
   ```
   The `vercel.json` routes all requests to the serverless Flask handler backed by `api/index.py`. Vercel builds the function with `@vercel/python`, installing `requirements.txt` automatically.

5. **Operational caveats**  
   - Long-running background workers (Celery) are not executed inside Vercel; provision those tasks on a separate worker host if needed. The HTTP routes continue to enqueue tasks but the worker must be running elsewhere.  
   - The readiness endpoints return the status of Oracle, Redis, and Celery probes. Missing infrastructure will show as `false`, which is expected inside the serverless environment unless you provide managed Redis/worker instances.

With these steps you can iterate locally (`vercel dev`) and push production deployments once the necessary managed services are reachable from Vercel's network.
