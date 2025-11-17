FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libaio1t64 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV ORACLE_TNS_ADMIN=/app/Wallet_PFE \
    PORT=8080

CMD ["python", "-m", "gunicorn", "main:app", "--bind", "0.0.0.0:8080"]
