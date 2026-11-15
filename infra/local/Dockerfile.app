# Image for CryptoSentinel's Python services (producers, sinks; later scorer and agent).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY common ./common
COPY producers ./producers
COPY processing ./processing

# Settings come from environment variables set by docker compose.
CMD ["python", "-c", "print('choose a module, e.g. python -m producers.price_producer')"]
