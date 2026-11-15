# CryptoSentinel — common commands. Run them from the repo root, inside WSL/Linux/macOS.
#   make help

COMPOSE := docker compose -f infra/local/docker-compose.yml
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip

.DEFAULT_GOAL := help
.PHONY: help venv env up down ps logs init smoke test check reset collect collect-stop collect-logs report history parquet shocks replay newstime features features-live parquet-news train train-vol train-anomaly score-live

help:  ## List the commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  make %-8s %s\n", $$1, $$2}'

venv:  ## Create .venv and install Python dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

env:  ## Create .env.local from .env.example (does not overwrite)
	@test -f .env.local && echo ".env.local already exists" || (cp .env.example .env.local && echo "created .env.local")

up:  ## Start Kafka, Kafbat UI, SeaweedFS and DynamoDB Local
	$(COMPOSE) up -d
	@echo "Waiting for Kafka to be healthy..."
	@for i in $$(seq 1 30); do \
	  if [ "$$(docker inspect -f '{{.State.Health.Status}}' cs-kafka 2>/dev/null)" = "healthy" ]; then echo "Kafka is up."; exit 0; fi; \
	  sleep 3; done; echo "Kafka did not become healthy: run 'make logs'"; exit 1

down:  ## Stop everything, collectors included (data is kept)
	$(COMPOSE) --profile collect down

ps:  ## Show container status
	$(COMPOSE) ps

logs:  ## Follow the logs of every container
	$(COMPOSE) logs -f --tail=100

init:  ## Create Kafka topics, the S3 bucket and DynamoDB tables (idempotent)
	$(PY) -m infra.local.init_local

smoke:  ## Phase 1 check: Kafka + S3 + DynamoDB round trip
	$(PY) -m infra.local.smoke

test:  ## Run the unit tests (no Docker needed)
	$(PY) -m pytest -q

check: up init smoke  ## up + init + smoke in one go

reset:  ## DANGER: stop the stack and delete all local data
	$(COMPOSE) --profile collect down -v

# ---- Phase 2: data collection ------------------------------------------------
collect: up init  ## Build and start the 4 collectors in Docker (they keep running)
	$(COMPOSE) --profile collect up -d --build price-producer news-producer s3-sink hot-writer
	@echo "Collectors started. Check them with: make collect-logs   and   make report"

collect-stop:  ## Stop the 4 collectors (Kafka and storage keep running)
	$(COMPOSE) --profile collect stop price-producer news-producer s3-sink hot-writer

collect-logs:  ## Follow the collectors' logs (Ctrl+C to quit, they keep running)
	$(COMPOSE) --profile collect logs -f --tail=30 price-producer news-producer s3-sink hot-writer

report:  ## How is collection going? Files in S3, latest candle per coin, news per coin
	$(PY) -m processing.collection_report

# ---- Phase 3: history, Parquet, shocks, replay ---------------------------------
history:  ## Download 12 months of Binance candles + the news dataset into S3 raw (resumable)
	$(PY) -m processing.backfill_history

parquet:  ## Raw zone -> Parquet (deduplicated) + data quality report
	$(PY) -m processing.raw_to_parquet

shocks:  ## Biggest shock EVENTS (market-wide / coin-specific) + news before -> eval/shock_events.csv
	$(PY) -m processing.find_shocks --top 20

newstime:  ## Check that news timestamps are UTC (run `make shocks` first)
	$(PY) -m processing.check_news_time

parquet-news:  ## Rebuild processed news (after setting NEWS_TIME_SHIFT_MINUTES in .env.local)
	$(PY) -m processing.raw_to_parquet --rebuild-news --no-report

DAY   ?= 2024-08-05
FROM  ?= 00:00
TO    ?= 04:00
SPEED ?= 10
replay:  ## Replay history into Kafka, e.g. make replay DAY=2024-08-05 FROM=00:00 TO=04:00 SPEED=10
	$(PY) -m replay.replay --start "$(DAY) $(FROM)" --end "$(DAY) $(TO)" --speed $(SPEED)

# ---- Phase 4: features --------------------------------------------------------
features:  ## Build the training table: processed/features/month=.../data.parquet (+ target)
	$(PY) -m processing.build_features

features-live:  ## Live features: prices.klines -> features.1m (start it BEFORE make replay)
	$(PY) -m processing.feature_consumer

# ---- Phase 5: models ----------------------------------------------------------
train-vol:  ## Volatility forecaster vs naive / HAR-RV / GARCH -> eval/results_vol.csv, models/
	$(PY) -m ml.train_vol

train-anomaly:  ## Anomaly detector vs z-rule at the same alert budget -> eval/results_anomaly.csv
	$(PY) -m ml.train_anomaly

train: train-vol train-anomaly  ## Both models

score-live:  ## Live scoring: features.1m -> alerts.anomaly (start after features-live)
	$(PY) -m processing.live_scorer
