# CryptoSentinel — common commands. Run them from the repo root, inside WSL/Linux/macOS.
#   make help

COMPOSE := docker compose -f infra/local/docker-compose.yml
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip

.DEFAULT_GOAL := help
.PHONY: help venv env up down ps logs init smoke test check reset

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

down:  ## Stop the stack (data is kept)
	$(COMPOSE) down

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
	$(COMPOSE) down -v
