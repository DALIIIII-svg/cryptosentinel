# CryptoSentinel

Explaining crypto price shocks as they happen. CryptoSentinel streams live prices for 10 coins
and crypto news, detects abnormal price moves, forecasts next-hour volatility, and uses an
LLM agent to explain each shock with the news that preceded it.

SUP'COM — INDP3 AIM module (Data Engineering, ML & AI Agents). Team: Mohamed Ali El Filali, Bader Guesmi.

## How we work

- **Local first.** Everything runs on a laptop with Docker; AWS (Learner Lab, $100 each) is used
  for the managed services and the demo. The code is the same in both places: only the env file
  changes (`.env.local` vs `.env.aws`).
- **Zero cost** apart from AWS credit: free tools, free data, free API tiers.
- One branch per feature, a pull request reviewed by the other person, `main` always runs.

## Quick start (WSL Ubuntu on Windows, or Linux/macOS)

Prerequisites: Docker Engine (or Docker Desktop) with `docker compose`, Python 3.10+, `make`.

```bash
make venv        # create .venv and install dependencies
make env         # create .env.local from .env.example (then add your free Groq key)
make up          # start Kafka, Kafbat UI, SeaweedFS, DynamoDB Local
make init        # create topics, bucket and tables
make smoke       # Phase 1 check: must print "Smoke test passed."
make test        # unit tests
```

| Service | Address |
| --- | --- |
| Kafka (from your laptop) | `localhost:9092` |
| Kafka (from other containers) | `kafka:29092` |
| Kafbat UI (browse topics) | http://localhost:8080 |
| SeaweedFS S3 API | http://localhost:8333 |
| SeaweedFS file browser | http://localhost:8888 |
| DynamoDB Local | http://localhost:8000 |

`make down` stops the stack and keeps the data; `make reset` deletes all local data.

## Repository layout

```
common/       config, Kafka / S3 / DynamoDB helpers, topic and table definitions
infra/local/  docker-compose stack, init and smoke-test scripts
infra/aws/    AWS scripts (day-1 checks now; deploy and teardown later)
producers/    Binance price producer, news producer            (Phase 2)
processing/   S3 sink, features, Glue job, history backfill     (Phases 2-4)
ml/           training, scorers, endpoint deployment            (Phase 5)
agent/        LangGraph agent, tools, prompts                   (Phase 6)
dashboard/    Streamlit app                                     (Phase 6)
eval/         labelled events, model and agent evaluation       (Phases 5-6)
replay/       re-stream a stored day into Kafka for demos       (Phase 6)
tests/        unit tests
```

## Status

| Phase | Content | Status |
| --- | --- | --- |
| 0 | Setup, day-1 AWS checks | scripts ready in `infra/aws/day1/` — to run |
| 1 | Local infrastructure | done — run `make check` |
| 2 | Ingestion (prices, news, S3 sink) | next |
| 3 | History backfill + batch processing | |
| 4 | Features | |
| 5 | Models and baselines | |
| 6 | Agent, replay, dashboard | |
| 7 | Migration to AWS (account A) | |
| 8 | Rebuild in demo account B | |
| 9 | Report, video, presentation | |

## Environment facts (fill in after the day-1 checks)

| Question | Answer | Date |
| --- | --- | --- |
| Binance `data-stream.binance.vision` reachable from us-east-1? | | |
| `api.binance.com` blocked (451) from us-east-1? | | |
| Bedrock usable? Which models? | | |
| SageMaker training with built-in XGBoost works? | | |
| SageMaker serverless endpoint allowed? | | |
| Teacher OK with Kafka on EC2 instead of MSK? | | |
| Teacher OK with Groq if Bedrock is blocked? | | |
| CryptoPanic free plan still available? | | |
