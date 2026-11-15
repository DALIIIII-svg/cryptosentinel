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

Easiest: `bash scripts/setup_wsl.sh` checks Docker, installs what is missing and runs everything below.

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

## Data collection (Phase 2)

```bash
make collect        # build and start the 4 collectors in Docker (restart automatically)
make collect-logs   # watch them work (Ctrl+C quits the view, not the collectors)
make report         # files in S3, latest candle per coin, news per coin
make collect-stop   # stop collecting
```

| Collector | Reads | Writes |
| --- | --- | --- |
| `price-producer` | Binance WebSocket `data-stream.binance.vision` (closed 1-min candles, 10 coins) + REST gap fill | Kafka `prices.klines` |
| `news-producer` | RSS: CoinDesk, Cointelegraph, Decrypt, The Block (+ CryptoPanic if `CRYPTOPANIC_TOKEN` is set), every 2 min | Kafka `news.raw` |
| `s3-sink` | `prices.klines`, `news.raw` | `raw/{topic}/dt=.../hour=.../*.jsonl.gz` every 60 s |
| `hot-writer` | `prices.klines`, `news.raw` | DynamoDB `prices-hot` (48 h) and `news` (30 days) |

Live mode is optional: run it while you work or during the demo. Training, evaluation and the
demo itself use the historical period below. After a sleep or restart, `make collect` again; the price
producer re-fetches missed candles (last 120 minutes on a restart, everything since the last candle on
a reconnect).

## Historical data, Parquet and replay (Phase 3)

The project works on a fixed, volatile period: **February 2024 to January 2025** (March 2024 all-time
high, 5 August 2024 crash, November 2024 election rally). Change it with `HISTORY_START` / `HISTORY_END`.

| Source | What | Where it lands |
| --- | --- | --- |
| Binance public data (`data.binance.vision`) | 1-minute candles, 10 coins, 12 monthly files each, checksum verified | `raw/history/klines/` |
| [CoinDesk crypto news 2020-2025](https://huggingface.co/datasets/maryamfakhari/crypto-news-coindesk-2020-2025) (Hugging Face, CC BY-NC 4.0, academic use) | ~229k articles from several outlets, timestamps to the second | `raw/history/news/coindesk_hf/` |
| Optional: [CryptoVision](https://data.mendeley.com/datasets/wvjjxr8bxx) (Mendeley, CC BY 4.0) | 188k articles 2017-2025; download by hand, then `python -m processing.backfill_history --only news --extra-news cryptovision=path.csv` | `raw/history/news/cryptovision/` |

Only the headline, a 300-character summary and the link of each article reach the processed zone.

```bash
make history        # download everything into S3 raw (resumable; ~25 MB of candles + 240 MB news file)
make parquet        # raw -> processed Parquet, deduplicated, + data quality report
make shocks         # shock EVENTS (market-wide / coin-specific) + news before -> eval/shock_events.csv
make newstime       # check that news timestamps are UTC (must say "look like UTC")
make replay DAY=2024-08-05 FROM=00:00 TO=04:00 SPEED=10   # replay history into Kafka
```

Replay first sends the 25 hours before the window instantly (marked `"warmup": true`) so live
features are complete from the first replayed minute. Replay publishes to the same Kafka topics as the live producers, marked `"replay": true`
(the S3 sink ignores those, everything downstream treats them as live). `--rebase` shifts the
timestamps so the replay looks like it is happening now.

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
| 2 | Live ingestion (prices, news, S3 sink, DynamoDB) | done — optional live mode: `make collect` |
| 3 | History (Feb 2024 - Jan 2025), Parquet, shocks, replay | done — `make history parquet shocks` |
| 4 | Features (shared live/offline module), training table | done — `make features` |
| 5 | Models and baselines | next |
| 4 | Features | |
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

## Features (Phase 4)

`processing/features.py` is the single feature module: the training table and the live consumer
both call it, and `tests/test_features.py` proves live and offline values are identical and that
no feature looks into the future.

| Feature | Meaning |
| --- | --- |
| `ret_1m`, `ret_5m`, `ret_15m`, `ret_60m` | log returns over the last 1 / 5 / 15 / 60 minutes |
| `rv_5`, `rv_15`, `rv_60`, `rv_1d` | realised volatility over 5 min ... 1 day |
| `vol_z`, `trades_z` | volume / trade count compared with the last 24 h (z-score) |
| `imbalance` | buy vs sell pressure, -1 .. +1 |
| `range_pct` | candle high-low range |
| `mkt_ret_1m`, `mkt_ret_5m`, `mkt_rv_15`, `idio_ret_5m` | market move and the coin-specific part |
| `hour_sin`, `hour_cos`, `filled` | time of day; 1 if the candle was missing |
| `y_log_rv_next_60` (target) | log realised volatility of the NEXT 60 minutes (training only) |

```bash
make features        # processed/features/month=YYYY-MM/data.parquet (about 4.4M rows)
make features-live   # live: prices.klines -> features.1m; start it before `make replay`
```
