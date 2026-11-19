# CryptoSentinel

Explaining crypto price shocks as they happen. CryptoSentinel streams 1-minute prices for 10 coins
and crypto news, detects abnormal price moves, forecasts next-hour volatility, and uses an
LLM agent to explain each shock with the news published before it. Each explanation is saved,
shown on a dashboard and sent as a notification.

SUP'COM — INDP3 AIM module (Data Engineering, ML & AI Agents). Team: Mohamed Ali El Filali, Bader Guesmi.

```
Binance prices ─┐                         ┌─> S3 raw zone ─> Parquet ─> training (XGBoost, anomaly detector)
                ├─> Kafka ─> features ─> anomaly scorer ─> alert ─> LLM agent ─> DynamoDB + dashboard + email
News (RSS) ─────┘                         └─> DynamoDB (hot data: last 48 h of prices, recent news)
```

## Contents

1. [How we work](#how-we-work)
2. [First-time setup (new teammate)](#first-time-setup-new-teammate)
3. [Build the data and the models (once)](#build-the-data-and-the-models-once)
4. [Run the demo (every time)](#run-the-demo-every-time)
5. [How each part works](#how-each-part-works)
6. [Results so far](#results-so-far)
7. [Repository layout](#repository-layout)
8. [Status](#status)
9. [Troubleshooting](#troubleshooting)

## How we work

- **Local first.** Everything runs on a laptop with Docker. AWS (Learner Lab, **$50 per student**) is
  used for the managed services and the final demo. The code is the same in both places: only the env
  file changes (`.env.local` vs `.env.aws`).
- **Zero cost** apart from the AWS credit: free tools, free data, free API tiers.
- **On AWS, nothing runs when nobody is using it**: stop EC2 and delete SageMaker endpoints at the end
  of every session.
- **Git:** one branch per feature, a pull request reviewed by the other person, `main` always runs.
  Secrets (API keys) go only in `.env.local`, which Git ignores. Never commit them.

## First-time setup (new teammate)

These steps are for Windows. On Linux or macOS, skip step 1 and run the commands in any terminal.

**1. Install the tools (once per PC)**
- **WSL with Ubuntu.** In PowerShell as administrator: `wsl --install -d Ubuntu`, then restart.
- **Docker Desktop.** In Settings → Resources → WSL integration, turn on **Ubuntu**.
- **Git for Windows**, plus `git config --global user.name "..."` and `user.email "..."`.

**2. Clone the repository (PowerShell)**
```powershell
cd $HOME\Documents
git clone https://github.com/DALIIIII-svg/cryptosentinel.git
```

**3. Set up the project (Ubuntu terminal)**

From here on, every command goes in the **Ubuntu** terminal, not PowerShell. Replace `<you>` with your
Windows user name.
```bash
cd /mnt/c/Users/<you>/Documents/cryptosentinel
bash scripts/setup_wsl.sh     # checks Docker, installs make/python, creates .venv, starts the stack, runs the smoke test
```
It must end with `Smoke test passed.`

**4. Add your free Groq key (the LLM used by the agent)**

Create a key at https://console.groq.com/keys, then open `.env.local` and set `GROQ_API_KEY=...`.
Check it with:
```bash
make llm-check
```
`.env.local` is created from `.env.example` and already contains `NEWS_TIME_SHIFT_MINUTES=-210`
(see [the news clock fix](#historical-data-and-replay)). Don't change it.

## Build the data and the models (once)

Data and models are **not** in Git: they live in the local Docker storage and in `models/`. Build them
once per PC. They stay there until you run `make reset`.

```bash
make up              # if the stack is not already running
make init
make history         # download 12 months of prices + the news dataset (~265 MB, resumable)
make parquet         # raw files -> clean Parquet tables + data quality report
make shocks          # find the 40 biggest shock events -> eval/shock_events.csv
make newstime        # must end with "No further correction needed"
make features        # 19 features for every coin and minute (about 4.4M rows)
make train           # volatility model + anomaly detector (a few minutes)
make test            # unit tests, all must pass
```

## Run the demo (every time)

First start **Docker Desktop** and wait for "Engine running".

**Ubuntu window 1**
```bash
cd /mnt/c/Users/<you>/Documents/cryptosentinel
git pull
make up                # Kafka, S3 (SeaweedFS), DynamoDB Local
make init              # safe to repeat
make demo-start        # hot writer, live features, live scorer, agent (in the background, logs in logs/)
make demo-status       # all 4 must say "running"
```

**Ubuntu window 2** (the dashboard keeps this window busy)
```bash
cd /mnt/c/Users/<you>/Documents/cryptosentinel
make dashboard         # then open http://localhost:8501
```

**Back in window 1**, after about 10 seconds:
```bash
make replay DAY=2024-08-05 FROM=00:00 TO=03:00 SPEED=20   # the 5 Aug 2024 crash, 20x faster
```
Alerts and the agent's explanations appear on the dashboard and in `logs/agent.log`.

**When you're done**
```bash
make demo-stop         # stop the 4 services
make down              # stop the containers (data is kept)
```

| Local service | Address |
| --- | --- |
| Dashboard | http://localhost:8501 |
| Kafbat UI (browse Kafka topics) | http://localhost:8080 |
| Kafka (from your laptop / from containers) | `localhost:9092` / `kafka:29092` |
| SeaweedFS S3 API / file browser | http://localhost:8333 / http://localhost:8888 |
| DynamoDB Local | http://localhost:8000 |

`make help` lists every command.

## How each part works

### Live collection (optional)

The demo uses replayed history, so live collection is optional. It's useful to show that the live path works.

```bash
make collect        # start the 4 collectors in Docker (they restart automatically)
make collect-logs   # watch them (Ctrl+C closes the view, not the collectors)
make report         # files in S3, latest candle per coin, news per coin
make collect-stop
```

| Collector | Reads | Writes |
| --- | --- | --- |
| `price-producer` | Binance WebSocket (closed 1-min candles, 10 coins) + REST gap fill | Kafka `prices.klines` |
| `news-producer` | RSS: CoinDesk, Cointelegraph, Decrypt, The Block (+ CryptoPanic if `CRYPTOPANIC_TOKEN` is set) | Kafka `news.raw` |
| `s3-sink` | `prices.klines`, `news.raw` | `raw/{topic}/dt=.../hour=.../*.jsonl.gz` every 60 s |
| `hot-writer` | `prices.klines`, `news.raw` | DynamoDB `prices-hot` (48 h) and `news` (30 days) |

### Historical data and replay

We work on a fixed, volatile year: **February 2024 to January 2025**. It includes the March 2024 all-time
high, the April halving, the 5 August crash, the November election rally, Bitcoin above $100K and the
January 2025 inauguration. That's one real Binance candle per minute for each of the 10 coins, about
5.3 million candles in total.

| Source | What |
| --- | --- |
| Binance public data (`data.binance.vision`) | 1-minute candles, 10 coins, 12 monthly files each, checksum verified |
| [CoinDesk crypto news 2020-2025](https://huggingface.co/datasets/maryamfakhari/crypto-news-coindesk-2020-2025) (Hugging Face, CC BY-NC 4.0, academic use) | ~229k articles from several outlets |

Only the headline, a 300-character summary and the link of each article reach the processed tables.

**The news clock fix.** The news dataset's timestamps are 3 h 30 later than the real publication time.
We found this by matching headlines like "Bitcoin falls below $53K" with the minute the price actually
crossed $53K. `NEWS_TIME_SHIFT_MINUTES=-210` corrects it when the Parquet tables are built. Without it,
the agent would miss the real news and cite unrelated articles. If you ever change this value, run
`make parquet-news`.

**Replay** (`make replay`) sends the stored history into the same Kafka topics as the live producers,
marked `"replay": true`, at N× speed. It first sends the 25 hours before the window instantly, so the
live features are complete from the first replayed minute.

### Features

`processing/features.py` is the **single** feature module: both the training table and the live
pipeline call it, so a model never sees features computed differently live than in training.
`tests/test_features.py` proves live and offline values are identical and that no feature looks
into the future.

| Feature | Meaning |
| --- | --- |
| `ret_1m`, `ret_5m`, `ret_15m`, `ret_60m` | price change over the last 1 / 5 / 15 / 60 minutes |
| `rv_5`, `rv_15`, `rv_60`, `rv_1d` | realised volatility over 5 min ... 1 day |
| `vol_z`, `trades_z` | volume / number of trades compared with the last 24 h |
| `imbalance` | buy vs sell pressure, -1 .. +1 |
| `range_pct` | candle high-low range |
| `mkt_ret_1m`, `mkt_ret_5m`, `mkt_rv_15`, `idio_ret_5m` | the whole market's move, and the coin's own part |
| `hour_sin`, `hour_cos`, `filled` | time of day; 1 if the candle was missing |
| `y_log_rv_next_60` (target) | volatility of the NEXT 60 minutes (training only) |

### Models

Fixed time split, never shuffled: **train** Feb-Oct 2024, **validation** Nov 2024, **test** Dec 2024 - Jan 2025.

- **Volatility forecaster** (`make train-vol`) predicts the next hour's volatility. XGBoost on the 19 features
  is compared with naive persistence, HAR-RV and GARCH(1,1). Outputs: `eval/results_vol.csv`, `models/vol_xgb.json`.
- **Anomaly detector** (`make train-anomaly`) compares a simple rule (5-minute move in standard deviations),
  an Isolation Forest, and their combinations. Each detector is allowed **10 alerts per day across all coins**.
  The live detector is **"rule_first"**: the rule takes 8 of the 10 daily alerts, and the forest adds its 2 rarest
  cases. On AWS the Isolation Forest becomes SageMaker Random Cut Forest. Output: `eval/results_anomaly.csv`.

### Agent

The detector raises the alert; the agent explains it. `agent/graph.py` (LangGraph):
`gather_context -> search_news -> classify (LLM) -> validate -> finalize`.

1. Code gathers the facts: the price path, how much the other coins moved, and the news from the hour
   **before** the move.
2. The LLM picks one cause (`news_driven`, `market_wide`, `possible_manipulation`, `unexplained`) and writes a
   confidence, a risk level, a 2-3 sentence summary and the news it relies on.
3. Code enforces the guardrails. Cited news must have been shown to the LLM and published before the move, and
   `news_driven` needs a valid citation. An invalid answer gets one retry, then a safe fallback. No trading advice.

When several coins move together, the agent explains the move once for the whole market (one LLM call).

LLM: Groq free tier, `openai/gpt-oss-120b` by default. Set it with `LLM_PROVIDER` / `LLM_MODEL` in `.env.local`:
`ollama` for a local model, `fake` for offline tests, `bedrock` on AWS if available.

### Agent evaluation

```bash
make eval-agent         # run the agent on the 40 shock events -> eval/agent_outputs.csv
make eval-agent-score   # accuracy vs eval/labelled_events.csv, "always market_wide" baseline, faithfulness, latency
```
`eval/labelled_events.csv` holds the true cause of each event. Labels were proposed from the headlines
and the price move only, without looking at the agent's answers, then checked and corrected by both of us.
The `notes` column gives the reason for each label. Note: `make eval-agent` rewrites the labels file only
if it doesn't exist yet.

## Results so far

**Volatility, test months** (lower RMSE / QLIKE and higher R² are better):

| Model | RMSE | QLIKE | R² |
| --- | --- | --- | --- |
| Naive (last hour) | 0.306 | 0.263 | 0.734 |
| GARCH(1,1) | 0.307 | 0.268 | 0.733 |
| HAR-RV | 0.279 | 0.229 | 0.780 |
| **XGBoost** | **0.265** | **0.207** | **0.802** |

**Anomaly detection** at 10 alerts per day: "rule_first" finds 96% of planted price jumps (the forest alone
finds 48%). It finds 100% of the real market-wide shocks and 91% of the coin-specific ones.

**Agent:** accuracy against the labels is pending (see [Agent evaluation](#agent-evaluation)).

## Repository layout

```
common/       config, Kafka / S3 / DynamoDB helpers, topic and table definitions
infra/local/  docker-compose stack, init and smoke-test scripts
infra/aws/    AWS scripts (day-1 checks now; deploy and teardown later)
producers/    Binance price producer, news producer
processing/   S3 sink, hot writer, history backfill, Parquet, shocks, features, live scorer
ml/           training, baselines, anomaly detector, scorers
agent/        LangGraph agent, LLM client, news sources, notifications
dashboard/    Streamlit app
eval/         agent evaluation, shock events, labels
replay/       re-stream stored history into Kafka for the demo
scripts/      setup_wsl.sh, demo.sh
tests/        unit tests (make test)
```

## Status

| Phase | Content | Status |
| --- | --- | --- |
| 0 | Setup, day-1 AWS checks | scripts ready in `infra/aws/day1/`, to run |
| 1 | Local infrastructure | done |
| 2 | Live ingestion (prices, news, S3 sink, DynamoDB) | done |
| 3 | History (Feb 2024 - Jan 2025), Parquet, shocks, replay | done |
| 4 | Features (shared live/offline module), training table | done |
| 5 | Models vs baselines, live scorer | done |
| 6 | Agent, evaluation, dashboard | done (label check pending) |
| 7 | Migration to AWS (account A) | next |
| 8 | Rebuild in demo account B | |
| 9 | Report, video, presentation | |

### AWS environment facts (fill in after the day-1 checks)

| Question | Answer | Date |
| --- | --- | --- |
| Binance `data-stream.binance.vision` reachable from us-east-1? | | |
| `api.binance.com` blocked (451) from us-east-1? | | |
| Bedrock usable? Which models? | | |
| SageMaker training with built-in XGBoost works? | | |
| SageMaker serverless endpoint allowed? | | |
| Teacher OK with Kafka on EC2 instead of MSK? | | |
| Teacher OK with Groq if Bedrock is blocked? | | |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `Cannot connect to the Docker daemon` | Start Docker Desktop and check that WSL integration is on for Ubuntu |
| `make: *** No rule to make target` | You're not in the project folder: `cd /mnt/c/Users/<you>/Documents/cryptosentinel` |
| Commands typed in PowerShell fail | Use the **Ubuntu** terminal for everything except `git clone` |
| A service shows "stopped" in `make demo-status` | Read its log, e.g. `tail -n 30 logs/agent.log` |
| Groq error 429 (rate limit) | Free-tier limit: wait a minute; `make eval-agent` already pauses between calls |
| Dashboard empty | Run `make replay ...` after `make demo-start`, and wait for the 25-hour warm-up to be sent |
| Want to start over | `make reset` deletes **all** local data, then redo [Build the data](#build-the-data-and-the-models-once) |
