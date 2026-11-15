"""Evaluate the agent on the historical shock events (no Kafka needed).

Step 1  python -m eval.eval_agent --run
        Runs the agent on every event of eval/shock_events.csv using the processed history,
        writes eval/agent_outputs.csv and creates eval/labelled_events.csv (if missing) for labelling:
        one row per event with the 5 most recent headlines before the move, and an EMPTY
        `true_cause` column. Fill it yourselves (news_driven / market_wide / possible_manipulation /
        unexplained) by reading the headlines and the price move, WITHOUT looking at the agent's answer.

Step 2  python -m eval.eval_agent --score
        Compares the agent with your labels: accuracy, per-cause recall, and the baseline
        "always market_wide". Also reports faithfulness (share of citations that exist and were
        published before the move; must be 100 %) and latency. Writes eval/results_agent.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from agent.graph import NEWS_LOOKBACK_MS, RiskAnalyst
from agent.schemas import CAUSES
from agent.sources import ParquetSource
from common.log import get_logger
from common.parquet_io import read_df
from common.schemas import to_epoch_ms
from processing.live_scorer import build_alert, coins_moved

log = get_logger("eval_agent")
EVENTS = Path("eval/shock_events.csv")
OUTPUTS = Path("eval/agent_outputs.csv")
LABELS = Path("eval/labelled_events.csv")


def events_frame() -> pd.DataFrame:
    if not EVENTS.exists():
        raise SystemExit("eval/shock_events.csv missing: run `make shocks` first")
    ev = pd.read_csv(EVENTS)
    ev["ts"] = to_epoch_ms(ev["time_utc"])
    return ev


def feature_rows(ev: pd.DataFrame) -> dict[int, list[dict]]:
    """Feature rows of every coin at the minute that ends each event (same input as live scoring)."""
    wanted = set((ev["ts"] - 60_000).astype("int64"))
    out: dict[int, list[dict]] = {}
    months = sorted(set(pd.to_datetime(list(wanted), unit="ms", utc=True).strftime("%Y-%m")))
    for m in months:
        f = read_df(f"processed/features/month={m}/data.parquet")
        if f is None:
            continue
        f = f[f["open_time"].isin(wanted)]
        for rec in f.to_dict("records"):
            out.setdefault(int(rec["open_time"]), []).append(rec)
    return out


def make_alert(event: pd.Series, rows: list[dict], scorer) -> dict:
    """The alert the live scorer would have sent for this event (falls back to a minimal alert)."""
    lead = next((r for r in rows if r["symbol"] == event["leader"]), None)
    if lead is None:
        return {"alert_id": event["event_id"], "symbol": event["leader"], "ts": int(event["ts"])}
    if scorer is not None:
        sc = scorer.score([lead]).iloc[0].to_dict()
    else:
        sc = {"anomaly_score": 0.0, "anomaly_threshold": 1.0, "z_5m": 0.0, "vol_forecast_pct": 0.0}
    alert = build_alert(lead, sc, coins_moved(rows).get(int(lead["open_time"])))
    alert["alert_id"] = event["event_id"]
    return alert


def run(limit: int | None, pause: float) -> int:
    ev = events_frame()
    if limit:
        ev = ev.head(limit)
    src = ParquetSource()
    analyst = RiskAnalyst(src)
    try:
        from ml.scorer import LocalScorer
        scorer = LocalScorer()
    except Exception as exc:  # noqa: BLE001
        log.warning("no trained models (%s): alerts without anomaly scores", exc)
        scorer = None
    feats = feature_rows(ev)
    rows, label_rows = [], []
    for n_done, (i, e) in enumerate(ev.iterrows()):
        if n_done and pause:
            time.sleep(pause)   # stay under the free LLM tier's per-minute token limit
        alert = make_alert(e, feats.get(int(e["ts"]) - 60_000, []), scorer)
        rec = analyst.explain(alert)
        rows.append({"event_id": e["event_id"], "type": e["type"], "symbol": e["leader"], "time_utc": e["time_utc"],
                     "cause": rec["cause"], "confidence": rec["confidence"], "risk_level": rec["risk_level"],
                     "summary": rec["summary"], "citations": json.dumps(rec["citations"]),
                     "news_considered": rec["news_considered"], "guardrails": "|".join(rec["guardrails"]),
                     "latency_ms": rec["latency_ms"]})
        news = src.news(e["leader"], int(e["ts"]) - NEWS_LOOKBACK_MS, int(e["ts"]))[-5:]
        label_rows.append({"event_id": e["event_id"], "time_utc": e["time_utc"], "symbol": e["leader"],
                           "type": e["type"], "leader_ret_5m_pct": e["leader_ret_5m_pct"],
                           "others_ret_5m_pct": e.get("others_ret_5m_pct"),
                           "headlines_before": " || ".join(
                               f"[{int((int(e['ts']) - n['published_at']) / 60000)} min before] {n['title']}" for n in news),
                           "true_cause": "", "notes": ""})
        log.info("%s %s: %s (%.0f%%) in %d ms", e["event_id"], e["leader"], rec["cause"],
                 rec["confidence"] * 100, rec["latency_ms"])
    pd.DataFrame(rows).to_csv(OUTPUTS, index=False)
    if not LABELS.exists():
        pd.DataFrame(label_rows).to_csv(LABELS, index=False)
        log.info("created %s: fill the true_cause column, then run --score", LABELS)
    log.info("wrote %s (%d events)", OUTPUTS, len(rows))
    return 0


def faithfulness(out: pd.DataFrame, ev: pd.DataFrame) -> float:
    ts = dict(zip(ev["event_id"], ev["ts"]))
    total = ok = 0
    for _, r in out.iterrows():
        for c in json.loads(r["citations"]):
            total += 1
            ok += int(c["published_at"] <= ts[r["event_id"]] and bool(c.get("url")))
    return ok / total if total else 1.0


def score() -> int:
    if not OUTPUTS.exists() or not LABELS.exists():
        raise SystemExit("run `python -m eval.eval_agent --run` first, then fill true_cause in eval/labelled_events.csv")
    out, lab, ev = pd.read_csv(OUTPUTS), pd.read_csv(LABELS), events_frame()
    lab = lab[lab["true_cause"].fillna("").str.strip().isin(CAUSES)]
    if lab.empty:
        raise SystemExit(f"no labels yet: fill true_cause with one of {CAUSES}")
    m = lab.merge(out, on="event_id", how="inner")
    acc = float((m["cause"] == m["true_cause"].str.strip()).mean())
    base = float((m["true_cause"].str.strip() == "market_wide").mean())
    rows = [{"metric": "labelled events", "value": len(m)},
            {"metric": "agent accuracy", "value": round(acc, 3)},
            {"metric": "baseline 'always market_wide' accuracy", "value": round(base, 3)},
            {"metric": "faithfulness (citations valid and before the move)", "value": round(faithfulness(out, ev), 3)},
            {"metric": "median latency (s)", "value": round(float(out["latency_ms"].median()) / 1000, 2)}]
    for c in CAUSES:
        sub = m[m["true_cause"].str.strip() == c]
        if len(sub):
            rows.append({"metric": f"recall {c} (n={len(sub)})", "value": round(float((sub["cause"] == c).mean()), 3)})
    res = pd.DataFrame(rows)
    res.to_csv("eval/results_agent.csv", index=False)
    print(res.to_string(index=False))
    print("\nConfusion (rows = your label, columns = agent):")
    print(pd.crosstab(m["true_cause"].str.strip(), m["cause"]).to_string())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", action="store_true")
    g.add_argument("--score", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N events (quick check)")
    ap.add_argument("--pause", type=float, default=6.0, help="seconds between LLM calls (free tier limits)")
    args = ap.parse_args()
    return run(args.limit, args.pause) if args.run else score()


if __name__ == "__main__":
    sys.exit(main())
