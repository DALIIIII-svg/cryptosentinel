"""CryptoSentinel dashboard (Streamlit, read-only). Run:  make dashboard  -> http://localhost:8501

Pages
  1. Market monitor   price of each coin with the alerts marked on the chart
  2. Alert feed       every explained alert: cause, confidence, summary, cited sources
  3. Volatility       forecast for the next hour vs the volatility that actually happened

Reads the DynamoDB hot tables (prices-hot, alerts, forecasts). During a replay the clock of the
dashboard is the replayed time (the newest candle), not the wall clock.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root (streamlit runs this file directly)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from boto3.dynamodb.conditions import Key

from common.aws_clients import dynamodb_resource
from common.config import get_settings

st.set_page_config(page_title="CryptoSentinel", layout="wide")
S = get_settings()
DB = dynamodb_resource()
CAUSE_COLOR = {"news_driven": "#1f77b4", "market_wide": "#7f7f7f", "possible_manipulation": "#d62728",
               "unexplained": "#ff7f0e"}
CAUSE_LABEL = {"news_driven": "News-driven", "market_wide": "Market-wide",
               "possible_manipulation": "Possible manipulation", "unexplained": "Unexplained"}


TEXT_COLUMNS = {"symbol", "cause", "risk_level", "summary", "alert_id", "citations", "model", "guardrails", "replay"}


def _num(df: pd.DataFrame) -> pd.DataFrame:
    """DynamoDB returns numbers as Decimal: convert every non-text column to float."""
    for c in df.columns:
        if c not in TEXT_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def query_symbol(table: str, symbol: str, since_ms: int) -> pd.DataFrame:
    t = DB.Table(S.table_name(table))
    items, kw = [], {"KeyConditionExpression": Key("symbol").eq(symbol) & Key("open_time").gte(since_ms)}
    while True:
        resp = t.query(**kw)
        items += resp.get("Items", [])
        if "LastEvaluatedKey" not in resp:
            break
        kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return _num(pd.DataFrame(items)) if items else pd.DataFrame()


def latest_time() -> int | None:
    t = DB.Table(S.table_name("prices-hot"))
    best = None
    for sym in S.symbol_list:
        r = t.query(KeyConditionExpression=Key("symbol").eq(sym), ScanIndexForward=False, Limit=1).get("Items", [])
        if r:
            best = max(best or 0, int(r[0]["open_time"]))
    return best


def all_alerts() -> pd.DataFrame:
    t = DB.Table(S.table_name("alerts"))
    items, kw = [], {}
    while True:
        resp = t.scan(**kw)
        items += resp.get("Items", [])
        if "LastEvaluatedKey" not in resp:
            break
        kw["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    for c in ("ts", "confidence", "ret_5m_pct", "others_ret_5m_pct", "anomaly_score", "vol_forecast_60m_pct"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("ts", ascending=False)


def fmt_time(ms) -> str:
    return pd.to_datetime(int(ms), unit="ms", utc=True).strftime("%Y-%m-%d %H:%M UTC")


st.title("CryptoSentinel")
now = latest_time()
if now is None:
    st.warning("No prices yet. Start the pipeline (`make demo-start`) and a replay (`make replay`).")
    st.stop()
st.caption(f"Data time: {fmt_time(now)}  ·  refresh the page (R) to update")

hours = st.sidebar.slider("Window (hours)", 1, 24, 4)
symbol = st.sidebar.selectbox("Coin", S.symbol_list, index=0)
since = now - hours * 3_600_000
alerts = all_alerts()

tab1, tab2, tab3 = st.tabs(["Market monitor", "Alert feed", "Volatility forecast"])

with tab1:
    p = query_symbol("prices-hot", symbol, since)
    if p.empty:
        st.info("No candles for this coin in the window.")
    else:
        p = p.sort_values("open_time")
        x = pd.to_datetime(p["open_time"], unit="ms", utc=True)
        fig = go.Figure(go.Candlestick(x=x, open=p["open"], high=p["high"], low=p["low"], close=p["close"], name=symbol))
        a = alerts[(alerts["symbol"] == symbol) & (alerts["ts"] >= since)] if not alerts.empty else pd.DataFrame()
        for cause, g in (a.groupby("cause") if not a.empty else []):
            near = pd.merge_asof(g.sort_values("ts")[["ts", "summary"]], p[["open_time", "close"]].rename(columns={"open_time": "ts"}),
                                 on="ts", direction="backward")
            fig.add_trace(go.Scatter(x=pd.to_datetime(near["ts"], unit="ms", utc=True), y=near["close"], mode="markers",
                                     marker=dict(size=14, symbol="triangle-down", color=CAUSE_COLOR.get(cause)),
                                     name=CAUSE_LABEL.get(cause, cause), hovertext=near["summary"]))
        fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
    cols = st.columns(len(S.symbol_list))
    for c, sym in zip(cols, S.symbol_list):
        q = query_symbol("prices-hot", sym, now - 3_600_000)
        if not q.empty:
            q = q.sort_values("open_time")
            chg = (q["close"].iloc[-1] / q["close"].iloc[0] - 1) * 100
            c.metric(sym.replace("USDT", ""), f"{q['close'].iloc[-1]:,.4g}", f"{chg:+.2f}% 1h")

with tab2:
    if alerts.empty:
        st.info("No explained alerts yet.")
    else:
        st.write(f"{len(alerts)} alerts")
        for _, r in alerts.head(50).iterrows():
            title = (f"{fmt_time(r['ts'])} · {r['symbol']} · {r.get('ret_5m_pct', 0) or 0:+.2f}% in 5 min · "
                     f"{CAUSE_LABEL.get(r['cause'], r['cause'])} ({float(r['confidence']):.0%})")
            with st.expander(title, expanded=False):
                st.write(r["summary"])
                st.caption(f"Other coins: {r.get('others_ret_5m_pct') or 0:+.2f}% · risk {r.get('risk_level')} · "
                           f"{int(r.get('news_considered') or 0)} news read · model {r.get('model')}")
                cits = r.get("citations")
                if isinstance(cits, list) and cits:
                    st.markdown("**Sources (published before the move)**")
                    for c in cits:
                        st.markdown(f"- [{c['title']}]({c['url']}) · {c.get('source', '')} · {fmt_time(c['published_at'])}")

with tab3:
    f = query_symbol("forecasts", symbol, since)
    p = query_symbol("prices-hot", symbol, since - 3_600_000)
    if f.empty or p.empty:
        st.info("No forecasts yet (the live scorer writes them once features are complete).")
    else:
        p = p.sort_values("open_time")
        r = np.log(p["close"] / p["close"].shift(1))
        realised = np.sqrt((r ** 2).rolling(60).sum()) * 100          # past 60 minutes, in %
        p["realised_next_60m"] = realised.shift(-60)                   # what happened in the hour after t
        m = f.sort_values("open_time").merge(p[["open_time", "realised_next_60m"]], on="open_time", how="left")
        x = pd.to_datetime(m["open_time"], unit="ms", utc=True)
        fig = go.Figure([go.Scatter(x=x, y=m["vol_forecast_pct"], name="forecast (next 60 min)"),
                         go.Scatter(x=x, y=m["realised_next_60m"], name="realised (next 60 min)")])
        fig.update_layout(height=420, yaxis_title="volatility over 60 min (%)", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Realised volatility appears 60 minutes later than the forecast, once the hour has happened.")
