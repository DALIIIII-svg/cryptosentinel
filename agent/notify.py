"""Notifications: console (local) or Amazon SNS email (AWS). NOTIFIER=console|sns."""
from __future__ import annotations

import boto3
import pandas as pd

from common.config import get_settings
from common.log import get_logger

log = get_logger("notify")

LABELS = {"news_driven": "news-driven", "market_wide": "market-wide",
          "possible_manipulation": "possible manipulation", "unexplained": "unexplained"}


def format_alert(rec: dict) -> tuple[str, str]:
    when = pd.to_datetime(rec["ts"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M UTC")
    move = rec.get("ret_5m_pct")
    subject = (f"[CryptoSentinel] {rec['symbol'].replace('USDT', '')} "
               f"{move:+.1f}% in 5 min: {LABELS.get(rec['cause'], rec['cause'])}" if move is not None
               else f"[CryptoSentinel] {rec['symbol']}: {LABELS.get(rec['cause'], rec['cause'])}")
    lines = [f"{rec['symbol']} at {when}{' (replay)' if rec.get('replay') else ''}",
             f"Move: {move:+.2f}% in 5 min; other coins: {rec.get('others_ret_5m_pct') or 0:+.2f}%" if move is not None else "",
             f"Cause: {LABELS.get(rec['cause'], rec['cause'])} (confidence {rec['confidence']:.0%}, risk {rec['risk_level']})",
             "", rec["summary"], ""]
    if rec.get("citations"):
        lines.append("Sources (published before the move):")
        for c in rec["citations"]:
            t = pd.to_datetime(c["published_at"], unit="ms", utc=True).strftime("%H:%M UTC")
            lines.append(f"  - {c['title']} ({c['source']}, {t}) {c['url']}")
    lines.append("\nAutomated analysis, not investment advice.")
    return subject[:100], "\n".join(x for x in lines if x is not None)


def send(rec: dict) -> None:
    s = get_settings()
    subject, body = format_alert(rec)
    if s.notifier == "sns" and s.sns_topic_arn:
        boto3.client("sns", region_name=s.aws_region).publish(TopicArn=s.sns_topic_arn, Subject=subject, Message=body)
        log.info("email sent: %s", subject)
    else:
        print("\n" + "=" * 80 + f"\n{subject}\n" + "-" * 80 + f"\n{body}\n" + "=" * 80, flush=True)
