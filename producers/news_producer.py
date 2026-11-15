"""News producer: RSS feeds (+ CryptoPanic if a free token is set) -> Kafka `news.raw`.

Every NEWS_POLL_SECONDS (default 120 s) it reads every feed, normalises each item
(headline, short summary, link, UTC time, tagged coins), drops duplicates by URL hash,
and publishes new items keyed by news_id. Only headline + short summary + link are kept.

Run:  python -m producers.news_producer
"""
from __future__ import annotations

import signal
import threading
from collections import OrderedDict

import feedparser
import requests

from common.config import get_settings
from common.kafka_io import make_producer, produce_json
from common.log import get_logger
from common.schemas import news_item, struct_time_to_ms
from common.topics import NEWS_RAW

log = get_logger("news_producer")

USER_AGENT = "CryptoSentinel/0.1 (student project; RSS reader)"
SEEN_MAX = 20_000


class SeenSet:
    """Bounded memory of news ids already published."""

    def __init__(self, maxlen: int = SEEN_MAX) -> None:
        self._d: OrderedDict[str, None] = OrderedDict()
        self.maxlen = maxlen

    def add(self, key: str) -> bool:
        """Return True if new."""
        if key in self._d:
            return False
        self._d[key] = None
        if len(self._d) > self.maxlen:
            self._d.popitem(last=False)
        return True


def parse_feed(name: str, content: bytes) -> list[dict]:
    parsed = feedparser.parse(content)
    items = []
    for e in parsed.entries:
        url = e.get("link")
        title = e.get("title")
        if not url or not title:
            continue
        published = struct_time_to_ms(e.get("published_parsed") or e.get("updated_parsed"))
        items.append(news_item(source=name, title=title, url=url, published_at=published,
                               summary=e.get("summary") or e.get("description")))
    return items


def parse_cryptopanic(payload: dict) -> list[dict]:
    """Defensive: field names differ between API versions/plans."""
    items = []
    for p in payload.get("results", []):
        url = p.get("original_url") or p.get("url")
        if not url and p.get("slug") and p.get("id"):
            url = f"https://cryptopanic.com/news/{p['id']}/{p['slug']}"
        title = p.get("title")
        if not url or not title:
            continue
        coins = [f"{c['code']}USDT" for c in (p.get("currencies") or []) if c.get("code")]
        published = None
        if p.get("published_at"):
            from datetime import datetime
            try:
                published = int(datetime.fromisoformat(p["published_at"].replace("Z", "+00:00")).timestamp() * 1000)
            except ValueError:
                published = None
        extra = {"votes": p["votes"]} if isinstance(p.get("votes"), dict) else None
        items.append(news_item(source="cryptopanic", title=title, url=url, published_at=published,
                               summary=p.get("description"), coins=coins, extra=extra))
    return items


class NewsProducer:
    def __init__(self) -> None:
        self.s = get_settings()
        self.producer = make_producer()
        self.seen = SeenSet()
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.stop = threading.Event()
        self.cryptopanic_ok = bool(self.s.cryptopanic_token)

    def fetch_rss(self) -> list[dict]:
        out = []
        for name, url in self.s.feed_list:
            try:
                resp = self.session.get(url, timeout=20)
                resp.raise_for_status()
                items = parse_feed(name, resp.content)
                out.extend(items)
            except Exception as exc:  # noqa: BLE001 - one bad feed must not stop the others
                log.warning("feed %s failed: %s", name, exc)
        return out

    def fetch_cryptopanic(self) -> list[dict]:
        if not self.cryptopanic_ok:
            return []
        try:
            resp = self.session.get(
                self.s.cryptopanic_url,
                params={"auth_token": self.s.cryptopanic_token, "public": "true", "kind": "news"},
                timeout=20,
            )
            if resp.status_code in (401, 403, 404):
                log.warning("CryptoPanic returned %s: disabling it (check token / URL)", resp.status_code)
                self.cryptopanic_ok = False
                return []
            resp.raise_for_status()
            return parse_cryptopanic(resp.json())
        except Exception as exc:  # noqa: BLE001
            log.warning("CryptoPanic failed: %s", exc)
            return []

    def poll_once(self) -> int:
        new = 0
        for item in self.fetch_rss() + self.fetch_cryptopanic():
            if self.seen.add(item["news_id"]):
                produce_json(self.producer, NEWS_RAW.name, item["news_id"], item)
                new += 1
        self.producer.flush(10)
        return new

    def run(self) -> None:
        log.info("polling %d feed(s)%s every %d s", len(self.s.feed_list),
                 " + CryptoPanic" if self.cryptopanic_ok else "", self.s.news_poll_seconds)
        while not self.stop.is_set():
            new = self.poll_once()
            log.info("published %d new item(s)", new)
            self.stop.wait(self.s.news_poll_seconds)
        self.producer.flush(10)


def main() -> None:
    np_ = NewsProducer()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: np_.stop.set())
    np_.run()


if __name__ == "__main__":
    main()
