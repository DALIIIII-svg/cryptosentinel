"""How is the data collection going? Prints, for the last days:
  - raw files and size per topic and day in S3
  - the latest stored candle per symbol and how old it is
  - news items per coin in DynamoDB

Run:  python -m processing.collection_report   (or: make report)
"""
from __future__ import annotations

from collections import defaultdict

from boto3.dynamodb.conditions import Key

from common.aws_clients import dynamodb_resource, s3_client
from common.config import get_settings
from common.schemas import now_ms
from common.topics import NEWS_RAW, PRICES_KLINES


def s3_summary() -> None:
    s = get_settings()
    stats: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s.s3_bucket, Prefix="raw/"):
        for obj in page.get("Contents", []):
            parts = obj["Key"].split("/")
            if len(parts) >= 3 and parts[2].startswith("dt="):
                st = stats[(parts[1], parts[2][3:])]
                st[0] += 1
                st[1] += obj["Size"]
    print("S3 raw zone (files / size):")
    if not stats:
        print("  nothing yet")
    for (topic, day), (n, size) in sorted(stats.items()):
        print(f"  {topic:15s} {day}  {n:5d} files  {size / 1024:8.1f} KB")


def prices_summary() -> None:
    s = get_settings()
    table = dynamodb_resource().Table(s.table_name("prices-hot"))
    now = now_ms()
    print("\nLatest closed candle per symbol (DynamoDB prices-hot):")
    for sym in s.symbol_list:
        resp = table.query(KeyConditionExpression=Key("symbol").eq(sym), ScanIndexForward=False, Limit=1)
        items = resp.get("Items", [])
        if not items:
            print(f"  {sym:10s} no data yet")
            continue
        it = items[0]
        age_min = (now - int(it["close_time"])) / 60000
        flag = "OK" if age_min < 3 else "LATE"
        print(f"  {sym:10s} close={float(it['close']):<14g} {age_min:6.1f} min ago  {flag}")


def news_summary() -> None:
    s = get_settings()
    table = dynamodb_resource().Table(s.table_name("news"))
    counts: dict[str, int] = defaultdict(int)
    kwargs: dict = {"ProjectionExpression": "coin"}
    while True:
        resp = table.scan(**kwargs)
        for it in resp.get("Items", []):
            counts[it["coin"]] += 1
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    print("\nNews items per coin (DynamoDB news, last 30 days):")
    if not counts:
        print("  nothing yet")
    for coin, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {coin:10s} {n}")


def main() -> None:
    print(f"Topics: {PRICES_KLINES.name}, {NEWS_RAW.name}\n")
    for part in (s3_summary, prices_summary, news_summary):
        try:
            part()
        except Exception as exc:  # noqa: BLE001
            print(f"  could not read ({type(exc).__name__}: {exc}). Is the stack up? Try: make up && make init")


if __name__ == "__main__":
    main()
