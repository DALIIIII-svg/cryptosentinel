#!/usr/bin/env bash
# Day-1 Check A: can AWS us-east-1 reach Binance market data?
# Run it ON a throwaway EC2 instance (Amazon Linux 2023), not on your laptop:
#   nano check_binance.sh   (paste this file, save)
#   bash check_binance.sh
set -u

ok()   { echo "  [PASS] $*"; }
bad()  { echo "  [FAIL] $*"; }

echo "== 1. REST  data-api.binance.vision (market-data host, should work) =="
code=$(curl -s -o /tmp/cs_klines.json -w '%{http_code}' \
  "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=2")
if [ "$code" = "200" ]; then ok "HTTP 200: $(head -c 120 /tmp/cs_klines.json)..."; else bad "HTTP $code"; fi

echo "== 2. REST  api.binance.com (normal host, expected to be blocked with 451 from the US) =="
code=$(curl -s -o /dev/null -w '%{http_code}' "https://api.binance.com/api/v3/ping")
echo "  HTTP $code  (451 = geo-blocked, as expected; 200 = not blocked)"

echo "== 3. WebSocket  data-stream.binance.vision =="
if ! python3 -c "import websockets" 2>/dev/null; then
  python3 -m pip --version >/dev/null 2>&1 || sudo dnf install -y -q python3-pip
  python3 -m pip install --user -q websockets
fi
python3 - <<'PY'
import asyncio, json
import websockets

async def main():
    url = "wss://data-stream.binance.vision/ws/btcusdt@kline_1m"
    async with websockets.connect(url, open_timeout=15) as ws:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
        k = msg.get("k", {})
        print(f"  [PASS] kline received: {msg.get('s')} close={k.get('c')} closed={k.get('x')}")

try:
    asyncio.run(main())
except Exception as exc:
    print(f"  [FAIL] {type(exc).__name__}: {exc}")
PY

echo "== 4. Fallback  Coinbase public REST =="
code=$(curl -s -o /tmp/cs_cb.json -w '%{http_code}' \
  "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=60")
if [ "$code" = "200" ]; then ok "Coinbase reachable"; else bad "Coinbase HTTP $code"; fi

echo
echo "Write the results in README.md > Environment facts, then TERMINATE this instance."
