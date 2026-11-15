#!/usr/bin/env bash
# Start / stop the 4 processing services of the demo in the background (logs in logs/).
#   bash scripts/demo.sh start     then: make replay ...   and   make dashboard
#   bash scripts/demo.sh status
#   bash scripts/demo.sh stop
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p logs
SERVICES="hot_writer:processing.hot_writer features:processing.feature_consumer scorer:processing.live_scorer agent:agent.consumer"

start() {
  for s in $SERVICES; do
    name=${s%%:*}; mod=${s#*:}
    if [ -f "logs/$name.pid" ] && kill -0 "$(cat logs/$name.pid)" 2>/dev/null; then
      echo "  $name already running"; continue
    fi
    nohup $PY -m "$mod" > "logs/$name.log" 2>&1 &
    echo $! > "logs/$name.pid"
    echo "  started $name (log: logs/$name.log)"
  done
  echo "Wait ~10 s, then: make replay DAY=2024-08-05 FROM=00:00 TO=03:00 SPEED=20"
}

stop() {
  for s in $SERVICES; do
    name=${s%%:*}
    if [ -f "logs/$name.pid" ]; then
      kill "$(cat logs/$name.pid)" 2>/dev/null && echo "  stopped $name"
      rm -f "logs/$name.pid"
    fi
  done
}

status() {
  for s in $SERVICES; do
    name=${s%%:*}
    if [ -f "logs/$name.pid" ] && kill -0 "$(cat logs/$name.pid)" 2>/dev/null; then
      echo "  $name: running  | $(tail -n 1 logs/$name.log 2>/dev/null | cut -c1-110)"
    else
      echo "  $name: stopped"
    fi
  done
}

case "${1:-status}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 start|stop|status"; exit 1 ;;
esac
