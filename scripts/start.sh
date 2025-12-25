#!/usr/bin/env sh
set -eu

HOST="${HOST:-0.0.0.0}"
PORT_VALUE="${PORT:-7860}"

case "$PORT_VALUE" in
  ''|*[!0-9]*)
    echo "WARN: PORT='$PORT_VALUE' is not an integer; falling back to 7860" >&2
    PORT_VALUE="7860"
    ;;
esac

exec granian --interface asgi server.app:app --host "$HOST" --port "$PORT_VALUE" --log-level info
