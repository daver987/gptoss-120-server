#!/usr/bin/env bash
set -euo pipefail
stop_one () {
  local name="$1" pidfile="$2"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || echo "")"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[stop] killing ${name} (pid ${pid})"
      kill "$pid" || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  else
    echo "[stop] ${name}: no pidfile"
  fi
}

stop_one "responses" ".run/responses.pid"
stop_one "max" ".run/max.pid"
echo "[stop] done."
