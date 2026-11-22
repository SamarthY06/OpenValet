#!/usr/bin/env bash
# End-to-end: build/OpenValet.app ⇄ scripts/fake_daemon.py over real unix sockets in a temp
# MAC_HOME (socket paths must stay < 104 bytes, hence $TMPDIR not a deep path).
# Plays a short, quiet (amplitude 0.05) tone; E2E_AMP=0 plays digital silence instead.
# Leaves nothing running.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="${1:-$ROOT/build/OpenValet.app}"
H="$(mktemp -d -t mace2e)"
# Only ever signal the processes THIS script started (other OpenValet.app instances may be live).
APP_PID=""
cleanup() {
  [[ -n $APP_PID ]] && kill -TERM "$APP_PID" 2>/dev/null || true
  [[ -n ${FAKE:-} ]] && kill "$FAKE" 2>/dev/null || true
  sleep 0.5; rm -rf "$H"
}
trap cleanup EXIT
MAC_HOME="$H" python3 "$ROOT/scripts/fake_daemon.py" --timeout 30 --amp "${E2E_AMP:-0.05}" >"$H/fake.out" 2>&1 &
FAKE=$!
sleep 0.5
# Exec the bundle's binary directly (bundle identity/Info.plist still apply) so we know its PID.
MAC_HOME="$H" MAC_LOG_DIR="$H/logs" MAC_SUPERVISE=0 MAC_NO_ONBOARDING=1 MAC_NO_HUD=1 MAC_AEC_IDLE_S=3 MAC_E2E=1 \
  "$APP/Contents/MacOS/Mac" >"$H/app.out" 2>&1 &
APP_PID=$!
rc=0; wait "$FAKE" || rc=$?
FAKE=
grep -E "PASS|FAIL|info:" "$H/fake.out" | sed 's/^\[fake [0-9:]*\] /  /'
echo "--- app log"; grep -E "engine started|HELLO sent|flushed|carried|watchdog" "$H/logs/app.log" | sed 's/^/  /' || true
[[ $rc == 0 ]] || { echo "E2E FAILED (rc=$rc)"; cat "$H/fake.out"; exit $rc; }
# The held UI thread must have been noticed, timed and sampled (the sample is written by a child).
for i in $(seq 1 20); do ls "$H/logs/stalls"/stall-*.txt >/dev/null 2>&1 && break; sleep 0.25; done
if grep -q "main thread is not responding" "$H/logs/app.log" && grep -Eq "main thread is back after [45]\.[0-9] s" "$H/logs/app.log" \
   && grep -q "Call graph" "$H/logs/stalls"/stall-*.txt 2>/dev/null; then
  echo "  PASS watchdog: stall logged with its length, stack sample saved"
else
  echo "  FAIL watchdog: stall not logged or no stack sample"; exit 1
fi

kill -TERM "$APP_PID" 2>/dev/null || true  # part 1's app (it holds the microphone); part 2 starts its own
for i in $(seq 1 20); do kill -0 "$APP_PID" 2>/dev/null || break; sleep 0.25; done
kill -KILL "$APP_PID" 2>/dev/null || true

# ---- part 2: Quit must not leave orphans. A stub "daemon" that ignores SIGTERM and spawns a
# grandchild (stand-in for the `claude` CLI, argv0 "mac-e2e-claude" so we never match the
# user's real claude). Expect: SIGKILL after the grace period, grandchild gone.
D="$H/stub"; mkdir -p "$D/.venv/bin"
cat >"$D/.venv/bin/python" <<'STUB'
#!/bin/bash
trap '' TERM
bash -c 'exec -a mac-e2e-claude sleep 1000' &
while true; do sleep 1; done
STUB
chmod +x "$D/.venv/bin/python"
MAC_HOME="$H" MAC_LOG_DIR="$H/logs2" MAC_DAEMON_DIR="$D" MAC_NO_ONBOARDING=1 MAC_NO_HUD=1 \
  "$APP/Contents/MacOS/Mac" >"$H/app2.out" 2>&1 &
APP_PID=$!
for i in $(seq 1 40); do pgrep -f mac-e2e-claude >/dev/null && break; sleep 0.25; done
if ! pgrep -f mac-e2e-claude >/dev/null; then echo "  FAIL quit_no_orphans: stub grandchild never appeared"; exit 1; fi
t0=$(date +%s); kill -TERM "$APP_PID"; while kill -0 "$APP_PID" 2>/dev/null; do sleep 0.2; done; dt=$(( $(date +%s) - t0 ))
sleep 0.5
if pgrep -f mac-e2e-claude >/dev/null; then echo "  FAIL quit_no_orphans: grandchild survived quit"; pkill -f mac-e2e-claude; exit 1; fi
echo "  PASS quit_no_orphans: app quit in ${dt}s (stub ignored SIGTERM → SIGKILL), grandchild killed"
APP_PID=""
echo "E2E OK"
exit 0
