#!/usr/bin/env bash
# Track B real runs against the shared endpoint, through the logging proxy.
#   bash scripts/run_real.sh one     # Task 3a: one episode
#   bash scripts/run_real.sh pilot   # Task 3b: 6-episode mini-pilot
# Refuses to start without ~/collusion_shared/endpoint.txt or while GPU_BUSY exists
# (the proxy additionally pauses mid-run whenever GPU_BUSY appears).
set -euo pipefail
cd "$(dirname "$0")/.."
which="${1:?usage: run_real.sh one|pilot}"
case "$which" in one) cfg=configs/real_one.yaml out=outputs/real_one ;; pilot) cfg=configs/pilot.yaml out=outputs/pilot ;; *) echo "one|pilot"; exit 2 ;; esac

shared="$HOME/collusion_shared"
[ -f "$shared/endpoint.txt" ] || { echo "no $shared/endpoint.txt yet"; exit 1; }
[ ! -e "$shared/GPU_BUSY" ] || { echo "GPU_BUSY present; wait"; exit 1; }
upstream="$(tr -d '[:space:]' < "$shared/endpoint.txt")"
echo "upstream=$upstream"

vpy=.venv/Scripts/python            # relative to repo/
[ -x "repo/$vpy" ] || vpy=.venv/bin/python
py="repo/$vpy"
log="outputs/proxy_${which}_$(date +%Y%m%d-%H%M%S).jsonl"
"$py" -I scripts/proxy.py --upstream "$upstream" --port 8032 --max-concurrency 4 --log "$log" &
proxy_pid=$!
trap 'kill $proxy_pid 2>/dev/null || true' EXIT
sleep 2

start=$(date +%s)
(cd repo && "$vpy" -m experiments.collusion.run --config "../$cfg")
echo "wall_clock_s=$(( $(date +%s) - start ))"

root="$out/$(ls -t "$out" | head -1)"
(cd repo && "$vpy" experiments/collusion/compute_jira_optimal.py --root "../$root" --write-json >/dev/null)
"$py" -I scripts/summarize.py "$root" --proxy-log "$log"
