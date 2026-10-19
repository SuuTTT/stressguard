#!/usr/bin/env bash
# Run a job file with bounded parallelism on a shared server: bash scripts/sg_remote_run.sh jobs.txt 40
# Jobs are niced, CPU-only, 2 threads each. Completed jobs (log ends with "done:") are skipped on re-run.
cd "$(dirname "$0")/.."
JOBS=$1; P=${2:-32}
mkdir -p results/logs
grep -v '^\s*$' "$JOBS" | while read -r line; do
  log=$(echo "$line" | sed -E 's/.*> ([^ ]+) 2>&1.*/\1/')
  if [ -f "$log" ] && grep -q "done:" "$log"; then continue; fi
  echo "$line"
done | xargs -P "$P" -I{} bash -c '{}'
echo "ALL_DONE $(date)"
