#!/bin/bash
# Self-healing experiment runner.
# Called by cron every 10 minutes. Checks if an experiment is in progress,
# and if not, runs the next one from the queue.

set -e

export PATH="/home/codespace/.python/current/bin:$PATH"

LOCKFILE="/tmp/transcription_lab_experiment.lock"
LOGFILE="/project/workspace/transcription-lab/results/runner.log"
PROJECT="/project/workspace/transcription-lab"
SCRIPT="$PROJECT/scripts/run_experiment_queue.py"

# If lock exists and process is still running, exit
if [ -f "$LOCKFILE" ]; then
    PID=$(cat "$LOCKFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "$(date -Iseconds) Experiment PID $PID still running, skipping" >> "$LOGFILE"
        exit 0
    else
        echo "$(date -Iseconds) Stale lock for PID $PID, cleaning up" >> "$LOGFILE"
        rm -f "$LOCKFILE"
    fi
fi

# Run next experiment
echo $$ > "$LOCKFILE"
echo "$(date -Iseconds) Starting next experiment (PID $$)" >> "$LOGFILE"

cd "$PROJECT"
python3 "$SCRIPT" >> "$LOGFILE" 2>&1
EXIT_CODE=$?

rm -f "$LOCKFILE"
echo "$(date -Iseconds) Experiment finished with exit code $EXIT_CODE" >> "$LOGFILE"

# Push results to GitHub (uses git credential helper configured externally)
cd /project/workspace
git add -A
git commit -m "Auto: experiment results update ($(date +%Y%m%d-%H%M%S))

Co-authored-by: factory-droid[bot] <138933559+factory-droid[bot]@users.noreply.github.com>" 2>/dev/null || true
git push origin master 2>/dev/null || echo "$(date -Iseconds) Push failed" >> "$LOGFILE"
