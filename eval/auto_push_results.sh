#!/bin/bash
# =============================================================================
# Auto-push benchmark results to GitHub after run_per_type.sh completes.
#
# Strategy: poll for all 4 expected report.md files. When all present + bench
# pipeline no longer running, commit + push results to a dedicated branch.
#
# Usage:
#   bash eval/auto_push_results.sh                          # default: poll forever
#   bash eval/auto_push_results.sh --branch results-v2      # custom branch name
#   bash eval/auto_push_results.sh --types factoid,broad    # subset
#   bash eval/auto_push_results.sh --max-wait 21600         # max 6h then exit
#   bash eval/auto_push_results.sh --once                   # check once, no poll
#
# Run alongside bench:
#   tmux new -d -s push 'bash eval/auto_push_results.sh'
# =============================================================================

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ---- Defaults ----
TYPES="factoid,relational,broad,aggregate"
BRANCH="main"             # push directly to main
POLL_SEC=120              # check every 2 min after initial sleep
MAX_WAIT=21600            # max 6h wait time
ONCE=false
INITIAL_SLEEP=21600       # default: sleep 6h before first check (bench expected duration)

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --types)         TYPES="$2"; shift 2 ;;
        --branch)        BRANCH="$2"; shift 2 ;;
        --poll)          POLL_SEC="$2"; shift 2 ;;
        --max-wait)      MAX_WAIT="$2"; shift 2 ;;
        --once)          ONCE=true; shift ;;
        --sleep)         INITIAL_SLEEP="$2"; shift 2 ;;
        --no-sleep)      INITIAL_SLEEP=0; shift ;;
        *)               echo "Unknown arg: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra TYPE_LIST <<< "$TYPES"

# ---- Banner ----
echo "============================================================"
echo "  Auto-Push Results Watcher"
echo "============================================================"
echo "REPO_DIR:       $REPO_DIR"
echo "Types:          ${TYPE_LIST[*]}"
echo "Branch:         $BRANCH"
echo "Initial sleep:  ${INITIAL_SLEEP}s ($(($INITIAL_SLEEP / 3600))h)"
echo "Poll every:     ${POLL_SEC}s"
echo "Max wait:       ${MAX_WAIT}s ($(($MAX_WAIT / 3600))h)"
echo "Mode:           $([ "$ONCE" = true ] && echo 'one-shot after sleep' || echo 'poll after sleep')"
echo "Start time:     $(date)"
echo "ETA push:       $(date -d "+${INITIAL_SLEEP} seconds" 2>/dev/null || date -v+${INITIAL_SLEEP}S 2>/dev/null || echo '?')"
echo "============================================================"

# ---- Initial sleep (wait for bench to finish) ----
if [ "$INITIAL_SLEEP" -gt 0 ]; then
    echo ""
    echo "[SLEEP] Waiting ${INITIAL_SLEEP}s before first check..."
    echo "[SLEEP] Wake up at: $(date -d "+${INITIAL_SLEEP} seconds" 2>/dev/null || date -v+${INITIAL_SLEEP}S 2>/dev/null)"
    sleep "$INITIAL_SLEEP"
    echo "[WAKE] Initial sleep done @ $(date). Starting checks..."
fi

# ---- Helper: check if all reports exist ----
all_reports_exist() {
    for t in "${TYPE_LIST[@]}"; do
        if [ ! -f "eval/results/$t/report.md" ]; then
            return 1
        fi
    done
    return 0
}

# ---- Helper: check if bench still running ----
bench_running() {
    # Any process running run_benchmark.py or evaluate_*.py?
    if pgrep -f "run_benchmark.py\|evaluate_benchmark.py\|evaluate_ragas.py\|fetch_chunks.py\|retry_failed_ragas.py" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ---- Helper: count completed reports ----
count_reports() {
    local cnt=0
    for t in "${TYPE_LIST[@]}"; do
        [ -f "eval/results/$t/report.md" ] && cnt=$((cnt + 1))
    done
    echo $cnt
}

# ---- Poll loop ----
elapsed=0
while true; do
    n_done=$(count_reports)
    total=${#TYPE_LIST[@]}
    is_running="no"
    bench_running && is_running="yes"

    echo "[$(date '+%H:%M:%S')] reports=$n_done/$total | bench_running=$is_running | elapsed=${elapsed}s"

    # All reports present AND bench not running → ready to push
    if all_reports_exist && ! bench_running; then
        echo ""
        echo "[READY] All ${total} reports present + bench idle. Pushing..."
        break
    fi

    if [ "$ONCE" = true ]; then
        echo "[ONCE] Not all done. Exiting (--once mode)."
        exit 1
    fi

    if [ $elapsed -ge $MAX_WAIT ]; then
        echo "[TIMEOUT] Reached max wait ${MAX_WAIT}s. Pushing what's available..."
        if [ "$n_done" -eq 0 ]; then
            echo "[FATAL] No reports to push."
            exit 1
        fi
        break
    fi

    sleep $POLL_SEC
    elapsed=$((elapsed + POLL_SEC))
done

# ---- Git push ----
echo ""
echo "============================================================"
echo "  Git Push"
echo "============================================================"

# Verify git available + we're in a repo
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "[FATAL] Not a git repo: $REPO_DIR"
    exit 1
fi

# Pre-flight: stash any unrelated changes to avoid mixing
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "[INFO] Local uncommitted changes detected — stashing first."
    git stash push -u -m "auto_push_stash_$(date +%s)" -- ':!eval/results/' || true
fi

# Current branch
current_branch=$(git rev-parse --abbrev-ref HEAD)
echo "Current branch: $current_branch"

# Switch to target branch if different
if [ "$current_branch" != "$BRANCH" ]; then
    echo "[INFO] Switching to $BRANCH"
    # Pull latest to avoid conflicts
    git fetch origin "$BRANCH" 2>&1 || true
    if git rev-parse --verify "$BRANCH" > /dev/null 2>&1; then
        git checkout "$BRANCH"
    elif git rev-parse --verify "origin/$BRANCH" > /dev/null 2>&1; then
        git checkout -b "$BRANCH" "origin/$BRANCH"
    else
        echo "[FATAL] Branch $BRANCH not found locally or on remote"
        exit 1
    fi
fi

# Pull latest to merge any remote changes
echo "[INFO] Pulling latest $BRANCH"
git pull origin "$BRANCH" --rebase 2>&1 || {
    echo "[WARN] Pull failed. Continuing — push may reject."
}

# Stage results — only .md and .json, skip *.log (noisy)
git add eval/results/**/*.md eval/results/**/*.json 2>/dev/null || true
# Fallback if shell globstar disabled
for t in "${TYPE_LIST[@]}"; do
    git add "eval/results/$t/report.md" 2>/dev/null || true
    git add "eval/results/$t/"*.json 2>/dev/null || true
done

# Show summary
echo ""
echo "Files staged:"
git diff --cached --stat 2>&1 | tail -20

# Commit (only if changes)
if git diff --cached --quiet; then
    echo "[INFO] No new results to commit."
else
    commit_msg="results: benchmark output for ${TYPE_LIST[*]} ($(date +%Y-%m-%d_%H%M))"
    if git commit -m "$commit_msg" \
        -m "Auto-pushed by auto_push_results.sh" \
        -m "Reports: $(count_reports)/$total complete"; then
        echo "[OK] Committed: $commit_msg"
    else
        echo "[WARN] Commit failed (maybe hooks? continuing to push attempt)"
    fi
fi

# Push
echo ""
echo "Pushing $BRANCH to origin..."
if git push -u origin "$BRANCH" 2>&1; then
    echo "[OK] Push succeeded."
    echo ""
    echo "Branch URL:"
    remote_url=$(git remote get-url origin 2>/dev/null | sed 's|\.git$||')
    if [[ "$remote_url" =~ github\.com ]]; then
        # Convert ssh to https for URL display
        echo "  ${remote_url/git@github.com:/https://github.com/}/tree/$BRANCH"
    fi
else
    echo "[FAIL] Push failed. Manual recovery needed:"
    echo "  cd $REPO_DIR"
    echo "  git status"
    echo "  git push -u origin $BRANCH"
    exit 1
fi

# Switch back to original branch only if we switched away
if [ "$current_branch" != "$BRANCH" ]; then
    echo ""
    echo "Switching back to $current_branch..."
    git checkout "$current_branch" 2>&1
fi

echo ""
echo "============================================================"
echo "  Done @ $(date)"
echo "============================================================"
