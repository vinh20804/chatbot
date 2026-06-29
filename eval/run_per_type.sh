#!/bin/bash
# =============================================================================
# Run benchmark across ALL 4 specialized query types sequentially.
# Skips types whose queries file doesn't exist.
#
# Usage:
#   bash eval/run_per_type.sh                        # all 4 types
#   bash eval/run_per_type.sh --types factoid,broad  # subset
#   bash eval/run_per_type.sh --skip-ragas           # BERTScore only
# =============================================================================

set -e
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

TYPES="factoid,relational,broad,aggregate"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --types) TYPES="$2"; shift 2 ;;
        *)       EXTRA_ARGS+=("$1"); shift ;;
    esac
done

IFS=',' read -ra TYPE_LIST <<< "$TYPES"

echo "============================================================"
echo "  Per-Type Benchmark Suite"
echo "============================================================"
echo "Types: ${TYPE_LIST[*]}"
echo "Extra args: ${EXTRA_ARGS[*]:-<none>}"
echo "============================================================"

for t in "${TYPE_LIST[@]}"; do
    qfile="eval/queries_${t}.json"
    if [ ! -f "$qfile" ]; then
        echo ""
        echo "[SKIP] $t — $qfile not found. Generate it with:"
        echo "  python eval/gen_specialized_queries.py --type $t"
        continue
    fi
    echo ""
    echo "▶ Running: $t"
    bash eval/run_all.sh --type "$t" "${EXTRA_ARGS[@]}"
done

echo ""
echo "============================================================"
echo "  All Done @ $(date)"
echo "============================================================"
echo "Reports:"
for t in "${TYPE_LIST[@]}"; do
    if [ -f "eval/results/${t}/report.md" ]; then
        echo "  eval/results/${t}/report.md"
    fi
done
