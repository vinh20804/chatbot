#!/bin/bash
# =============================================================================
# LightRAG Benchmark Pipeline — Per-Type Specialized Runner
# Runs bench + BERTScore + RAGAS for a SINGLE query type, isolates each type
# into its own results dir to make per-mode strengths visible.
#
# Usage:
#   bash eval/run_all.sh --type factoid              # bench just factoid
#   bash eval/run_all.sh --type relational
#   bash eval/run_all.sh --type broad
#   bash eval/run_all.sh --type aggregate
#   bash eval/run_all.sh --queries path/to.json --type custom_name
#
# Run ALL 4 types sequentially:
#   bash eval/run_per_type.sh
#
# Flags:
#   --type NAME           type label (also output dir name) — REQUIRED
#   --queries FILE        query JSON (default: eval/queries_{type}.json)
#   --modes LIST          comma-separated modes (default: bm25,naive,hybrid,mix,graph)
#   --skip-bench          reuse existing results_raw.json
#   --skip-ragas          skip RAGAS (BERTScore only)
#   --server-url URL      default http://localhost:9621
#
# Output:
#   eval/results/{type}/results_raw.json
#   eval/results/{type}/results_eval.json
#   eval/results/{type}/results_chunks.json
#   eval/results/{type}/results_ragas.json
#   eval/results/{type}/report.md
#
# Prerequisites:
#   - .env configured (POSTGRES, LLM_BINDING_*, COHERE_*, WORKING_DIR)
#   - LightRAG server running on port 9621
#   - tlu_workspace/ contains graph_chunk_entity_relation.graphml
#   - Python venv activated
# =============================================================================

set -e
set -u
set -o pipefail

# Auto-detect repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# ---- Defaults ----
TYPE=""
QUERIES_FILE=""
MODES="bm25,naive,hybrid,mix,graph"
SKIP_BENCH=false
SKIP_RAGAS=false
SERVER_URL="http://localhost:9621"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)           TYPE="$2"; shift 2 ;;
        --queries)        QUERIES_FILE="$2"; shift 2 ;;
        --modes)          MODES="$2"; shift 2 ;;
        --skip-bench)     SKIP_BENCH=true; shift ;;
        --skip-ragas)     SKIP_RAGAS=true; shift ;;
        --server-url)     SERVER_URL="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | head -40
            exit 0 ;;
        *)                echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Validate --type
if [ -z "$TYPE" ]; then
    echo "[FATAL] --type is required. Use: --type factoid|relational|broad|aggregate"
    exit 1
fi

# Default queries file based on type
if [ -z "$QUERIES_FILE" ]; then
    QUERIES_FILE="eval/queries_${TYPE}.json"
fi

# Output dir per type
OUT_DIR="eval/results/${TYPE}"
mkdir -p "$OUT_DIR"

# ---- Banner ----
echo "============================================================"
echo "  LightRAG Per-Type Benchmark — Type: ${TYPE}"
echo "============================================================"
echo "REPO_DIR:    $REPO_DIR"
echo "Queries:     $QUERIES_FILE"
echo "Modes:       $MODES"
echo "Output:      $OUT_DIR"
echo "Server URL:  $SERVER_URL"
echo "Skip bench:  $SKIP_BENCH"
echo "Skip RAGAS:  $SKIP_RAGAS"
echo "Time:        $(date)"
echo "============================================================"
echo ""

# ---- Pre-flight ----
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo "[WARN] No venv activated."
fi

if [ ! -f "$QUERIES_FILE" ]; then
    echo "[FATAL] Queries file not found: $QUERIES_FILE"
    echo "  Generate it with: python eval/gen_specialized_queries.py --type $TYPE"
    exit 1
fi

if [ ! -f ".env" ] && [ ! -f "eval/.env" ]; then
    echo "[FATAL] No .env found at repo root or eval/.env"
    exit 1
fi

if [ "$SKIP_BENCH" = false ]; then
    if ! curl -s -o /dev/null --max-time 5 "$SERVER_URL/health"; then
        echo "[FATAL] LightRAG server not responding at $SERVER_URL"
        exit 1
    fi
    echo "[OK] Server alive at $SERVER_URL"
fi

# ---- Step 1: Run benchmark ----
RESULTS_RAW="$OUT_DIR/results_raw.json"
if [ "$SKIP_BENCH" = false ]; then
    echo ""
    echo "============================================================"
    echo "  Step 1/4: Run benchmark"
    echo "============================================================"
    python -u eval/run_benchmark.py \
        --queries "$QUERIES_FILE" \
        --modes "$MODES" \
        --out "$RESULTS_RAW" \
        --server-url "$SERVER_URL" \
        --resume \
        2>&1 | tee "$OUT_DIR/run_benchmark.log"
    echo "[OK] Step 1 done → $RESULTS_RAW"
else
    echo "[SKIP] Step 1 (benchmark)"
fi

# ---- Step 2: BERTScore (only for factoid; other types use pairwise instead) ----
if [ "$TYPE" = "factoid" ]; then
    echo ""
    echo "============================================================"
    echo "  Step 2/4: BERTScore evaluation (factoid only)"
    echo "============================================================"
    python -u eval/evaluate_benchmark.py \
        --queries "$QUERIES_FILE" \
        --raw "$RESULTS_RAW" \
        --out-eval "$OUT_DIR/results_eval.json" \
        --out-report "$OUT_DIR/report.md" \
        2>&1 | tee "$OUT_DIR/evaluate_benchmark.log"
    echo "[OK] Step 2 done → $OUT_DIR/results_eval.json + report.md"
else
    echo ""
    echo "[SKIP] Step 2 BERTScore for type=$TYPE (use pairwise judge later)"
fi

# ---- Step 3: RAGAS ----
if [ "$SKIP_RAGAS" = false ]; then
    echo ""
    echo "============================================================"
    echo "  Step 3a: Fetch chunks for RAGAS"
    echo "============================================================"
    python -u eval/fetch_chunks.py \
        --queries "$QUERIES_FILE" \
        --eval "$OUT_DIR/results_eval.json" \
        --out "$OUT_DIR/results_chunks.json" \
        --server-url "$SERVER_URL" \
        2>&1 | tee "$OUT_DIR/fetch_chunks.log"

    echo ""
    echo "============================================================"
    echo "  Step 3b: RAGAS 4 metrics"
    echo "============================================================"
    python -u eval/evaluate_ragas.py \
        --metric faithfulness,answer_relevancy \
        --queries "$QUERIES_FILE" \
        --eval "$OUT_DIR/results_eval.json" \
        --chunks "$OUT_DIR/results_chunks.json" \
        --out "$OUT_DIR/results_ragas.json" \
        2>&1 | tee "$OUT_DIR/evaluate_ragas.log"

    echo ""
    echo "============================================================"
    echo "  Step 3c: Retry failed RAGAS faithfulness"
    echo "============================================================"
    python -u eval/retry_failed_ragas.py \
        --metric faithfulness \
        --queries "$QUERIES_FILE" \
        --eval "$OUT_DIR/results_eval.json" \
        --chunks "$OUT_DIR/results_chunks.json" \
        --ragas "$OUT_DIR/results_ragas.json" \
        2>&1 | tee "$OUT_DIR/retry_faithfulness.log"
else
    echo "[SKIP] Step 3 (RAGAS)"
fi

# ---- Summary ----
echo ""
echo "============================================================"
echo "  Complete: ${TYPE} @ $(date)"
echo "============================================================"
ls -lh "$OUT_DIR/"*.json "$OUT_DIR/"*.md 2>/dev/null || true
echo ""
echo "Next: cat $OUT_DIR/report.md"
