"""Run RAG benchmark: 20 queries × 4 modes → save responses.

Loop:
  for query in queries:
    for mode in [naive, hybrid, mix, graph]:
      POST /query → save {response, latency}
  Throttle 2s between calls (NIM 40 RPM safe).

Usage:
  python run_benchmark.py                    # 1-run, all queries × all modes
  python run_benchmark.py --runs 3           # 3-run for median later
  python run_benchmark.py --modes naive,mix  # subset modes
  python run_benchmark.py --resume           # skip already-done (query, mode, run)
"""

import argparse
import json
import os
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

# Force UTF-8 stdout (Windows console cp1252 default breaks on Vietnamese)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).parent
QUERIES_FILE = HERE / "benchmark_queries.json"
RESULTS_FILE = HERE / "results_raw.json"

SERVER_URL = "http://localhost:9621"
ENDPOINT = f"{SERVER_URL}/query"  # rebuilt in main() if --server-url passed
DEFAULT_MODES = ["naive", "hybrid", "mix", "graph", "bm25"]
SERVER_MODES = {"naive", "hybrid", "mix", "graph"}
LOCAL_MODES = {"bm25"}
BM25_TOP_K = 10
THROTTLE_SEC = 2.0
TIMEOUT_SEC = 300  # multihop queries need more time for heavy modes

# Lazy-loaded BM25 index (only when bm25 mode used)
_bm25_index = None


def get_bm25_index():
    """Lazy-load BM25 index on first bm25 call."""
    global _bm25_index
    if _bm25_index is None:
        import bm25_index as bm25_mod
        _bm25_index = bm25_mod.load_index()
    return _bm25_index


def load_queries(path=None):
    p = Path(path) if path else QUERIES_FILE
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def load_results(path=None):
    p = Path(path) if path else RESULTS_FILE
    if not p.exists():
        return {"meta": {}, "results": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(data, path=None):
    p = Path(path) if path else RESULTS_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def already_done(results, qid, mode, run_idx):
    for r in results:
        if r["query_id"] == qid and r["mode"] == mode and r["run"] == run_idx:
            return True
    return False


def call_query_server(query: str, mode: str) -> dict:
    """POST /query to LightRAG server. Returns {response, latency_sec, error}."""
    payload = {"query": query, "mode": mode, "stream": False}
    t0 = time.time()
    try:
        r = requests.post(ENDPOINT, json=payload, timeout=TIMEOUT_SEC)
        r.raise_for_status()
        elapsed = time.time() - t0
        data = r.json()
        return {
            "response": data.get("response", ""),
            "latency_sec": round(elapsed, 2),
            "error": None,
        }
    except requests.HTTPError as e:
        return {
            "response": "",
            "latency_sec": round(time.time() - t0, 2),
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
        }
    except Exception as e:
        return {
            "response": "",
            "latency_sec": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def call_query_bm25(query: str) -> dict:
    """BM25 retrieve top-K chunks + LLM synthesize. Returns same shape as server."""
    import bm25_index as bm25_mod
    import llm_call

    t0 = time.time()
    try:
        idx = get_bm25_index()
        chunks = bm25_mod.search(query, top_k=BM25_TOP_K, index=idx)
        if not chunks:
            return {
                "response": "Không tìm thấy thông tin trong tài liệu.",
                "latency_sec": round(time.time() - t0, 2),
                "error": None,
            }
        llm_result = llm_call.call_llm(query, chunks)
        elapsed = time.time() - t0
        return {
            "response": llm_result["response"],
            "latency_sec": round(elapsed, 2),
            "error": llm_result["error"],
        }
    except Exception as e:
        return {
            "response": "",
            "latency_sec": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }


def call_query(query: str, mode: str) -> dict:
    """Dispatch mode → server or local BM25."""
    if mode in LOCAL_MODES:
        return call_query_bm25(query)
    return call_query_server(query, mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs", type=int, default=1, help="Number of runs per (query,mode) for median"
    )
    parser.add_argument(
        "--modes", default=",".join(DEFAULT_MODES), help="Comma-separated modes"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already-completed entries"
    )
    parser.add_argument(
        "--throttle", type=float, default=THROTTLE_SEC, help="Seconds between calls"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of queries (debug)"
    )
    # CLI overrides for paths (used by run_all.sh per-type)
    parser.add_argument("--queries", default=None, help="Queries JSON path override")
    parser.add_argument("--out", default=None, help="Output results_raw.json path")
    parser.add_argument("--server-url", default=None, help="LightRAG server URL")
    args = parser.parse_args()

    # Apply CLI overrides
    global ENDPOINT, RESULTS_FILE
    if args.server_url:
        ENDPOINT = f"{args.server_url.rstrip('/')}/query"
    if args.out:
        RESULTS_FILE = Path(args.out)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    data = load_results(args.out) if args.resume else {"meta": {}, "results": []}
    data["meta"] = {
        "started_at": datetime.now().isoformat(),
        "modes": modes,
        "runs": args.runs,
        "throttle_sec": args.throttle,
        "total_queries": len(queries),
        "total_calls_planned": len(queries) * len(modes) * args.runs,
    }

    total = len(queries) * len(modes) * args.runs
    done_count = 0
    skip_count = 0
    error_count = 0

    print(
        f"[BENCHMARK] {len(queries)} queries × {len(modes)} modes × {args.runs} runs = {total} calls"
    )
    print(f"[BENCHMARK] Throttle: {args.throttle}s | Endpoint: {ENDPOINT}")
    print(
        f"[BENCHMARK] Estimated wall time: ~{total * (args.throttle + 8) / 60:.1f} min\n"
    )

    # Warm-up: load model into VRAM + BM25 index into memory before timing starts
    server_modes = [m for m in modes if m in SERVER_MODES]
    if server_modes:
        print("[WARMUP] Sending warm-up query to server (load model into VRAM)...")
        call_query_server("xin chào", server_modes[0])
        print("[WARMUP] Server done.")
    if "bm25" in modes:
        print("[WARMUP] Loading BM25 index...")
        get_bm25_index()
        print("[WARMUP] BM25 done.")
    print()

    for run_idx in range(1, args.runs + 1):
        for q in queries:
            for mode in modes:
                done_count += 1
                tag = f"[{done_count}/{total}] run={run_idx} q={q['id']} mode={mode}"

                if args.resume and already_done(
                    data["results"], q["id"], mode, run_idx
                ):
                    print(f"{tag} SKIP (resume)")
                    skip_count += 1
                    continue

                result = call_query(q["query"], mode)
                entry = {
                    "query_id": q["id"],
                    "type": q["type"],
                    "topic": q.get("topic", q.get("subject", "")),
                    "subject": q.get("subject", ""),
                    "mode": mode,
                    "run": run_idx,
                    "query": q["query"],
                    "response": result["response"],
                    "latency_sec": result["latency_sec"],
                    "error": result["error"],
                    "timestamp": datetime.now().isoformat(),
                }
                data["results"].append(entry)

                if result["error"]:
                    error_count += 1
                    print(f"{tag} ERROR ({result['latency_sec']}s): {result['error']}")
                else:
                    resp_preview = result["response"][:80].replace("\n", " ")
                    print(f"{tag} OK ({result['latency_sec']}s) | {resp_preview}...")

                # Save after EVERY call (crash-safe)
                save_results(data)

                # Throttle (skip on last call)
                if done_count < total:
                    time.sleep(args.throttle)

    data["meta"]["finished_at"] = datetime.now().isoformat()
    data["meta"]["errors"] = error_count
    data["meta"]["skipped"] = skip_count
    save_results(data)

    print(f"\n[DONE] {done_count} calls | {error_count} errors | {skip_count} skipped")
    print(f"[DONE] Results: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
