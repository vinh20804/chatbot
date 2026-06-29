"""Fetch retrieved chunks for each (query_id, mode) by re-querying LightRAG.

POST /query with only_need_context=true → returns context string with chunks.
Parse + save to results_chunks.json.

For BM25 mode: call bm25_index directly (no server roundtrip).

Output format:
  [{"query_id": "...", "mode": "...", "chunks": ["text1", "text2", ...]}, ...]

Usage:
  python fetch_chunks.py                    # All modes × all queries
  python fetch_chunks.py --modes naive,mix  # Subset
  python fetch_chunks.py --limit 3          # Smoke test
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).parent
QUERIES_FILE = HERE / "benchmark_queries.json"
CHUNKS_FILE = HERE / "results_chunks.json"

SERVER_URL = "http://localhost:9621"
ENDPOINT = f"{SERVER_URL}/query"
DEFAULT_MODES = ["naive", "hybrid", "mix", "graph", "bm25"]
TIMEOUT = 180


def load_queries(path=None):
    from pathlib import Path as _P
    p = _P(path) if path else QUERIES_FILE
    with open(p, encoding="utf-8") as f:
        return json.load(f)["queries"]


def parse_chunks_from_context(context_text: str) -> list:
    """Parse LightRAG context output into chunk list.

    Context format includes sections; we split on common chunk separators.
    Fallback: return whole context as single chunk.
    """
    # LightRAG context blocks are separated by --- or numbered prefixes
    # Try splitting on chunk delimiters
    chunks = re.split(r"\n-{3,}\n|\n={3,}\n|\n\[\d+\]\s*", context_text)
    chunks = [c.strip() for c in chunks if c.strip() and len(c.strip()) > 30]
    if not chunks:
        return [context_text]
    return chunks[:15]  # cap


def fetch_server(query: str, mode: str) -> list:
    import requests
    payload = {"query": query, "mode": mode, "only_need_context": True, "stream": False}
    r = requests.post(ENDPOINT, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    ctx = r.json().get("response", "")
    return parse_chunks_from_context(ctx)


def fetch_bm25(query: str, top_k: int = 10) -> list:
    import bm25_index
    chunks = bm25_index.search(query, top_k=top_k)
    return [c["content"] for c in chunks]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--throttle", type=float, default=2.0)
    parser.add_argument("--resume", action="store_true", help="Skip existing entries")
    # Path overrides
    parser.add_argument("--queries", default=None, help="Queries JSON path")
    parser.add_argument("--eval", default=None, help="results_eval.json path (for query/mode list)")
    parser.add_argument("--out", default=None, help="Output results_chunks.json path")
    parser.add_argument("--server-url", default=None, help="LightRAG server URL")
    args = parser.parse_args()

    global ENDPOINT, CHUNKS_FILE
    if args.server_url:
        ENDPOINT = f"{args.server_url.rstrip('/')}/query"
    if args.out:
        CHUNKS_FILE = Path(args.out)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]

    # Load existing
    existing = []
    done_keys = set()
    if args.resume and CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, encoding="utf-8") as f:
            existing = json.load(f)
        done_keys = {(c["query_id"], c["mode"]) for c in existing}
        print(f"[RESUME] {len(existing)} entries already done")

    total = len(queries) * len(modes)
    cnt = 0
    for q in queries:
        for mode in modes:
            cnt += 1
            key = (q["id"], mode)
            if key in done_keys:
                print(f"[{cnt}/{total}] {q['id']}/{mode} SKIP (resume)")
                continue

            tag = f"[{cnt}/{total}] {q['id']}/{mode}"
            t0 = time.time()
            try:
                if mode == "bm25":
                    chunks = fetch_bm25(q["query"])
                else:
                    chunks = fetch_server(q["query"], mode)
                elapsed = time.time() - t0
                existing.append({
                    "query_id": q["id"],
                    "mode": mode,
                    "chunks": chunks,
                    "chunk_count": len(chunks),
                    "latency_sec": round(elapsed, 2),
                })
                print(f"{tag} OK ({elapsed:.1f}s) | {len(chunks)} chunks")
            except Exception as ex:
                existing.append({
                    "query_id": q["id"],
                    "mode": mode,
                    "chunks": [],
                    "error": f"{type(ex).__name__}: {str(ex)[:100]}",
                })
                print(f"{tag} ERROR: {ex}")

            # Save after every entry (crash-safe)
            with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

            time.sleep(args.throttle)

    print(f"\n[DONE] {len(existing)} entries → {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
