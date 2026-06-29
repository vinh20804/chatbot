"""Phase G eval: simulate WebUI query calls against backend.

Hits same /query endpoint UI uses, same auth header (X-API-Key).
For each test query: run across selected modes, print answer + diff.

Run:
    venv/Scripts/python.exe eval_phase_g.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# Try loading from local eval/.env first, then fallback to root .env
eval_env = ROOT / "eval" / ".env"
if eval_env.exists():
    load_dotenv(eval_env)
else:
    load_dotenv(ROOT / ".env")

import httpx


API_BASE = os.environ.get("LIGHTRAG_API_BASE", "http://localhost:9621")
API_KEY = os.environ.get("LIGHTRAG_API_KEY", "")  # empty if auth disabled
HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["X-API-Key"] = API_KEY


# Test queries: (question, modes_to_test, expected_keywords_in_answer)
QUERIES = [
    # --- 5 semantic ---
    {
        "q": "Laravel hoạt động dựa trên kiến trúc nào?",
        "modes": ["naive", "hybrid"],
        "expect_kw": ["MVC"],
    },
    {
        "q": "Thuật toán Dijkstra dùng cấu trúc dữ liệu gì?",
        "modes": ["naive", "hybrid"],
        "expect_kw": ["priority queue", "hàng đợi", "heap"],
    },
    {
        "q": "Kiến trúc Von Neumann gồm những thành phần chính nào?",
        "modes": ["naive", "hybrid"],
        "expect_kw": ["CPU", "memory", "bộ nhớ", "I/O"],
    },
    {
        "q": "MVC pattern có những thành phần nào?",
        "modes": ["naive", "hybrid"],
        "expect_kw": ["Model", "View", "Controller"],
    },
    {
        "q": "Cache là gì và dùng để làm gì?",
        "modes": ["naive", "hybrid"],
        "expect_kw": ["bộ nhớ đệm", "tốc độ", "CPU"],
    },
    # --- 2 stat (bibliography via Neo4j Text2Cypher) ---
    {
        "q": "Có bao nhiêu Document trong cơ sở dữ liệu?",
        "modes": ["stat"],
        "expect_kw": ["5000", "Document"],
    },
    {
        "q": "Liệt kê 5 ngành (Major) bất kỳ.",
        "modes": ["stat"],
        "expect_kw": ["1.", "2.", "3."],
    },
]


async def run_query(client: httpx.AsyncClient, query: str, mode: str) -> dict:
    payload = {"query": query, "mode": mode, "stream": False}
    t0 = time.time()
    try:
        resp = await client.post(
            f"{API_BASE}/query",
            json=payload,
            headers=HEADERS,
            timeout=120.0,
        )
        elapsed = time.time() - t0
        if resp.status_code != 200:
            return {"ok": False, "err": f"HTTP {resp.status_code}: {resp.text[:200]}", "elapsed": elapsed}
        data = resp.json()
        return {
            "ok": True,
            "answer": data.get("response", ""),
            "elapsed": elapsed,
        }
    except Exception as e:
        return {"ok": False, "err": str(e)[:200], "elapsed": time.time() - t0}


def score_keywords(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    answer_low = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_low)
    return hits / len(keywords)


async def main():
    print(f"[eval] API base: {API_BASE}")
    print(f"[eval] auth: {'X-API-Key set' if API_KEY else 'no auth'}")
    print(f"[eval] queries: {len(QUERIES)}\n")

    summary = []
    async with httpx.AsyncClient() as client:
        for i, item in enumerate(QUERIES, 1):
            print(f"=" * 70)
            print(f"[{i}/{len(QUERIES)}] {item['q']}")
            print(f"  expect keywords: {item['expect_kw']}")
            for mode in item["modes"]:
                print(f"\n  --- mode={mode} ---")
                result = await run_query(client, item["q"], mode)
                if not result["ok"]:
                    print(f"  ERROR: {result['err']}")
                    summary.append((i, mode, "ERROR", 0.0, result["elapsed"]))
                    continue
                ans = result["answer"]
                kw_score = score_keywords(ans, item["expect_kw"])
                preview = ans[:300].replace("\n", " ")
                print(f"  latency: {result['elapsed']:.2f}s")
                print(f"  keyword recall: {kw_score:.2f}")
                print(f"  answer: {preview}{'...' if len(ans) > 300 else ''}")
                summary.append((i, mode, "OK", kw_score, result["elapsed"]))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Q#':<4}{'Mode':<10}{'Status':<10}{'Recall':<10}{'Latency':<10}")
    print("-" * 50)
    for q, mode, status, recall, lat in summary:
        print(f"{q:<4}{mode:<10}{status:<10}{recall:<10.2f}{lat:<10.2f}")


if __name__ == "__main__":
    asyncio.run(main())
