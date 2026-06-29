"""Manual eval: 10 query → run pipeline → dump JSON for human judgment.

Output: manual_eval_results.json — đọc và mark Y/N/partial từng câu.
"""
from __future__ import annotations

import asyncio
import json
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

API_BASE = "http://localhost:9621"

QUERIES = [
    # Corpus semantic — test naive vs hybrid
    {"id": 1, "q": "Laravel sử dụng Eloquent ORM để làm gì?", "modes": ["naive", "hybrid"]},
    {"id": 2, "q": "Cache L1 và L2 khác nhau thế nào?", "modes": ["naive", "hybrid"]},
    {"id": 3, "q": "Booth Algorithm hoạt động ra sao?", "modes": ["naive", "hybrid"]},
    {"id": 4, "q": "MVC gồm những thành phần nào trong Laravel?", "modes": ["naive", "hybrid"]},
    {"id": 5, "q": "Register trong CPU dùng để làm gì?", "modes": ["naive", "hybrid"]},
    # Stat — Neo4j bibliography
    {"id": 6, "q": "5 tác giả viết nhiều nhất là ai?", "modes": ["stat"]},
    {"id": 7, "q": "Có bao nhiêu thesis viết năm 2024?", "modes": ["stat"]},
    {"id": 8, "q": "Liệt kê 5 chủ đề phổ biến nhất", "modes": ["stat"]},
    {"id": 9, "q": "Đồ án ngành Công nghệ thông tin có bao nhiêu?", "modes": ["stat"]},
    # Edge case — multi-hop hybrid
    {"id": 10, "q": "Cache Replacement Algorithm dùng strategy gì?", "modes": ["hybrid"]},
]


async def run(client, q, mode):
    payload = {"query": q, "mode": mode, "stream": False}
    t0 = time.time()
    try:
        r = await client.post(f"{API_BASE}/query", json=payload, timeout=120.0)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"ok": False, "err": f"HTTP {r.status_code}: {r.text[:200]}", "elapsed": elapsed}
        return {"ok": True, "answer": r.json().get("response", ""), "elapsed": elapsed}
    except Exception as e:
        return {"ok": False, "err": str(e)[:200], "elapsed": time.time() - t0}


async def main():
    results = []
    async with httpx.AsyncClient() as client:
        for item in QUERIES:
            print(f"[{item['id']}/{len(QUERIES)}] {item['q']}")
            entry = {"id": item["id"], "query": item["q"], "results": {}}
            for mode in item["modes"]:
                print(f"  → mode={mode}", flush=True)
                r = await run(client, item["q"], mode)
                entry["results"][mode] = r
                if r.get("ok"):
                    print(f"    {r['elapsed']:.1f}s — {r['answer'][:200]}")
                else:
                    print(f"    ERR — {r.get('err','')}")
            results.append(entry)
    out = ROOT / "manual_eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
