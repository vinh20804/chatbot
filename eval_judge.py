~"""LLM-as-judge eval for RAG faithfulness + relevancy.

Tests 2 modes (naive, hybrid). Skip stat (programmatic match).
For each query × mode:
  1. Capture answer via /query
  2. Capture context (chunks) via /query/data
  3. Judge faithfulness: claims in answer supported by chunks?
  4. Judge relevancy: answer addresses query?
  5. Save row + print summary

Run:
    venv/Scripts/python.exe eval_judge.py
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

from lightrag.llm.openai import openai_complete_if_cache


API_BASE = os.environ.get("LIGHTRAG_API_BASE", "http://localhost:9621")
# ROUTER_* for smart-judge model via 9router (server's LLM may be Ollama for benchmark)
LLM_HOST = os.environ.get("ROUTER_HOST") or os.environ["LLM_BINDING_HOST"]
LLM_KEY = os.environ.get("ROUTER_API_KEY") or os.environ["LLM_BINDING_API_KEY"]
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gh/gpt-4o-mini")

# 5 common knowledge + 5 TLU-specific. Latter SHOULD have higher faithfulness
# since LLM cannot answer them from pretraining alone.
QUERIES = [
    # --- Common knowledge (LLM may leak from pretraining) ---
    "Laravel sử dụng Eloquent ORM để làm gì?",
    "Cache L1 và L2 khác nhau thế nào?",
    "Booth Algorithm hoạt động ra sao?",
    "MVC gồm những thành phần nào trong Laravel?",
    "Register trong CPU dùng để làm gì?",
    # --- TLU-specific (only in corpus, no pretraining) ---
    "Trong tài liệu TLU, Cache Replacement Algorithm nói về những thuật toán nào?",
    "Slide TLU dạy về MVC pattern dùng framework gì làm ví dụ?",
    "Trong slide TLU, các loại địa chỉ trong instruction là gì?",
    "Trong tài liệu TLU, Eloquent có những phương thức nào được giới thiệu?",
    "Slide TLU nêu các thành phần chính nào của một CPU?",
]


SYSTEM_FAITH = """Bạn là evaluator faithfulness cho RAG system.

Input bạn nhận:
- Query: câu hỏi user
- Answer: câu trả lời do RAG sinh ra
- Chunks: các đoạn text retrieved từ corpus

Task:
1. Tách Answer thành các CLAIM đơn (mỗi câu fact = 1 claim).
2. Mỗi claim, đánh giá:
   - SUPPORTED: claim có thông tin trực tiếp hoặc paraphrase trong Chunks
   - NOT_SUPPORTED: claim KHÔNG có trong Chunks (LLM bịa hoặc dùng pretraining)
3. Output JSON ONLY (không markdown, không giải thích thêm):
{
  "claims": [
    {"text": "claim text", "supported": true | false, "evidence": "trích chunks nếu supported, null nếu không"}
  ],
  "total_claims": <int>,
  "supported_claims": <int>,
  "faithfulness": <float 0-1>,
  "leak_warning": <bool — true nếu có claim NOT_SUPPORTED nhưng concrete factual (có thể từ pretraining)>
}

Strict: chỉ accept SUPPORTED nếu Chunks có thông tin tương ứng. Đừng giả định.
"""


SYSTEM_RELEVANCY = """Bạn là evaluator answer relevancy.

Input:
- Query: câu hỏi
- Answer: câu trả lời

Task: đánh giá Answer có trả lời đúng intent của Query không.

Score:
- 1.0 = Answer đầy đủ và đúng trọng tâm Query
- 0.7 = Đúng nhưng thiếu sót một số khía cạnh
- 0.5 = Partial — đề cập topic nhưng không đáp đúng câu hỏi cụ thể
- 0.0 = Off-topic, không trả lời

Output JSON ONLY:
{"relevancy": <float 0-1>, "reason": "<brief explanation>"}
"""


async def call_query(client: httpx.AsyncClient, q: str, mode: str) -> str:
    payload = {"query": q, "mode": mode, "stream": False}
    r = await client.post(f"{API_BASE}/query", json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json().get("response", "")


async def call_query_data(client: httpx.AsyncClient, q: str, mode: str) -> dict:
    payload = {"query": q, "mode": mode, "stream": False}
    r = await client.post(f"{API_BASE}/query/data", json=payload, timeout=120.0)
    r.raise_for_status()
    return r.json().get("data", {})


def serialize_chunks(data: dict, max_chunks: int = 10) -> str:
    """Pick top N chunks, format with index for judge."""
    chunks = data.get("chunks", [])[:max_chunks]
    if not chunks:
        return "(no chunks retrieved)"
    return "\n\n".join(
        f"[CHUNK {i + 1}]\n{c.get('content', '')}"
        for i, c in enumerate(chunks)
    )


async def judge(system: str, user: str) -> dict:
    """Call LLM judge with JSON response. Retry once on parse fail."""
    for attempt in range(2):
        try:
            resp = await openai_complete_if_cache(
                JUDGE_MODEL,
                user,
                system_prompt=system,
                api_key=LLM_KEY,
                base_url=LLM_HOST,
                response_format={"type": "json_object"},
            )
            return json.loads(resp)
        except json.JSONDecodeError as e:
            if attempt == 0:
                continue
            return {"_parse_error": str(e), "_raw": resp[:500] if resp else ""}
        except Exception as e:
            return {"_judge_error": str(e)[:200]}


async def judge_faithfulness(query: str, answer: str, chunks_text: str) -> dict:
    user = f"Query: {query}\n\nAnswer: {answer}\n\nChunks:\n{chunks_text}"
    return await judge(SYSTEM_FAITH, user)


async def judge_relevancy(query: str, answer: str) -> dict:
    user = f"Query: {query}\n\nAnswer: {answer}"
    return await judge(SYSTEM_RELEVANCY, user)


async def eval_one(client: httpx.AsyncClient, q: str, mode: str) -> dict:
    t0 = time.time()
    answer = await call_query(client, q, mode)
    data = await call_query_data(client, q, mode)
    chunks_text = serialize_chunks(data)

    faith = await judge_faithfulness(q, answer, chunks_text)
    relev = await judge_relevancy(q, answer)

    elapsed = time.time() - t0
    return {
        "query": q,
        "mode": mode,
        "elapsed_s": round(elapsed, 1),
        "answer_preview": answer[:200],
        "answer_full": answer,
        "chunks_count": len(data.get("chunks", [])),
        "context_preview": chunks_text[:300],
        "faithfulness": faith.get("faithfulness"),
        "supported_claims": faith.get("supported_claims"),
        "total_claims": faith.get("total_claims"),
        "leak_warning": faith.get("leak_warning"),
        "faith_full": faith,
        "relevancy": relev.get("relevancy"),
        "relev_reason": relev.get("reason"),
        "relev_full": relev,
    }


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'#':<3}{'Query':<55}{'Mode':<10}{'Faith':<8}{'Relev':<8}{'Leak':<6}{'Sec':<6}")
    print("-" * 100)
    for i, r in enumerate(results, 1):
        q_short = r["query"][:50] + ("…" if len(r["query"]) > 50 else "")
        f = r.get("faithfulness")
        v = r.get("relevancy")
        leak = "Y" if r.get("leak_warning") else " "
        f_str = f"{f:.2f}" if isinstance(f, (int, float)) else "?"
        v_str = f"{v:.2f}" if isinstance(v, (int, float)) else "?"
        print(f"{i:<3}{q_short:<55}{r['mode']:<10}{f_str:<8}{v_str:<8}{leak:<6}{r['elapsed_s']:<6}")
    print()

    # Aggregate per mode
    print("AGGREGATES PER MODE")
    print("-" * 60)
    for mode in ("naive", "hybrid"):
        subset = [r for r in results if r["mode"] == mode]
        if not subset:
            continue
        faiths = [r["faithfulness"] for r in subset if isinstance(r.get("faithfulness"), (int, float))]
        relevs = [r["relevancy"] for r in subset if isinstance(r.get("relevancy"), (int, float))]
        leaks = sum(1 for r in subset if r.get("leak_warning"))
        avg_f = sum(faiths) / len(faiths) if faiths else 0
        avg_r = sum(relevs) / len(relevs) if relevs else 0
        print(f"  {mode}: avg_faith={avg_f:.2f}  avg_relev={avg_r:.2f}  leak_warnings={leaks}/{len(subset)}")


async def main():
    print(f"[eval-judge] API base: {API_BASE}")
    print(f"[eval-judge] Judge model: {JUDGE_MODEL}")
    print(f"[eval-judge] Queries: {len(QUERIES)}, modes: naive + hybrid\n")

    results = []
    async with httpx.AsyncClient() as client:
        for i, q in enumerate(QUERIES, 1):
            for mode in ("naive", "hybrid"):
                print(f"[{i}/{len(QUERIES)}] mode={mode}: {q[:60]}…")
                try:
                    r = await eval_one(client, q, mode)
                    print(f"  faith={r.get('faithfulness')} relev={r.get('relevancy')} leak={r.get('leak_warning')} {r['elapsed_s']}s")
                except Exception as e:
                    err_str = str(e)[:200]
                    print(f"  ERROR: {err_str}")
                    r = {"query": q, "mode": mode, "error": err_str}
                results.append(r)

    out_path = ROOT / "eval_judge_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] saved {out_path}")

    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
