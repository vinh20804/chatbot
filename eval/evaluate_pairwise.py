"""LLM pairwise judge — compare 5 modes' responses per query.

Follows LightRAG/PathRAG paper methodology (Option F):
  - Show ONLY query + 5 responses (NO chunks, NO reference)
  - Per-type rubrics tailored to query characteristics
  - Anonymized labels A-E to avoid mode-name bias
  - 5-way ranking (1=best, 5=worst); Borda points = 6 - rank

Output: results_pairwise.json with per-query ranks + aggregate Borda scores.

Usage:
  python evaluate_pairwise.py --eval results_eval.json --out results_pairwise.json
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


_env_failover = os.getenv("ROUTER_FAILOVER_MODELS", "")
JUDGE_MODELS = (
    [m.strip() for m in _env_failover.split(",") if m.strip()]
    if _env_failover
    else ["gh/gpt-4o", "gc/gemini-2.5-flash"]
)

ROUTER_HOST = os.getenv("ROUTER_HOST", "http://localhost:20128/v1")
ROUTER_KEY = os.getenv("ROUTER_API_KEY", "dummy")
CONCURRENCY = int(os.getenv("PAIRWISE_CONCURRENCY", "5"))
TIMEOUT = 60
MODES_ORDER = ["bm25", "naive", "hybrid", "mix", "graph"]


# Per-type rubrics adapted from LightRAG/PathRAG papers.
# Each rubric: list of (name, definition) pairs - judge ranks based on these.
TYPE_RUBRICS = {
    "factoid": [
        ("Accuracy", "Câu trả lời có đúng về mặt nội dung không? Có chứa fact sai hay nhầm khái niệm không?"),
        ("Conciseness", "Trả lời gọn, đúng trọng tâm câu hỏi, không lan man dài dòng không liên quan."),
        ("Relevance", "Có thực sự trả lời câu hỏi factoid được hỏi không, hay đi lệch?"),
    ],
    "relational": [
        ("Relationship Clarity", "Có giải thích RÕ MỐI QUAN HỆ giữa các đối tượng không (không chỉ định nghĩa từng đối tượng riêng lẻ)?"),
        ("Logicality", "Lập luận có logic, có liên kết nguyên nhân-kết quả hay luồng chuyển tiếp hợp lý không?"),
        ("Comprehensiveness", "Có cover đủ các chiều của mối quan hệ (2 chiều, vai trò mỗi bên, tương tác)?"),
        ("Accuracy", "Mối quan hệ được mô tả có đúng không, hay có sai sót về fact?"),
    ],
    "broad": [
        ("Comprehensiveness", "Có cover ĐỦ RỘNG các khía cạnh quan trọng của câu hỏi không?"),
        ("Diversity", "Có nhiều góc nhìn, ví dụ, hay khía cạnh khác nhau không (không chỉ 1 chiều)?"),
        ("Coherence", "Có cấu trúc mạch lạc, các phần liên kết logic, đọc dễ hiểu không?"),
        ("Empowerment", "Sau khi đọc, người dùng có hiểu sâu vấn đề và đưa ra phán đoán được không?"),
    ],
    "aggregate": [
        ("Completeness", "Có liệt kê ĐỦ các items cần thiết để trả lời câu hỏi không?"),
        ("Accuracy", "Các items được liệt kê có đúng/hợp lệ không, có item sai không?"),
        ("Relevance", "Các items có thực sự thuộc danh sách được hỏi không, hay có item lạc đề?"),
    ],
    "multihop": [
        ("Cross-Domain Bridging", "Câu trả lời có THỰC SỰ kết nối được 2 lĩnh vực không? Hay chỉ trả lời 1 lĩnh vực rồi đề cập qua loa lĩnh vực kia?"),
        ("Accuracy", "Nội dung từ cả hai lĩnh vực có đúng không? Không nhầm khái niệm giữa hai môn học."),
        ("Reasoning Chain", "Có lý luận rõ ràng về MỐI LIÊN HỆ giữa hai khái niệm không (không chỉ liệt kê song song)?"),
        ("Depth", "Câu trả lời có đủ sâu để hiểu cả hai lĩnh vực cùng lúc không, hay quá nông?"),
    ],
    "multihop_2hop": [
        ("Connection Discovery", "Câu trả lời có TÌM RA được mối liên hệ gián tiếp giữa 2 khái niệm không (khi chúng không xuất hiện cùng tài liệu)? Hay bỏ cuộc / nói không liên quan?"),
        ("Reasoning Chain", "Có lần theo được chuỗi suy luận trung gian (bắc cầu qua khái niệm thứ ba) không, hay chỉ chắp vá võ đoán?"),
        ("Accuracy", "Các fact từ cả hai lĩnh vực có đúng không? Mối liên hệ tìm ra có hợp lý hay bịa đặt?"),
        ("Groundedness", "Câu trả lời có dựa trên thông tin thực tế không, hay hallucinate mối quan hệ không tồn tại để lấp đầy?"),
    ],
}


def clean_response(text: str) -> str:
    """Strip image markers + references for fair compare."""
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[IMG_[^\]]+\]|IMG_\w+", " ", text)
    text = re.sub(r"\n#{0,3}\s*References?\s*\n(?:.*?\n)*", "\n", text,
                  flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\[\d+\]|\[reference_id:\s*\d+\]", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def build_prompt(query: str, query_type: str, responses_by_label: dict[str, str]) -> str:
    """Build prompt with per-type rubric. Labels A-E are randomized mode assignment."""
    rubric = TYPE_RUBRICS.get(query_type, TYPE_RUBRICS["factoid"])
    rubric_text = "\n".join(f"{i+1}. **{name}**: {defn}" for i, (name, defn) in enumerate(rubric))

    blocks = []
    for label in "ABCDE":
        resp = clean_response(responses_by_label.get(label, ""))[:3000]
        blocks.append(f"### Response {label}\n{resp or '(empty)'}\n")

    return f"""Bạn là chuyên gia đánh giá hệ thống RAG cho domain giáo dục đại học (chương trình CNTT - ĐH Thủy Lợi).

Đọc câu hỏi và 5 câu trả lời (A, B, C, D, E). Xếp hạng chất lượng từ 1 (TỐT NHẤT) đến 5 (KÉM NHẤT), không trùng hạng.

### Loại câu hỏi: {query_type}

### Tiêu chí đánh giá ({len(rubric)} tiêu chí)
{rubric_text}

### Hướng dẫn
- Đánh giá thuần dựa trên nội dung trả lời, KHÔNG xem chunks hay reference (không có).
- Không thiên vị câu trả lời dài. Câu ngắn nhưng chất lượng vẫn có thể xếp cao hơn.
- Cân nhắc TỔNG HỢP các tiêu chí trên để ra ranking tổng thể.
- Không cho phép trùng hạng. Mỗi label có 1 rank duy nhất từ 1-5.

---

### Câu hỏi
{query}

---

{''.join(blocks)}

---

Trả về JSON duy nhất, không bao quanh bằng markdown code fence:
{{
  "ranking": {{"A": <rank>, "B": <rank>, "C": <rank>, "D": <rank>, "E": <rank>}},
  "reason": "1-2 câu giải thích ngắn gọn lý do chọn winner"
}}"""


async def judge_one(client, model: str, query: str, query_type: str, responses_by_label: dict):
    """Single judge call. Returns (parsed_ranking, error)."""
    prompt = build_prompt(query, query_type, responses_by_label)
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            timeout=TIMEOUT,
        )
        text = resp.choices[0].message.content or ""
        # Extract JSON robustly
        m = re.search(r"\{[^{}]*\"ranking\"[^{}]*\{[^{}]+\}[^{}]*\}", text, re.DOTALL)
        if not m:
            m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None, "no JSON in response"
        data = json.loads(m.group(0))
        ranking = data.get("ranking", {})
        if set(ranking.keys()) != set("ABCDE"):
            return None, f"incomplete ranking keys: {list(ranking.keys())}"
        ranks = {k: int(v) for k, v in ranking.items()}
        if sorted(ranks.values()) != [1, 2, 3, 4, 5]:
            return None, f"invalid rank values: {sorted(ranks.values())}"
        return {"ranks_by_label": ranks, "reason": data.get("reason", "")}, None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:120]}"


async def score_with_failover(clients_by_model, query, query_type, responses_by_label, entry_idx):
    """Round-robin + failover across judge models."""
    n = len(JUDGE_MODELS)
    last_err = None
    for k in range(n):
        model = JUDGE_MODELS[(entry_idx + k) % n]
        client = clients_by_model[model]
        result, err = await judge_one(client, model, query, query_type, responses_by_label)
        if result is not None:
            return result, model, None
        last_err = err
    return None, None, last_err


def group_by_query(entries):
    """Group entries by query_id. Returns {query_id: {mode: entry}}."""
    grouped = defaultdict(dict)
    for e in entries:
        grouped[e["query_id"]][e["mode"]] = e
    return grouped


async def run_pairwise(eval_path: str, out_path: str, seed: int = 42, resume: bool = False):
    from openai import AsyncOpenAI

    data = json.load(open(eval_path, encoding="utf-8"))
    entries = data["results"]
    grouped = group_by_query(entries)
    print(f"[LOAD] {len(grouped)} unique queries × 5 modes from {eval_path}")

    # Resume: load existing judgements, skip already-done qids
    existing_results = []
    skip_qids: set[str] = set()
    if resume and Path(out_path).exists():
        prev = json.load(open(out_path, encoding="utf-8"))
        existing_results = [r for r in prev.get("results", []) if "ranks" in r]
        skip_qids = {r["query_id"] for r in existing_results}
        print(f"[RESUME] Loaded {len(existing_results)} existing judgements, skipping {len(skip_qids)} qids")

    clients_by_model = {
        m: AsyncOpenAI(base_url=ROUTER_HOST, api_key=ROUTER_KEY, timeout=TIMEOUT)
        for m in JUDGE_MODELS
    }
    print(f"[POOL] {JUDGE_MODELS} | concurrency={CONCURRENCY}")
    print(f"[RUBRICS] types: {list(TYPE_RUBRICS.keys())}")

    sem = asyncio.Semaphore(CONCURRENCY)
    qids = sorted(qid for qid in grouped.keys() if qid not in skip_qids)
    total = len(qids)
    rng = random.Random(seed)

    async def judge_query(idx, qid):
        modes_dict = grouped[qid]
        if len(modes_dict) < 5:
            print(f"  [{idx+1}/{total}] {qid} SKIP (only {len(modes_dict)} modes)")
            return None
        query_text = modes_dict[MODES_ORDER[0]]["query"]
        query_type = modes_dict[MODES_ORDER[0]].get("type", "factoid")

        # Randomize label → mode mapping (anti-position-bias)
        shuffled_modes = list(MODES_ORDER)
        rng_local = random.Random(seed + idx)
        rng_local.shuffle(shuffled_modes)
        label_to_mode = dict(zip("ABCDE", shuffled_modes))

        responses_by_label = {
            label: modes_dict[mode]["response"]
            for label, mode in label_to_mode.items()
        }

        async with sem:
            res, model_used, err = await score_with_failover(
                clients_by_model, query_text, query_type, responses_by_label, idx
            )

        if res is None:
            print(f"  [{idx+1}/{total}] {qid} FAIL: {err}")
            return {"query_id": qid, "type": query_type, "error": err}

        # Map label-based ranks back to mode-based ranks
        mode_ranks = {label_to_mode[lbl]: rk for lbl, rk in res["ranks_by_label"].items()}
        winner = min(mode_ranks, key=mode_ranks.get)
        print(f"  [{idx+1}/{total}] {qid} ({query_type}) winner={winner} via={model_used}")
        return {
            "query_id": qid,
            "type": query_type,
            "label_to_mode": label_to_mode,
            "ranks": mode_ranks,
            "reason": res["reason"],
            "judge_model": model_used,
        }

    tasks = [judge_query(i, qid) for i, qid in enumerate(qids)]
    raw_results = await asyncio.gather(*tasks)
    new_results = [r for r in raw_results if r is not None]
    results = existing_results + new_results
    if existing_results:
        print(f"[RESUME] +{len(new_results)} new, {len(existing_results)} carried over → {len(results)} total")

    # Aggregate: Borda count + win count + mean rank
    by_type_mode_borda = defaultdict(lambda: defaultdict(list))
    win_count = defaultdict(lambda: defaultdict(int))
    overall_borda = defaultdict(list)
    overall_wins = defaultdict(int)

    for r in results:
        if "ranks" not in r:
            continue
        t = r["type"]
        ranks = r["ranks"]
        winner = min(ranks, key=ranks.get)
        win_count[t][winner] += 1
        overall_wins[winner] += 1
        for mode, rk in ranks.items():
            borda = 6 - rk  # rank 1 → 5 points, rank 5 → 1 point
            by_type_mode_borda[t][mode].append(borda)
            overall_borda[mode].append(borda)

    def aggregate_borda(borda_lists):
        return {
            m: {
                "mean_borda": round(sum(s)/len(s), 3),
                "mean_rank": round(6 - sum(s)/len(s), 3),
                "n": len(s),
            }
            for m, s in borda_lists.items() if s
        }

    aggregate = {
        "by_type": {
            t: {
                "win_count": dict(win_count[t]),
                "borda": aggregate_borda(by_type_mode_borda[t]),
            }
            for t in by_type_mode_borda
        },
        "overall": {
            "win_count": dict(overall_wins),
            "borda": aggregate_borda(overall_borda),
        },
        "total_queries": len(results),
    }

    print(f"\n=== Pairwise Results ({len(results)} queries) ===")
    for t, agg in aggregate["by_type"].items():
        print(f"\n--- {t} ---")
        print(f"  Win count: {agg['win_count']}")
        print(f"  Borda (higher=better):")
        for m, info in sorted(agg["borda"].items(), key=lambda x: -x[1]["mean_borda"]):
            print(f"    {m:<8}: mean_borda={info['mean_borda']} | mean_rank={info['mean_rank']} | n={info['n']}")

    print(f"\n=== Overall ===")
    print(f"  Win count: {aggregate['overall']['win_count']}")
    for m, info in sorted(aggregate["overall"]["borda"].items(), key=lambda x: -x[1]["mean_borda"]):
        print(f"    {m:<8}: mean_borda={info['mean_borda']} | mean_rank={info['mean_rank']} | n={info['n']}")

    out = {"aggregate": aggregate, "results": results}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, help="results_raw.json or results_eval.json path")
    parser.add_argument("--out", required=True, help="Output results_pairwise.json")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for label shuffling")
    parser.add_argument("--resume", action="store_true", help="Skip already-judged qids in existing --out file")
    args = parser.parse_args()

    global CONCURRENCY
    if args.concurrency:
        CONCURRENCY = args.concurrency

    asyncio.run(run_pairwise(args.eval, args.out, args.seed, resume=args.resume))


if __name__ == "__main__":
    main()
