"""Retry only failed entries (score=None) from previous RAGAS run.

Usage:
  python retry_failed_ragas.py --metric faithfulness
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv


def clean_response(text: str) -> str:
    """Strip image markers + references."""
    if not text:
        return ""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[IMG_[^\]]+\]|IMG_\w+", " ", text)
    text = re.sub(
        r"\n#{0,3}\s*References?\s*\n(?:.*?\n)*",
        "\n",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[reference_id:\s*\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_chunks(chunks):
    return [clean_response(c) for c in chunks if c and c.strip()]

load_dotenv(Path(__file__).parent.parent / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).parent
RESULTS_FILE = HERE / "results_eval.json"
QUERIES_FILE = HERE / "benchmark_queries.json"
CHUNKS_FILE = HERE / "results_chunks.json"
RAGAS_FILE = HERE / "results_ragas.json"


def load_all(queries_path=None, eval_path=None, chunks_path=None, ragas_path=None):
    qp = Path(queries_path) if queries_path else QUERIES_FILE
    ep = Path(eval_path) if eval_path else RESULTS_FILE
    cp = Path(chunks_path) if chunks_path else CHUNKS_FILE
    rp = Path(ragas_path) if ragas_path else RAGAS_FILE
    with open(qp, encoding="utf-8") as f:
        queries = {q["id"]: q for q in json.load(f)["queries"]}
    with open(ep, encoding="utf-8") as f:
        entries = json.load(f)["results"]
    with open(cp, encoding="utf-8") as f:
        chunks_map = {(c["query_id"], c["mode"]): c["chunks"] for c in json.load(f)}
    with open(rp, encoding="utf-8") as f:
        ragas = json.load(f)
    return queries, entries, chunks_map, ragas


JUDGE_FAILOVER_MODELS = [
    "gh/gpt-4o-mini",
    "gc/gemini-2.5-flash",
    "kc/stepfun/step-3.5-flash:free",
]


def make_judge_llm_single(model):
    from langchain_openai import ChatOpenAI
    base_url = os.getenv("ROUTER_HOST", "http://localhost:20128/v1")
    api_key = os.getenv("ROUTER_API_KEY", "")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0.0, timeout=180)


def make_judge_llm(model=None):
    """Single LLM (no built-in fallback — handled at retry loop level)."""
    return make_judge_llm_single(model or JUDGE_FAILOVER_MODELS[0])


def truncate_chunks(chunks, max_chars=15000):
    """Truncate chunks to avoid prompt token overflow."""
    total = 0
    out = []
    for c in chunks:
        if total + len(c) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                out.append(c[:remaining])
            break
        out.append(c)
        total += len(c)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True, choices=["faithfulness", "context_precision"])
    parser.add_argument("--throttle", type=float, default=0.5)
    parser.add_argument("--model", default=None, help="Override judge model")
    parser.add_argument("--max-chunk-chars", type=int, default=12000, help="Truncate chunks to N chars")
    # Path overrides
    parser.add_argument("--queries", default=None, help="Queries JSON path")
    parser.add_argument("--eval", default=None, help="results_eval.json path")
    parser.add_argument("--chunks", default=None, help="results_chunks.json path")
    parser.add_argument("--ragas", default=None, help="results_ragas.json path (read+write)")
    args = parser.parse_args()

    global RAGAS_FILE
    if args.ragas:
        RAGAS_FILE = Path(args.ragas)

    queries, entries, chunks_map, ragas = load_all(
        queries_path=args.queries,
        eval_path=args.eval,
        chunks_path=args.chunks,
        ragas_path=args.ragas,
    )
    if args.metric not in ragas:
        print(f"[ERR] {args.metric} not in {RAGAS_FILE}. Run full first.")
        sys.exit(1)

    old_scores = ragas[args.metric]["scores"]
    failed_idx = [s["entry_idx"] for s in old_scores if s["score"] is None]
    print(f"[RETRY] Found {len(failed_idx)} failed entries to retry")
    if not failed_idx:
        print("[DONE] Nothing to retry")
        return

    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    if args.metric == "faithfulness":
        from ragas.metrics import Faithfulness as Metric
    else:
        from ragas.metrics import LLMContextPrecisionWithReference as Metric

    # Build one metric instance per model — try them in order on error
    models_to_try = [args.model] if args.model else JUDGE_FAILOVER_MODELS
    metrics_pool = [(m, Metric(llm=LangchainLLMWrapper(make_judge_llm_single(m)))) for m in models_to_try]
    print(f"[POOL] Models: {[m for m, _ in metrics_pool]}")

    new_scores = list(old_scores)
    for i, idx in enumerate(failed_idx):
        e = entries[idx]
        chunks = chunks_map.get((e["query_id"], e["mode"]), [])
        if not chunks:
            continue
        # Truncate to avoid token overflow
        chunks_trunc = truncate_chunks(chunks, args.max_chunk_chars)

        cleaned_chunks = clean_chunks(chunks_trunc)
        cleaned_response = clean_response(e["response"])
        if args.metric == "faithfulness":
            sample = SingleTurnSample(
                user_input=e["query"],
                response=cleaned_response,
                retrieved_contexts=cleaned_chunks,
            )
        else:
            reference = queries[e["query_id"]].get("reference", "")
            sample = SingleTurnSample(
                user_input=e["query"],
                response=cleaned_response,
                reference=reference,
                retrieved_contexts=cleaned_chunks,
            )

        # Try each model in pool until success
        score_result = None
        last_err = None
        for model_name, metric in metrics_pool:
            try:
                score = asyncio.run(metric.single_turn_ascore(sample))
                score_result = (float(score), model_name)
                break
            except Exception as ex:
                last_err = ex
                continue

        if score_result:
            score_val, used_model = score_result
            new_scores[idx] = {"entry_idx": idx, "score": score_val, "reason": None, "judge_model": used_model}
            print(f"  [{i+1}/{len(failed_idx)}] idx={idx} {e['query_id']}/{e['mode']} score={score_val:.3f} via={used_model}")
        else:
            new_scores[idx] = {"entry_idx": idx, "score": None, "reason": f"{type(last_err).__name__}: {str(last_err)[:120]}"}
            print(f"  [{i+1}/{len(failed_idx)}] idx={idx} {e['query_id']}/{e['mode']} ALL_FAIL: {last_err}")
        time.sleep(args.throttle)

    # Re-aggregate
    from collections import defaultdict
    by_mode = defaultdict(list)
    by_type_mode = defaultdict(lambda: defaultdict(list))
    for entry, sc in zip(entries, new_scores):
        if sc["score"] is None:
            continue
        by_mode[entry["mode"]].append(sc["score"])
        by_type_mode[entry["type"]][entry["mode"]].append(sc["score"])

    print(f"\n=== {args.metric} by Mode (after retry) ===")
    for mode in ["bm25", "naive", "hybrid", "mix", "graph"]:
        sc = by_mode.get(mode, [])
        if sc:
            print(f"  {mode:<8}: n={len(sc):<3} mean={sum(sc)/len(sc):.4f}")

    aggr = {
        "by_mode": {m: sum(s) / len(s) for m, s in by_mode.items() if s},
        "by_type_mode": {t: {m: sum(s) / len(s) for m, s in mm.items() if s} for t, mm in by_type_mode.items()},
    }
    ragas[args.metric] = {"scores": new_scores, "aggregate": aggr}
    with open(RAGAS_FILE, "w", encoding="utf-8") as f:
        json.dump(ragas, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {RAGAS_FILE}")


if __name__ == "__main__":
    main()
