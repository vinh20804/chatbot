"""Evaluate benchmark with RAGAS metrics.

Metrics:
  - answer_relevancy   (only needs Q + A)
  - answer_correctness (needs Q + A + reference)
  - faithfulness       (needs Q + A + retrieved_chunks)  ← run fetch_chunks.py first
  - context_precision  (needs Q + A + retrieved_chunks + reference)

Judge: gemini-2.5-flash (free tier 60 RPM)
Embeddings: sentence-transformers multilingual-MiniLM-L12-v2 (local, no API)

Chunks file (optional): results_chunks.json
  Format: [{"query_id": "...", "mode": "...", "chunks": ["text1", "text2", ...]}, ...]
  Generate via: python fetch_chunks.py

Usage:
  python evaluate_ragas.py --metric answer_relevancy
  python evaluate_ragas.py --metric answer_correctness
  python evaluate_ragas.py --metric faithfulness
  python evaluate_ragas.py --metric context_precision
  python evaluate_ragas.py --metric all
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv


def clean_response(text: str) -> str:
    """Strip image markers + references from response — fair compare across modes."""
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


def clean_chunks(chunks: list) -> list:
    """Strip image markers from chunks."""
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


def load_queries(path=None):
    p = Path(path) if path else QUERIES_FILE
    with open(p, encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)["queries"]}


def load_results(path=None):
    p = Path(path) if path else RESULTS_FILE
    if not p.exists():
        print(f"[ERR] {p} missing. Run evaluate_benchmark.py first.")
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_chunks(path=None):
    """Load chunks lookup {(query_id, mode): [chunk_text, ...]}."""
    p = Path(path) if path else CHUNKS_FILE
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {(c["query_id"], c["mode"]): c["chunks"] for c in data}


_env_failover = os.getenv("ROUTER_FAILOVER_MODELS", "")
JUDGE_FAILOVER_MODELS = (
    [m.strip() for m in _env_failover.split(",") if m.strip()]
    if _env_failover
    else [
        "gh/gpt-4o-mini",
        "gc/gemini-2.5-flash",
        "kc/stepfun/step-3.5-flash:free",
    ]
)


def make_judge_llm(model=None):
    """Use 9router proxy (OpenAI-compatible) with failover."""
    from langchain_openai import ChatOpenAI
    # ROUTER_* is for modules that need smart models via 9router proxy,
    # independent of the server's LLM_BINDING_* (which may point to Ollama for benchmark).
    base_url = os.getenv("ROUTER_HOST", "http://localhost:20128/v1")
    api_key = os.getenv("ROUTER_API_KEY", "")
    m = model or os.getenv("JUDGE_MODEL", JUDGE_FAILOVER_MODELS[0])
    return ChatOpenAI(
        model=m,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
        timeout=30,
    )


class FailoverLLM:
    """Wrap multiple ChatOpenAI instances. Try primary, fallback on 4xx/5xx."""

    def __init__(self, models=None):
        models = models or JUDGE_FAILOVER_MODELS
        self.llms = [make_judge_llm(m) for m in models]
        self.model_names = list(models)
        self._failure_count = {m: 0 for m in self.model_names}

    async def ainvoke(self, *args, **kwargs):
        last_err = None
        for i, llm in enumerate(self.llms):
            try:
                result = await llm.ainvoke(*args, **kwargs)
                return result
            except Exception as ex:
                last_err = ex
                self._failure_count[self.model_names[i]] += 1
                continue
        raise last_err

    def invoke(self, *args, **kwargs):
        last_err = None
        for i, llm in enumerate(self.llms):
            try:
                return self.llms[i].invoke(*args, **kwargs)
            except Exception as ex:
                last_err = ex
                self._failure_count[self.model_names[i]] += 1
                continue
        raise last_err

    def __getattr__(self, name):
        # Delegate any attribute to primary LLM (for compat with LangChain wrappers)
        return getattr(self.llms[0], name)


def make_judge_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


def get_wrappers(need_embeddings: bool = True):
    """Get RAGAS LLM (single primary model) + optional embeddings."""
    from ragas.llms import LangchainLLMWrapper
    llm = LangchainLLMWrapper(make_judge_llm())
    if not need_embeddings:
        return llm, None
    from ragas.embeddings import LangchainEmbeddingsWrapper
    emb = LangchainEmbeddingsWrapper(make_judge_embeddings())
    return llm, emb


def get_metric_pool(metric_class, need_embeddings: bool = False):
    """Build N metric instances, one per failover model. Try in order on error.

    Returns list of (model_name, metric_instance).
    """
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    pool = []
    emb = None
    if need_embeddings:
        emb = LangchainEmbeddingsWrapper(make_judge_embeddings())
    for m in JUDGE_FAILOVER_MODELS:
        llm = LangchainLLMWrapper(make_judge_llm(m))
        if need_embeddings:
            metric = metric_class(llm=llm, embeddings=emb)
        else:
            metric = metric_class(llm=llm)
        pool.append((m, metric))
    return pool


CONCURRENCY = int(os.getenv("RAGAS_CONCURRENCY", "5"))


async def score_async(metric_pool, sample, entry_idx: int):
    """Round-robin model assignment + async fallback. Returns (score, model_used, error)."""
    n = len(metric_pool)
    last_err = None
    for k in range(n):
        model_name, metric = metric_pool[(entry_idx + k) % n]
        try:
            score = await metric.single_turn_ascore(sample)
            return float(score), model_name, None
        except Exception as ex:
            last_err = ex
            continue
    return None, None, f"{type(last_err).__name__}: {str(last_err)[:120]}"


async def run_parallel(entries, build_sample_fn, metric_pool, label: str):
    """Run metric for all entries in parallel with semaphore-limited concurrency."""
    sem = asyncio.Semaphore(CONCURRENCY)
    total = len(entries)
    results = [None] * total

    async def score_one(i, e):
        sample = build_sample_fn(i, e)
        if sample is None:
            results[i] = {"entry_idx": i, "score": None, "reason": "empty"}
            return
        async with sem:
            sc, used, err = await score_async(metric_pool, sample, i)
        results[i] = {"entry_idx": i, "score": sc, "reason": err, "judge_model": used}
        msg = f"score={sc:.3f} via={used}" if sc is not None else f"ALL_FAIL: {err}"
        print(f"  [{i+1}/{total}] {e['query_id']}/{e['mode']} {msg}")

    await asyncio.gather(*[score_one(i, e) for i, e in enumerate(entries)])
    return results


def run_answer_relevancy(entries, queries, chunks_map, throttle):
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import ResponseRelevancy

    metric_pool = get_metric_pool(ResponseRelevancy, need_embeddings=True)
    print(f"[POOL] {[m for m, _ in metric_pool]} | concurrency={CONCURRENCY}")

    def build(i, e):
        if e.get("error") or not e.get("response"):
            return None
        return SingleTurnSample(user_input=e["query"], response=clean_response(e["response"]))

    return asyncio.run(run_parallel(entries, build, metric_pool, "answer_relevancy"))


def run_answer_correctness(entries, queries, chunks_map, throttle):
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import AnswerCorrectness

    metric_pool = get_metric_pool(AnswerCorrectness, need_embeddings=True)
    print(f"[POOL] {[m for m, _ in metric_pool]} | concurrency={CONCURRENCY}")

    def build(i, e):
        if e.get("error") or not e.get("response"):
            return None
        return SingleTurnSample(
            user_input=e["query"],
            response=clean_response(e["response"]),
            reference=queries[e["query_id"]].get("reference", ""),
        )

    return asyncio.run(run_parallel(entries, build, metric_pool, "answer_correctness"))


def run_faithfulness(entries, queries, chunks_map, throttle):
    if chunks_map is None:
        print("[ERR] faithfulness needs retrieved chunks. Run fetch_chunks.py first.")
        sys.exit(1)

    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import Faithfulness

    metric_pool = get_metric_pool(Faithfulness, need_embeddings=False)
    print(f"[POOL] {[m for m, _ in metric_pool]} | concurrency={CONCURRENCY}")

    def build(i, e):
        if e.get("error") or not e.get("response"):
            return None
        chunks = chunks_map.get((e["query_id"], e["mode"]), [])
        if not chunks:
            return None
        return SingleTurnSample(
            user_input=e["query"],
            response=clean_response(e["response"]),
            retrieved_contexts=clean_chunks(chunks),
        )

    return asyncio.run(run_parallel(entries, build, metric_pool, "faithfulness"))


def run_context_precision(entries, queries, chunks_map, throttle):
    if chunks_map is None:
        print("[ERR] context_precision needs retrieved chunks. Run fetch_chunks.py first.")
        sys.exit(1)

    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import LLMContextPrecisionWithReference

    metric_pool = get_metric_pool(LLMContextPrecisionWithReference, need_embeddings=False)
    print(f"[POOL] {[m for m, _ in metric_pool]} | concurrency={CONCURRENCY}")

    def build(i, e):
        if e.get("error") or not e.get("response"):
            return None
        chunks = chunks_map.get((e["query_id"], e["mode"]), [])
        if not chunks:
            return None
        return SingleTurnSample(
            user_input=e["query"],
            response=clean_response(e["response"]),
            reference=queries[e["query_id"]].get("reference", ""),
            retrieved_contexts=clean_chunks(chunks),
        )

    return asyncio.run(run_parallel(entries, build, metric_pool, "context_precision"))


METRIC_FNS = {
    "answer_relevancy": run_answer_relevancy,
    "answer_correctness": run_answer_correctness,
    "faithfulness": run_faithfulness,
    "context_precision": run_context_precision,
}


def aggregate(entries, scores, metric_name):
    by_mode = defaultdict(list)
    by_type_mode = defaultdict(lambda: defaultdict(list))
    for entry, sc in zip(entries, scores):
        if sc["score"] is None:
            continue
        by_mode[entry["mode"]].append(sc["score"])
        by_type_mode[entry["type"]][entry["mode"]].append(sc["score"])

    print(f"\n=== {metric_name} by Mode ===")
    for mode in ["bm25", "naive", "hybrid", "mix", "graph"]:
        sc = by_mode.get(mode, [])
        if sc:
            mean = sum(sc) / len(sc)
            print(f"  {mode:<8}: n={len(sc):<3} mean={mean:.4f}")

    return {
        "by_mode": {m: sum(s) / len(s) for m, s in by_mode.items() if s},
        "by_type_mode": {t: {m: sum(s) / len(s) for m, s in mm.items() if s} for t, mm in by_type_mode.items()},
    }


def save_result(metric_name, scores, aggr):
    existing = {}
    if RAGAS_FILE.exists():
        with open(RAGAS_FILE, encoding="utf-8") as f:
            existing = json.load(f)
    existing[metric_name] = {"scores": scores, "aggregate": aggr}
    with open(RAGAS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {RAGAS_FILE} [{metric_name}]")


def main():
    global RAGAS_FILE, CONCURRENCY
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metric",
        default="faithfulness,answer_relevancy",
        help="Comma-separated metrics or 'all'. Default: faithfulness,answer_relevancy",
    )
    parser.add_argument("--concurrency", type=int, default=None,
                        help=f"Parallel workers (default: RAGAS_CONCURRENCY env or {CONCURRENCY})")
    parser.add_argument("--throttle", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true", help="Only retry entries with score=None from previous run")
    # Path overrides for per-type bench
    parser.add_argument("--queries", default=None, help="Queries JSON path")
    parser.add_argument("--eval", default=None, help="results_eval.json path")
    parser.add_argument("--chunks", default=None, help="results_chunks.json path")
    parser.add_argument("--out", default=None, help="results_ragas.json output path")
    args = parser.parse_args()

    if args.out:
        RAGAS_FILE = Path(args.out)

    queries = load_queries(args.queries)
    data = load_results(args.eval)
    entries = data["results"]
    if args.limit:
        entries = entries[: args.limit]

    chunks_map = load_chunks(args.chunks)
    if chunks_map:
        print(f"[CHUNKS] Loaded {len(chunks_map)} (query_id, mode) chunk sets")
    else:
        print("[CHUNKS] None — faithfulness/context_precision will fail. Run fetch_chunks.py.")

    if args.concurrency:
        CONCURRENCY = args.concurrency

    print(f"[LOAD] {len(entries)} entries")
    print(f"[CONCURRENCY] {CONCURRENCY}")

    if args.metric == "all":
        metrics_to_run = list(METRIC_FNS.keys())
    else:
        metrics_to_run = [m.strip() for m in args.metric.split(",") if m.strip()]
    for m in metrics_to_run:
        print(f"\n[METRIC] {m}")
        fn = METRIC_FNS[m]
        try:
            scores = fn(entries, queries, chunks_map, args.throttle)
            aggr = aggregate(entries, scores, m)
            save_result(m, scores, aggr)
        except SystemExit:
            print(f"[SKIP] {m} (missing prerequisite)")
            continue


if __name__ == "__main__":
    main()
