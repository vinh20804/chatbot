"""Evaluate benchmark responses with BERTScore.

Load results_raw.json + benchmark_queries.json → score each response
vs reference using BERTScore (xlm-roberta-large, lang=vi).

Outputs:
  - results_eval.json: per-entry P/R/F1
  - report.md: per-mode + per-type aggregate with bootstrap CI 95%

Usage:
  python evaluate_benchmark.py
  python evaluate_benchmark.py --bootstrap 1000  # CI resamples
  python evaluate_benchmark.py --model xlm-roberta-base  # faster, less accurate
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def clean_response(text: str) -> str:
    """Strip image markers + references section before scoring.

    Server injects ![](/images/xxx.png) inline + IMG_xxx markers + References list.
    These pollute BERTScore (markers don't exist in reference text).
    """
    if not text:
        return ""
    # Strip markdown image syntax
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    # Strip raw IMG_ identifiers
    text = re.sub(r"\[IMG_[^\]]+\]|IMG_\w+", " ", text)
    # Strip References section at end
    text = re.sub(
        r"\n#{0,3}\s*References?\s*\n(?:.*?\n)*",
        "\n",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    # Strip [N] reference numbers like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    # Strip [reference_id: N] markers injected by server
    text = re.sub(r"\[reference_id:\s*\d+\]", "", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

HERE = Path(__file__).parent
# Defaults (can override via CLI)
QUERIES_FILE = HERE / "benchmark_queries.json"
RESULTS_FILE = HERE / "results_raw.json"
EVAL_FILE = HERE / "results_eval.json"
REPORT_FILE = HERE / "report.md"


def load_queries(path=None):
    p = Path(path) if path else QUERIES_FILE
    with open(p, "r", encoding="utf-8") as f:
        return {q["id"]: q for q in json.load(f)["queries"]}


def load_results(path=None):
    p = Path(path) if path else RESULTS_FILE
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def bootstrap_ci(scores: list, n_resamples: int = 1000, ci: float = 0.95) -> tuple:
    """Return (mean, lower, upper) of bootstrap CI."""
    if not scores:
        return (0.0, 0.0, 0.0)
    if len(scores) == 1:
        return (scores[0], scores[0], scores[0])
    rng = random.Random(42)
    means = []
    n = len(scores)
    for _ in range(n_resamples):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1 - ci) / 2
    lo = means[int(alpha * n_resamples)]
    hi = means[int((1 - alpha) * n_resamples)]
    return (mean(scores), lo, hi)


def median_per_query_mode(entries: list) -> list:
    """If multi-run, collapse to median F1 per (query_id, mode)."""
    grouped = defaultdict(list)
    for e in entries:
        if e.get("error"):
            continue
        key = (e["query_id"], e["mode"])
        grouped[key].append(e)
    collapsed = []
    for (qid, mode), runs in grouped.items():
        if len(runs) == 1:
            collapsed.append(runs[0])
            continue
        # Median F1 across runs — pick the entry closest to median
        sorted_by_f1 = sorted(runs, key=lambda r: r["bertscore_f1"])
        mid = sorted_by_f1[len(sorted_by_f1) // 2]
        collapsed.append(mid)
    return collapsed


def score_with_bertscore(
    cands: list, refs: list, model_type: str, lang: str = "vi", device: str = None
) -> tuple:
    """Return (P_list, R_list, F1_list). Lazy import to avoid heavy load when only viewing."""
    from bert_score import score as bert_score
    import torch

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[BERTScore] model={model_type} lang={lang} device={device} | scoring {len(cands)} pairs..."
    )
    try:
        P, R, F1 = bert_score(
            cands,
            refs,
            model_type=model_type,
            lang=lang,
            device=device,
            verbose=False,
            rescale_with_baseline=False,  # raw scores, more interpretable
        )
    except Exception as e:
        if device == "cuda":
            print(f"\n[WARNING] BERTScore CUDA failed ({e}). Falling back to CPU...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            device = "cpu"
            print(f"[BERTScore] Retrying on device={device}...")
            P, R, F1 = bert_score(
                cands,
                refs,
                model_type=model_type,
                lang=lang,
                device=device,
                verbose=False,
                rescale_with_baseline=False,
            )
        else:
            raise e
    return P.tolist(), R.tolist(), F1.tolist()


def format_md_table(rows: list, headers: list) -> str:
    """Simple markdown table."""
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="xlm-roberta-large", help="BERTScore model")
    parser.add_argument("--lang", default="vi", help="Language code")
    parser.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap CI resamples")
    parser.add_argument("--skip-bertscore", action="store_true", help="Skip scoring")
    parser.add_argument("--device", default=None, help="Force torch device (e.g. cpu, cuda)")
    # CLI overrides for paths (used by run_all.sh per-type mode)
    parser.add_argument("--queries", default=None, help="Queries JSON path")
    parser.add_argument("--raw", default=None, help="results_raw.json path")
    parser.add_argument("--out-eval", default=None, help="Output results_eval.json path")
    parser.add_argument("--out-report", default=None, help="Output report.md path")
    args = parser.parse_args()

    # Override module-level defaults
    global EVAL_FILE, REPORT_FILE
    if args.out_eval:
        EVAL_FILE = Path(args.out_eval)
    if args.out_report:
        REPORT_FILE = Path(args.out_report)

    queries = load_queries(args.queries)
    raw = load_results(args.raw)
    entries = raw["results"]
    print(f"[LOAD] {len(entries)} raw entries | {len(queries)} queries in benchmark")

    if not args.skip_bertscore:
        # Build cand/ref lists, skip errors. Clean image markers + refs from response.
        valid = [e for e in entries if not e.get("error") and e["response"].strip()]
        cands = [clean_response(e["response"]) for e in valid]
        refs = [queries[e["query_id"]]["reference"] for e in valid]
        print(
            f"[VALID] {len(valid)}/{len(entries)} entries scoreable (non-empty, no error)"
        )

        P, R, F1 = score_with_bertscore(cands, refs, args.model, args.lang, args.device)
        for e, p, r, f1 in zip(valid, P, R, F1):
            e["bertscore_p"] = round(p, 4)
            e["bertscore_r"] = round(r, 4)
            e["bertscore_f1"] = round(f1, 4)

        # Errors get None scores
        for e in entries:
            if e.get("error") or not e["response"].strip():
                e["bertscore_p"] = None
                e["bertscore_r"] = None
                e["bertscore_f1"] = None

        eval_data = {
            "meta": {**raw["meta"], "bertscore_model": args.model, "lang": args.lang},
            "results": entries,
        }
        with open(EVAL_FILE, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] {EVAL_FILE}")
    else:
        with open(EVAL_FILE, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
        entries = eval_data["results"]

    # --- Aggregate ---
    # If multi-run, collapse to median per (query, mode) first
    collapsed = median_per_query_mode(entries)

    # Per-mode
    by_mode = defaultdict(list)
    for e in collapsed:
        if e.get("bertscore_f1") is not None:
            by_mode[e["mode"]].append(e["bertscore_f1"])

    mode_rows = []
    for mode in ["bm25", "naive", "hybrid", "mix", "graph"]:
        scores = by_mode.get(mode, [])
        if not scores:
            mode_rows.append([mode, 0, "—", "—", "—"])
            continue
        m, lo, hi = bootstrap_ci(scores, args.bootstrap)
        mode_rows.append(
            [
                mode,
                len(scores),
                f"{m:.4f}",
                f"[{lo:.4f}, {hi:.4f}]",
                f"{median(scores):.4f}",
            ]
        )

    # Per-type × mode
    by_type_mode = defaultdict(lambda: defaultdict(list))
    for e in collapsed:
        if e.get("bertscore_f1") is not None:
            by_type_mode[e["type"]][e["mode"]].append(e["bertscore_f1"])

    type_rows = []
    for typ in ["factoid", "relational", "broad", "aggregate"]:
        row = [typ]
        for mode in ["bm25", "naive", "hybrid", "mix", "graph"]:
            scores = by_type_mode[typ].get(mode, [])
            row.append(f"{mean(scores):.4f}" if scores else "—")
        type_rows.append(row)

    # Per-mode latency
    lat_by_mode = defaultdict(list)
    for e in collapsed:
        if e.get("latency_sec") is not None:
            lat_by_mode[e["mode"]].append(e["latency_sec"])
    lat_rows = []
    for mode in ["bm25", "naive", "hybrid", "mix", "graph"]:
        lats = lat_by_mode.get(mode, [])
        if lats:
            lat_rows.append(
                [
                    mode,
                    len(lats),
                    f"{mean(lats):.2f}s",
                    f"{median(lats):.2f}s",
                    f"{max(lats):.2f}s",
                ]
            )
        else:
            lat_rows.append([mode, 0, "—", "—", "—"])

    # Error count per mode
    err_by_mode = defaultdict(int)
    for e in entries:
        if e.get("error"):
            err_by_mode[e["mode"]] += 1

    # Build report
    report = []
    report.append(f"# RAG Benchmark Report\n")
    report.append(f"**Model**: {eval_data['meta'].get('bertscore_model', '?')}  ")
    report.append(f"**Lang**: {eval_data['meta'].get('lang', '?')}  ")
    report.append(f"**Bootstrap resamples**: {args.bootstrap}  ")
    report.append(
        f"**Queries**: {len(queries)} | **Total entries**: {len(entries)} | **After median collapse**: {len(collapsed)}\n"
    )

    report.append("\n## BERTScore F1 by Mode (with 95% CI)\n")
    report.append(
        format_md_table(mode_rows, ["Mode", "N", "Mean F1", "95% CI", "Median"])
    )

    report.append("\n\n## BERTScore F1 by Type × Mode (mean)\n")
    report.append(
        format_md_table(type_rows, ["Type", "bm25", "naive", "hybrid", "mix", "graph"])
    )

    report.append("\n\n## Latency by Mode\n")
    report.append(format_md_table(lat_rows, ["Mode", "N", "Mean", "Median", "Max"]))

    report.append("\n\n## Errors by Mode\n")
    err_rows = [[m, err_by_mode.get(m, 0)] for m in ["bm25", "naive", "hybrid", "mix", "graph"]]
    report.append(format_md_table(err_rows, ["Mode", "Error count"]))

    # Expected vs actual best mode (per-query winner)
    report.append("\n\n## Expected vs Actual Best Mode (per query)\n")
    by_q = defaultdict(dict)
    for e in collapsed:
        if e.get("bertscore_f1") is not None:
            by_q[e["query_id"]][e["mode"]] = e["bertscore_f1"]

    match_count = 0
    total_q = 0
    eva_rows = []
    for qid, modes_f1 in by_q.items():
        if not modes_f1:
            continue
        total_q += 1
        actual_best = max(modes_f1, key=modes_f1.get)
        expected = queries[qid].get("expected_best_mode", "?")
        match = "✓" if actual_best == expected else "✗"
        if actual_best == expected:
            match_count += 1
        eva_rows.append(
            [
                qid,
                queries[qid]["type"],
                expected,
                actual_best,
                f"{modes_f1[actual_best]:.4f}",
                match,
            ]
        )
    eva_rows.sort(key=lambda r: r[0])
    report.append(
        format_md_table(
            eva_rows, ["Query", "Type", "Expected", "Actual", "F1", "Match"]
        )
    )
    report.append(
        f"\n**Expected-mode hit rate**: {match_count}/{total_q} = {100*match_count/total_q:.1f}%"
        if total_q
        else "\n(no scored queries)"
    )

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"[REPORT] {REPORT_FILE}")
    print("\n".join(report[-10:]))  # tail preview


if __name__ == "__main__":
    main()
