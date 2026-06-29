"""Recall@K cho multi-hop queries (CHỈ Recall@K).

Recall@K = |top-K retrieved chunk_ids ∩ gold_evidence_chunks| / |gold|

Gold evidence (suy từ graphml — các query này được mine từ đồ thị):
  - 2hop: source_id(edge A-B) ∪ source_id(edge B-C)   (A,B,C có trong schema, match lowercase)

Retrieved chunk_ids (cùng không gian chunk-<hash>):
  - naive/hybrid/mix/graph: POST /query/data → data.chunks[].chunk_id (theo thứ tự)
  - bm25:                   eval/bm25_index.search() (id từ lightrag_doc_chunks)

Usage:
  venv/bin/python eval/compute_recall.py
  venv/bin/python eval/compute_recall.py --modes naive,hybrid,mix,graph   # bỏ bm25
  venv/bin/python eval/compute_recall.py --sets 2hop --limit 3            # smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests
import networkx as nx

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).parent
GRAPHML = "/home/tts/AI/aiQuoc/tlu_workspace/graph_chunk_entity_relation.graphml"
SERVER_URL = "http://localhost:9621"
OUT_DIR = HERE / "results" / "recall"
THROTTLE_SEC = 0.2  # không sinh LLM → nhẹ
QFILES = {"2hop": "queries_2hop.json"}


def _chunks(sid) -> set:
    return {s for s in str(sid or "").split("<SEP>") if s}


def build_gold(G, sets):
    """Return {qid: (set_name, gold_set)} + skipped[]."""
    gold, skipped = {}, []
    if "2hop" in sets:
        for x in json.load(open(HERE / QFILES["2hop"], encoding="utf-8"))["queries"]:
            a, b, c = x["A"].lower().strip(), x["B"].lower().strip(), x["C"].lower().strip()
            if G.has_edge(a, b) and G.has_edge(b, c):
                g = _chunks(G.edges[a, b].get("source_id")) | _chunks(G.edges[b, c].get("source_id"))
                gold[x["id"]] = ("2hop", g)
            else:
                skipped.append(x["id"])
    return gold, skipped


def load_query_text(sets):
    qtext = {}
    for s in sets:
        for x in json.load(open(HERE / QFILES[s], encoding="utf-8"))["queries"]:
            qtext[x["id"]] = x["query"]
    return qtext


def retrieve_server(query, mode, rerank=False, chunk_top_k=None):
    payload = {"query": query, "mode": mode, "stream": False}
    if rerank:
        payload["enable_rerank"] = True
    if chunk_top_k:
        payload["chunk_top_k"] = chunk_top_k
    r = requests.post(f"{SERVER_URL}/query/data", json=payload, timeout=300)
    r.raise_for_status()
    chunks = r.json().get("data", {}).get("chunks", [])
    ids = [c.get("chunk_id") or c.get("id") for c in chunks]
    return [i for i in ids if i]


SERVER_LOGS = [
    "/home/tts/AI/aiQuoc/TLU-Chatbot/server.log",
    "/home/tts/AI/aiQuoc/TLU-Chatbot/lightrag.log",
]


def preflight_rerank_check(qtext, log_paths):
    """Bắn 1 query rerank rồi đọc log mới: nếu server fallback (429/hết quota/không có
    model) thì ABORT to tiếng — tránh ghi số rác như lần trước.

    `/query/data` không expose rerank_score nên log là tín hiệu đáng tin duy nhất.
    """
    import os

    offsets = {p: (os.path.getsize(p) if os.path.exists(p) else None) for p in log_paths}
    if all(v is None for v in offsets.values()):
        sys.exit(
            "[GUARD] Không tìm thấy server log để verify rerank "
            f"({log_paths}). Dùng --server-log để chỉ đúng file, hoặc bỏ --rerank."
        )

    probe_q = next(iter(qtext.values()))
    print("[GUARD] Pre-flight: bắn 1 query rerank để kiểm tra server có rerank thật...")
    try:
        retrieve_server(probe_q, "mix", rerank=True, chunk_top_k=20)
    except Exception as e:
        sys.exit(f"[GUARD] Probe query lỗi: {type(e).__name__}: {e}")
    time.sleep(1.5)  # chờ server flush log

    new = ""
    for p, off in offsets.items():
        if off is None:
            continue
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                f.seek(off)
                new += f.read()
        except OSError:
            pass

    bad = any(s in new for s in (
        "Rerank API error", "using original chunks", "no rerank model is configured",
        "Error during reranking",
    ))
    good = "Successfully reranked" in new or "Rerank filtering" in new
    if bad or not good:
        sys.exit(
            "\n[GUARD] RERANK FALLBACK / KHONG KICH HOAT — KHONG ghi so rac.\n"
            "  Server khong rerank that (429/het quota/khong co model). "
            "Thay key Cohere con quota hoac dung reranker local roi chay lai.\n"
            f"  Log moi (duoi):\n{new[-600:] or '(trong — khong thay dong rerank nao)'}"
        )
    print("[GUARD] OK — rerank chay that (thay 'Successfully reranked' trong log).\n")


def retrieve_bm25(query, max_k, bm25_index):
    import bm25_index as bm25_mod
    return [c["id"] for c in bm25_mod.search(query, top_k=max_k, index=bm25_index) if c.get("id")]


def recall_at_k(retrieved, gold, k):
    return len(set(retrieved[:k]) & gold) / len(gold) if gold else None


def main():
    global SERVER_URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="2hop")
    ap.add_argument("--modes", default="bm25,naive,hybrid,mix,graph")
    ap.add_argument("--k", default="3,5,10")
    ap.add_argument("--server-url", default=SERVER_URL)
    ap.add_argument("--rerank", action="store_true", help="bật enable_rerank cho server modes")
    ap.add_argument("--chunk-top-k", type=int, default=0, help="ép chunk_top_k (kiểm soát confound slice khi A/B rerank)")
    ap.add_argument("--out-tag", default="", help="hậu tố file output, vd '_rerank_on' → recall_summary_rerank_on.json")
    ap.add_argument("--server-log", default="", help="đường dẫn log server để guard verify rerank (mặc định: server.log + lightrag.log)")
    ap.add_argument("--no-guard", action="store_true", help="bỏ qua pre-flight guard chống fallback (KHÔNG khuyến nghị)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: chỉ N câu đầu mỗi set")
    args = ap.parse_args()

    SERVER_URL = args.server_url
    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    Ks = sorted(int(k) for k in args.k.split(","))
    max_k = max(Ks)

    print("Loading graphml...")
    G = nx.read_graphml(GRAPHML)
    gold, skipped = build_gold(G, sets)
    qtext = load_query_text(sets)

    if args.limit:
        kept, seen = {}, {}
        for qid, (sn, g) in gold.items():
            seen[sn] = seen.get(sn, 0) + 1
            if seen[sn] <= args.limit:
                kept[qid] = (sn, g)
        gold = kept
    print(f"Gold: {len(gold)} queries | skipped: {len(skipped)} {skipped}\n")

    # Guard: nếu yêu cầu rerank, verify server thực sự rerank trước khi tốn call
    server_modes = [m for m in modes if m != "bm25"]
    if args.rerank and server_modes and not args.no_guard:
        log_paths = [args.server_log] if args.server_log else SERVER_LOGS
        preflight_rerank_check(qtext, log_paths)

    bm25_index = None
    if "bm25" in modes:
        import bm25_index as bm25_mod
        bm25_index = bm25_mod.load_index()

    raw = []
    for qid, (set_name, g) in gold.items():
        q = qtext[qid]
        for mode in modes:
            try:
                retrieved = (retrieve_bm25(q, max_k, bm25_index) if mode == "bm25"
                             else retrieve_server(q, mode, rerank=args.rerank,
                                                  chunk_top_k=args.chunk_top_k or None))
                if mode != "bm25":
                    time.sleep(THROTTLE_SEC)
                rec = {f"recall@{k}": recall_at_k(retrieved, g, k) for k in Ks}
                raw.append({"qid": qid, "set": set_name, "mode": mode,
                            "n_gold": len(g), "n_retrieved": len(retrieved),
                            "retrieved_ids_top_max_k": retrieved[:max_k],
                            "gold": sorted(g), **rec})
            except Exception as e:
                raw.append({"qid": qid, "set": set_name, "mode": mode,
                            "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(f"  ERR {qid} {mode}: {e}")
        print(f"  done {qid} ({set_name})")

    # Aggregate
    summary = {}
    for s in sets:
        summary[s] = {}
        for mode in modes:
            rows = [r for r in raw if r["set"] == s and r["mode"] == mode and "error" not in r]
            if rows:
                summary[s][mode] = {"n": len(rows),
                                    **{f"recall@{k}": round(sum(r[f"recall@{k}"] for r in rows) / len(rows), 4)
                                       for k in Ks}}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.out_tag
    json.dump({"raw": raw, "skipped": skipped, "config": {"rerank": args.rerank, "chunk_top_k": args.chunk_top_k}},
              open(OUT_DIR / f"recall_raw{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(summary, open(OUT_DIR / f"recall_summary{tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    for s in sets:
        if not summary.get(s):
            continue
        print(f"\n=== Recall@K — {s} ===")
        hdr = "mode".ljust(8) + "n".rjust(5) + "".join(f"R@{k}".rjust(9) for k in Ks)
        print(hdr); print("-" * len(hdr))
        for mode in modes:
            if mode in summary[s]:
                m = summary[s][mode]
                print(mode.ljust(8) + str(m["n"]).rjust(5)
                      + "".join(f"{m[f'recall@{k}']:.4f}".rjust(9) for k in Ks))
    print(f"\n[SAVED] {OUT_DIR}/recall_raw{tag}.json, recall_summary{tag}.json")


if __name__ == "__main__":
    main()
