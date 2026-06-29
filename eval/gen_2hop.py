"""Generate 2-hop cross-subject benchmark queries.

Strategy:
  Mine graphml for paths A → B → C where:
    - A strictly in subject X (no other subject)
    - C strictly in subject Y (no other subject), X ≠ Y
    - A and C share NO source files (so vector search can't collocate them)
    - B is the intermediate node (may be in any subject)
    - Both edges A-B and B-C have edge descriptions

  Query asks about A↔C relationship WITHOUT mentioning B.
  Correct answer requires 2-hop graph traversal.
  Vector search should struggle since A and C are never in the same chunk.

Usage:
  python gen_2hop.py --count 5 --pairs 12 --out eval/queries_2hop.json
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time as _time
from collections import defaultdict
from pathlib import Path

import networkx as nx
import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from gen_queries import SUBJECTS, detect_subject, _strip_images

ROUTER_HOST = os.getenv("ROUTER_HOST", "http://localhost:20128/v1")
ROUTER_KEY = os.getenv("ROUTER_API_KEY", "dummy")
GEN_MODEL = os.getenv("GEN_QUERY_MODEL", "nvidia/minimaxai/minimax-m2.7")
_env_fb = os.getenv("GEN_FAILOVER_MODELS", "") or os.getenv("ROUTER_FAILOVER_MODELS", "")
GEN_FAILOVER = [m.strip() for m in _env_fb.split(",") if m.strip() and m.strip() != GEN_MODEL]
GEN_CHAIN = [GEN_MODEL] + GEN_FAILOVER
CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "3"))
MAX_RPM = int(os.getenv("GEN_MAX_RPM", "40"))
MIN_INTERVAL = 60.0 / MAX_RPM

GRAPHML = Path("/home/tts/AI/aiQuoc/tlu_workspace/graph_chunk_entity_relation.graphml")

SUBSTANTIVE_TYPES = {
    "concept", "algorithm", "data_structure", "language",
    "framework", "tool", "architecture", "metric",
}

GENERIC = {
    "data", "title", "information", "system", "content", "process", "value",
    "user", "users", "method", "type", "name", "result", "model", "table",
    "size", "format", "id", "key", "field", "object", "function", "page",
    "input", "output", "code", "file", "task", "test", "service", "tool",
    "step", "stage", "phase", "concept", "ai", "it", "image", "text",
    "understanding", "knowledge", "study", "research", "example", "case",
    "application", "form", "speed", "variable", "class", "element",
    "attribute", "label", "rule", "customer", "modeling", "maintenance",
    "database", "web", "user input", "server", "client", "network",
}

PROMPT_2HOP = """Sinh 1 câu hỏi yêu cầu suy luận 2 bước qua 2 lĩnh vực khác nhau.

Bối cảnh:
- Khái niệm A: **{A}** (thuộc lĩnh vực {subject_a})
  Mô tả: {desc_A}

- Khái niệm C: **{C}** (thuộc lĩnh vực {subject_c})
  Mô tả: {desc_C}

- Mối liên hệ A→B: {rel_AB}
- Mối liên hệ B→C: {rel_BC}

Yêu cầu đặc biệt:
- Câu hỏi phải hỏi về mối quan hệ TRỰC TIẾP giữa **{A}** và **{C}**
- KHÔNG được nhắc đến khái niệm trung gian
- Câu hỏi KHÔNG THỂ trả lời nếu chỉ biết 1 trong 2 lĩnh vực
- Reference phải giải thích chuỗi suy luận 2 bước (3-5 câu)
- evidence_keywords gồm cả 2 khái niệm A, C và 2-3 từ khoá kết nối

Bắt buộc viết câu hỏi và câu trả lời bằng TIẾNG VIỆT.
Trả về JSON duy nhất (không markdown):
{{"query": "...", "reference": "...", "evidence_keywords": ["...", "..."], "expected_best_mode": "graph"}}
"""

_last_call_time = [0.0]
_rate_lock = None


def _connect_pg():
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        dbname=os.environ['POSTGRES_DATABASE'],
    )


def load_entity_chunks(entity_names: set) -> dict:
    """Load chunk content for given entity names from Postgres."""
    if not entity_names:
        return {}
    conn = _connect_pg()
    cur = conn.cursor()
    # lightrag_doc_chunks has id = chunk-xxx; entities link via source_id
    # We load by chunk ID set
    conn.close()
    return {}


def mine_2hop_paths(G, entity_subjects, entity_files, entity_descs):
    """Return {pair: [path_dict]} where pair = (sa, sc) sorted."""
    paths_by_pair = defaultdict(list)

    for A, a_subjs in entity_subjects.items():
        if len(a_subjs) != 1:
            continue
        if A.lower().strip() in GENERIC or len(A) < 4:
            continue
        a_type = G.nodes[A].get('entity_type', '').lower().strip()
        if a_type not in SUBSTANTIVE_TYPES:
            continue
        sa = next(iter(a_subjs))
        desc_A = entity_descs.get(A, '')
        if len(desc_A) < 80:
            continue

        try:
            neighbors_A = list(G.neighbors(A))
        except Exception:
            continue

        for B in neighbors_A:
            if B.lower().strip() in GENERIC:
                continue
            e_ab_data = G.get_edge_data(A, B) or {}
            rel_AB = e_ab_data.get('description', '') or e_ab_data.get('weight', '')
            if not rel_AB or len(str(rel_AB)) < 20:
                continue

            try:
                neighbors_B = list(G.neighbors(B))
            except Exception:
                continue

            for C in neighbors_B:
                if C == A or C == B:
                    continue
                if C not in entity_subjects:
                    continue
                if C.lower().strip() in GENERIC or len(C) < 4:
                    continue
                c_type = G.nodes[C].get('entity_type', '').lower().strip()
                if c_type not in SUBSTANTIVE_TYPES:
                    continue
                c_subjs = entity_subjects[C]
                if len(c_subjs) != 1:
                    continue
                sc = next(iter(c_subjs))
                if sa == sc:
                    continue

                # A and C must not share any source files
                if entity_files.get(A, set()) & entity_files.get(C, set()):
                    continue

                desc_C = entity_descs.get(C, '')
                if len(desc_C) < 80:
                    continue

                e_bc_data = G.get_edge_data(B, C) or {}
                rel_BC = e_bc_data.get('description', '') or e_bc_data.get('weight', '')
                if not rel_BC or len(str(rel_BC)) < 20:
                    continue

                pair = tuple(sorted([sa, sc]))
                # Score: longer descriptions + longer edge descriptions = more meaningful
                score = len(desc_A) / 500 + len(desc_C) / 500 + len(str(rel_AB)) / 200 + len(str(rel_BC)) / 200
                paths_by_pair[pair].append({
                    'A': A, 'B': B, 'C': C,
                    'sa': sa, 'sc': sc,
                    'desc_A': desc_A[:400],
                    'desc_C': desc_C[:400],
                    'rel_AB': str(rel_AB)[:300],
                    'rel_BC': str(rel_BC)[:300],
                    'score': score,
                })

    for pair in paths_by_pair:
        paths_by_pair[pair].sort(key=lambda x: -x['score'])

    return paths_by_pair


async def _rate_limit_wait():
    global _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    async with _rate_lock:
        now = _time.time()
        wait = _last_call_time[0] + MIN_INTERVAL - now
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_time[0] = _time.time()


async def gen_one(client, path: dict):
    prompt = PROMPT_2HOP.format(
        A=path['A'],
        C=path['C'],
        subject_a=SUBJECTS[path['sa']]['label'],
        subject_c=SUBJECTS[path['sc']]['label'],
        desc_A=path['desc_A'],
        desc_C=path['desc_C'],
        rel_AB=path['rel_AB'],
        rel_BC=path['rel_BC'],
    )
    last_err = None
    for model in GEN_CHAIN:
        await _rate_limit_wait()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                timeout=60,
            )
            text = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                last_err = f"{model}: no JSON"
                continue
            data = json.loads(m.group(0))
            if not {"query", "reference", "evidence_keywords"}.issubset(data.keys()):
                last_err = f"{model}: missing keys"
                continue
            data.setdefault("expected_best_mode", "graph")
            return data, None
        except Exception as ex:
            last_err = f"{model}: {type(ex).__name__}: {str(ex)[:80]}"
            continue
    return None, last_err


async def main(count_per_pair: int, num_pairs: int, out_path: Path):
    from openai import AsyncOpenAI

    print(f"[GRAPH] Loading {GRAPHML}")
    G = nx.read_graphml(GRAPHML)
    print(f"[GRAPH] Nodes={G.number_of_nodes()} Edges={G.number_of_edges()}")

    entity_subjects = {}
    entity_files = {}
    entity_descs = {}
    for node, attrs in G.nodes(data=True):
        fp = attrs.get('file_path', '')
        desc = attrs.get('description', '')
        if not fp:
            continue
        files = [f for f in fp.split('<SEP>') if f.strip()]
        subjs = set(s for f in files for s in [detect_subject(f)] if s)
        if subjs:
            entity_subjects[node] = subjs
            entity_files[node] = set(files)
            entity_descs[node] = desc.split('<SEP>')[0]  # take first description

    print(f"[MINE] Finding 2-hop paths...")
    paths_by_pair = mine_2hop_paths(G, entity_subjects, entity_files, entity_descs)

    sorted_pairs = sorted(paths_by_pair.items(), key=lambda x: -len(x[1]))[:num_pairs]
    print(f"\n[PAIRS] Selected {len(sorted_pairs)} pairs:")
    for (sa, sb), lst in sorted_pairs:
        print(f"  {sa:>7} ↔ {sb:<7}: {len(lst):>4} paths (top: {lst[0]['A'][:20]}→{lst[0]['B'][:15]}→{lst[0]['C'][:20]})")

    client = AsyncOpenAI(base_url=ROUTER_HOST, api_key=ROUTER_KEY, timeout=60)
    sem = asyncio.Semaphore(CONCURRENCY)
    rng = random.Random(42)

    tasks = []

    async def gen_with_id(pair_idx, seq, sa, sc, path):
        async with sem:
            result, err = await gen_one(client, path)
        qid = f"2hop-{sa}-{sc}-{seq:02d}"
        if result is None:
            print(f"  FAIL {qid} ({path['A'][:20]}→{path['B'][:15]}→{path['C'][:20]}): {err}")
            return None
        new_q = {
            "id": qid,
            "type": "multihop_2hop",
            "subject": f"{SUBJECTS[sa]['label']} ↔ {SUBJECTS[sc]['label']}",
            "pair": f"{sa}_{sc}",
            "chain": f"{path['A']} → {path['B']} → {path['C']}",
            "A": path['A'], "B": path['B'], "C": path['C'],
            "source_file": "",
            "query": result["query"],
            "reference": result["reference"],
            "evidence_keywords": result["evidence_keywords"],
            "expected_best_mode": result.get("expected_best_mode", "graph"),
        }
        print(f"  OK   {qid} ({path['A'][:18]}→{path['B'][:12]}→{path['C'][:18]}): {result['query'][:70]}")
        return new_q

    for pair_idx, ((sa, sc), lst) in enumerate(sorted_pairs):
        # Deduplicate: avoid same A or same C, and avoid reversed chains (A-B-C vs C-B-A)
        seen_A, seen_C, seen_AC = set(), set(), set()
        selected = []
        for p in lst:
            ac_key = tuple(sorted([p['A'], p['C']]))
            if p['A'] in seen_A or p['C'] in seen_C or ac_key in seen_AC:
                continue
            selected.append(p)
            seen_A.add(p['A'])
            seen_C.add(p['C'])
            seen_AC.add(ac_key)
            if len(selected) >= count_per_pair:
                break
        if len(selected) < count_per_pair:
            print(f"  [WARN] {sa}↔{sc}: only {len(selected)} unique paths (need {count_per_pair})")
        for seq, p in enumerate(selected, 1):
            tasks.append(gen_with_id(pair_idx, seq, sa, sc, p))

    print(f"\n[GEN] Generating {len(tasks)} 2-hop queries...")
    results = await asyncio.gather(*tasks)
    queries = [r for r in results if r is not None]

    out = {
        "meta": {"total": len(queries), "type": "multihop_2hop",
                 "note": "A strictly in subj_X, C strictly in subj_Y, no shared files. B hidden from query."},
        "queries": queries
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {out_path} | total={len(queries)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--out", default="eval/queries_2hop.json")
    args = parser.parse_args()
    asyncio.run(main(args.count, args.pairs, Path(args.out)))
