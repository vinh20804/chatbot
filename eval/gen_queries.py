"""Generate benchmark queries for each type by sampling chunks from corpus.

Reads chunks from Postgres, samples per subject, calls LLM to gen 1 query+reference
per chunk for the given type. Appends to existing queries_{type}.json (preserves
old queries; assigns next available ID per subject).

Usage:
  python gen_queries.py --type factoid --count 14   # 14 new per subject = 70 total
  python gen_queries.py --type relational --count 14
  python gen_queries.py --type broad --count 14
  python gen_queries.py --type aggregate --count 14
"""

import argparse
import asyncio
import json
import os
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

import time as _time

ROUTER_HOST = os.getenv("ROUTER_HOST", "http://localhost:20128/v1")
ROUTER_KEY = os.getenv("ROUTER_API_KEY", "dummy")
GEN_MODEL = os.getenv("GEN_QUERY_MODEL", "nvidia/minimaxai/minimax-m2.7")
# Failover chain: primary GEN_MODEL first, then GEN_FAILOVER_MODELS (csv), then ROUTER_FAILOVER_MODELS
_env_fb = os.getenv("GEN_FAILOVER_MODELS", "") or os.getenv("ROUTER_FAILOVER_MODELS", "")
GEN_FAILOVER = [m.strip() for m in _env_fb.split(",") if m.strip() and m.strip() != GEN_MODEL]
GEN_CHAIN = [GEN_MODEL] + GEN_FAILOVER
CONCURRENCY = int(os.getenv("GEN_CONCURRENCY", "3"))

# Rate limit: max 40 req/min = 1 req every 1.5s globally
MAX_RPM = int(os.getenv("GEN_MAX_RPM", "40"))
MIN_INTERVAL = 60.0 / MAX_RPM
_last_call_time = [0.0]
_rate_lock = None  # initialized in event loop

# Map source_file substring/folder hint → subject code (used for query ID prefix)
# Plus subject code → human label for prompt context
SUBJECTS = {
    "ktmt":   {"label": "Kiến trúc máy tính",          "match": ["KienTrucMayTinh", "Kien_truc", "kien truc may tinh"]},
    "httt":   {"label": "Chuyên đề Hệ thống thông tin", "match": ["Chu de", "HTTT", "ChuyenDe", "he thong thong minh", "Tong quan ve he thong"]},
    "cnw":    {"label": "Công nghệ web (PHP, HTML, CSS, JS)", "match": ["CNWeb", "CSS", "Jquery", "JavaScript", "HTML", "PHP", "JQuery"]},
    "ntw":    {"label": "Nền tảng web",                 "match": ["NenTangWeb", "NTW", "nen tang web"]},
    "pttkht": {"label": "Phân tích thiết kế hệ thống",  "match": ["PTTKHT", "Use case", "use_case", "UseCase", "UML", "Biểu đồ", "Bieu do", "Activity Diagram", "Yêu cầu"]},
    "hci":    {"label": "Tương tác người-máy (HCI, thiết kế giao diện)", "match": ["Bai 1", "Bai 2", "Bai 3", "Bai 4", "Bai 5", "Bai 6", "Bai 7", "Bai 8", "Bai 9", "Bai 10", "Bai 11", "Bai 12", "Bai 13", "Bai 14", "Bai 15", "Bai 16", "Bai_1", "Bai_2", "Bai_3", "Bai_4", "Bai_5", "Bai_6", "Bai_7", "Bai_8", "Bai_9", "giao dien", "Tinh tien dung"]},
    "csdl":   {"label": "Cơ sở dữ liệu",                "match": ["Chuong0-", "Chuong1-", "Chuong2-", "Chuong3-", "Chuong4_", "Chuong5-", "Chuong6-", "TongQuanCSDL", "MoHinhTTLK", "MoHinhDuLieuQuanHe", "NgonNguThaoTacDuLieu", "PTH va Khoa", "DangChuanVaChuanhoa"]},
    "kpdl":   {"label": "Khai phá dữ liệu",             "match": ["Chapter0_", "Chapter1_", "Chapter2_", "Chapter3_", "Chapter4_", "Chapter5_", "LuatKetHop", "PhanLopDuLieu", "PhanCumDuLieu", "TienXuLyDuLieu"]},
}


TYPE_PROMPTS = {
    "factoid": """Sinh 1 câu hỏi FACTOID (hỏi 1 fact đơn lẻ, có đáp án ngắn rõ ràng) dựa trên chunk dưới đây.

Yêu cầu:
- Câu hỏi tiếng Việt, tự nhiên, có thể trả lời bằng 1-3 câu ngắn
- Reference là câu trả lời CHÍNH XÁC, NGẮN GỌN (1-3 câu)
- evidence_keywords là 3-5 từ khoá quan trọng trong câu trả lời
- expected_best_mode dự đoán mode nào trả lời tốt nhất (factoid thường là 'naive' hoặc 'bm25')

Chunk:
{chunk}

Trả về JSON duy nhất (không markdown):
{{"query": "...", "reference": "...", "evidence_keywords": ["...", "..."], "expected_best_mode": "naive"}}""",

    "relational": """Sinh 1 câu hỏi RELATIONAL (hỏi về mối quan hệ, tương tác, ảnh hưởng giữa 2+ khái niệm) dựa trên chunk.

Yêu cầu:
- Câu hỏi tiếng Việt yêu cầu giải thích MỐI QUAN HỆ (vd: "Quan hệ giữa X và Y?", "X ảnh hưởng thế nào đến Y?", "Khi X xảy ra, Y bị tác động ra sao?")
- Reference giải thích RÕ mối quan hệ (3-5 câu, không chỉ định nghĩa từng đối tượng riêng lẻ)
- evidence_keywords là 3-5 từ khoá đại diện cho các đối tượng + mối quan hệ
- expected_best_mode (relational thường là 'hybrid' hoặc 'graph')

Chunk:
{chunk}

Trả về JSON duy nhất:
{{"query": "...", "reference": "...", "evidence_keywords": ["...", "..."], "expected_best_mode": "hybrid"}}""",

    "broad": """Sinh 1 câu hỏi BROAD (rộng, cần tổng hợp nhiều khía cạnh, không 1 fact đơn lẻ) dựa trên chunk.

Yêu cầu:
- Câu hỏi tiếng Việt mang tính tổng quan/khám phá (vd: "Tổng quan về...", "Các khía cạnh của...", "Vai trò và ý nghĩa...")
- Reference cover NHIỀU khía cạnh, có structure (5-10 câu)
- evidence_keywords là 5-7 từ khoá đại diện chủ đề rộng
- expected_best_mode (broad thường là 'mix' hoặc 'graph')

Chunk:
{chunk}

Trả về JSON duy nhất:
{{"query": "...", "reference": "...", "evidence_keywords": ["...", "..."], "expected_best_mode": "mix"}}""",

    "aggregate": """Sinh 1 câu hỏi AGGREGATE (liệt kê, đếm, gom nhóm các items) dựa trên chunk.

Yêu cầu:
- Câu hỏi tiếng Việt yêu cầu LIỆT KÊ items (vd: "Liệt kê các...", "Có những ... nào?", "Các loại ... là gì?")
- Reference là danh sách CỤ THỂ, ĐẦY ĐỦ items (bullet hoặc đánh số)
- evidence_keywords là các tên items chính
- expected_best_mode (aggregate có thể là 'mix' hoặc 'hybrid')

Chunk:
{chunk}

Trả về JSON duy nhất:
{{"query": "...", "reference": "...", "evidence_keywords": ["...", "..."], "expected_best_mode": "mix"}}""",
}


def detect_subject(file_path: str, content_hint: str = "") -> str | None:
    """Map file_path → subject code. Falls back to content_hint (first 1KB of doc)
    to recover orphan docs where doc_name = doc-XXX (no real filename)."""
    haystack = (file_path or "") + " " + (content_hint or "")[:1500]
    if not haystack.strip():
        return None
    for code, info in SUBJECTS.items():
        for hint in info["match"]:
            if hint.lower() in haystack.lower():
                return code
    return None


def _strip_images(text: str) -> str:
    """Remove image markdown markers."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _connect_pg():
    import psycopg2
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        dbname=os.environ['POSTGRES_DATABASE'],
    )


def load_chunks_by_subject():
    """Load chunks from Postgres, group by detected subject."""
    conn = _connect_pg()
    cur = conn.cursor()
    cur.execute("SELECT id, content, file_path, tokens FROM lightrag_doc_chunks WHERE tokens > 100")
    chunks_by_subject = defaultdict(list)
    skipped = 0
    for chunk_id, content, file_path, tokens in cur.fetchall():
        if not content or len(content) < 200:
            skipped += 1
            continue
        subj = detect_subject(file_path, content)
        if subj is None:
            skipped += 1
            continue
        chunks_by_subject[subj].append({
            "id": chunk_id, "content": content, "file_path": file_path, "tokens": tokens,
        })
    conn.close()
    print(f"[CHUNKS] Loaded by subject:")
    for s, lst in chunks_by_subject.items():
        print(f"  {s}: {len(lst)} chunks")
    print(f"  (skipped {skipped} short/unmatched)")
    return chunks_by_subject


def load_full_docs_by_subject():
    """Load full MD docs from Postgres, group by detected subject."""
    conn = _connect_pg()
    cur = conn.cursor()
    cur.execute("SELECT id, doc_name, content FROM lightrag_doc_full")
    docs_by_subject = defaultdict(list)
    skipped = 0
    for doc_id, doc_name, content in cur.fetchall():
        if not content or len(content) < 500:
            skipped += 1
            continue
        subj = detect_subject(doc_name or "", content)
        if subj is None:
            skipped += 1
            continue
        clean = _strip_images(content)
        docs_by_subject[subj].append({
            "id": doc_id, "content": clean, "file_path": doc_name,
        })
    conn.close()
    print(f"[FULL DOCS] Loaded by subject:")
    for s, lst in docs_by_subject.items():
        avg_len = sum(len(d["content"]) for d in lst) // max(len(lst), 1)
        print(f"  {s}: {len(lst)} docs (avg {avg_len:,} chars)")
    print(f"  (skipped {skipped} short/unmatched)")
    return docs_by_subject


async def _rate_limit_wait():
    """Global pacing: ensure at least MIN_INTERVAL between consecutive API calls."""
    global _rate_lock
    if _rate_lock is None:
        _rate_lock = asyncio.Lock()
    async with _rate_lock:
        now = _time.time()
        wait = _last_call_time[0] + MIN_INTERVAL - now
        if wait > 0:
            await asyncio.sleep(wait)
        _last_call_time[0] = _time.time()


async def gen_one_query(client, query_type: str, subject_code: str, source: dict):
    """Call LLM to gen 1 query from a chunk OR full doc, with failover.

    For factoid: source is a chunk (short, ~500-1000 tokens).
    For others: source is a full MD doc (longer, ~10-50K chars).
    """
    # Truncate generously for full docs; chunks pass through
    max_len = 3000 if query_type == "factoid" else 30000
    prompt = TYPE_PROMPTS[query_type].format(chunk=source["content"][:max_len])
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
            required = {"query", "reference", "evidence_keywords"}
            if not required.issubset(data.keys()):
                last_err = f"{model}: missing keys {required - set(data.keys())}"
                continue
            data.setdefault("expected_best_mode", "hybrid")
            return data, None
        except Exception as ex:
            last_err = f"{model}: {type(ex).__name__}: {str(ex)[:80]}"
            continue
    return None, last_err


def load_existing_queries(qfile: Path):
    """Load existing queries; return data dict + max id per subject."""
    if not qfile.exists():
        return {"meta": {}, "queries": []}, defaultdict(int)
    with open(qfile, encoding="utf-8") as f:
        data = json.load(f)
    max_id = defaultdict(int)
    for q in data.get("queries", []):
        m = re.match(r"^([a-z]+)-[a-z]+-(\d+)$", q["id"])
        if m:
            subj, num = m.group(1), int(m.group(2))
            if num > max_id[subj]:
                max_id[subj] = num
    return data, max_id


async def gen_for_type(query_type: str, count_per_subject: int, qfile: Path, subjects_filter: list[str] | None = None):
    """Gen count_per_subject queries per subject for the given type.

    subjects_filter: if provided, only gen for these subject codes (e.g. ["csdl","kpdl"]).
    """
    from openai import AsyncOpenAI
    import random

    data, max_id = load_existing_queries(qfile)
    # Factoid samples chunks (cheap, 1 fact in 1 chunk).
    # Relational/broad/aggregate need full doc context to gen realistic queries.
    if query_type == "factoid":
        source_by_subject = load_chunks_by_subject()
        source_label = "chunk"
    else:
        source_by_subject = load_full_docs_by_subject()
        source_label = "full_doc"

    if subjects_filter:
        unknown = [s for s in subjects_filter if s not in SUBJECTS]
        if unknown:
            raise SystemExit(f"[FATAL] Unknown subject codes in --subjects: {unknown}. Known: {list(SUBJECTS.keys())}")
        source_by_subject = {s: source_by_subject.get(s, []) for s in subjects_filter}
        print(f"[FILTER] Restricted to subjects: {subjects_filter}")

    client = AsyncOpenAI(base_url=ROUTER_HOST, api_key=ROUTER_KEY, timeout=60)
    sem = asyncio.Semaphore(CONCURRENCY)

    # Type abbreviation for ID (factoid → fact, relational → rel, etc.)
    type_short = {"factoid": "fact", "relational": "rel", "broad": "broad", "aggregate": "agg"}[query_type]

    new_queries = []
    tasks = []

    async def gen_with_id(subj, item, seq):
        async with sem:
            result, err = await gen_one_query(client, query_type, subj, item)
        if result is None:
            print(f"  FAIL {subj}-{seq}: {err}")
            return None
        qid = f"{subj}-{type_short}-{seq:02d}"
        source_file = item["file_path"].replace(".pdf", "").replace(" ", "_")
        new_q = {
            "id": qid,
            "type": query_type,
            "subject": SUBJECTS[subj]["label"],
            "source_file": source_file,
            "query": result["query"],
            "reference": result["reference"],
            "evidence_keywords": result["evidence_keywords"],
            "expected_best_mode": result.get("expected_best_mode", "hybrid"),
        }
        print(f"  OK   {qid}: {result['query'][:80]}")
        return new_q

    rng = random.Random(42)
    for subj, items in source_by_subject.items():
        if not items:
            print(f"[SKIP] {subj}: no {source_label}s")
            continue
        # If pool < count, sample with replacement (full docs case)
        if count_per_subject <= len(items):
            sampled = rng.sample(items, count_per_subject)
        else:
            sampled = rng.choices(items, k=count_per_subject)
        start_id = max_id[subj] + 1
        for i, item in enumerate(sampled):
            tasks.append(gen_with_id(subj, item, start_id + i))

    print(f"\n[GEN] Generating {len(tasks)} new queries...")
    results = await asyncio.gather(*tasks)
    new_queries = [r for r in results if r is not None]

    data["queries"] = data.get("queries", []) + new_queries
    data["meta"]["total"] = len(data["queries"])
    data["meta"]["last_gen"] = {"type": query_type, "count_per_subject": count_per_subject, "added": len(new_queries)}

    qfile.parent.mkdir(parents=True, exist_ok=True)
    with open(qfile, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] {qfile} | total={len(data['queries'])} (+{len(new_queries)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=list(TYPE_PROMPTS.keys()))
    parser.add_argument("--count", type=int, default=12, help="New queries PER subject (subject with fewer chunks may get less)")
    parser.add_argument("--out", default=None, help="Output queries file (default: eval/queries_{type}.json)")
    parser.add_argument("--subjects", default=None, help="Comma-separated subject codes to restrict gen to (e.g. csdl,kpdl). Default: all subjects in SUBJECTS dict.")
    args = parser.parse_args()

    subjects_filter = [s.strip() for s in args.subjects.split(",") if s.strip()] if args.subjects else None
    qfile = Path(args.out) if args.out else Path(__file__).parent / f"queries_{args.type}.json"
    asyncio.run(gen_for_type(args.type, args.count, qfile, subjects_filter))


if __name__ == "__main__":
    main()
