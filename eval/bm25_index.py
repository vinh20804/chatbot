"""BM25 index for RAG benchmark baseline.

Build BM25Okapi index from PG chunks (lightrag_doc_chunks workspace='lightrag').
Save pickle to avoid rebuild on every run.

Usage:
  python bm25_index.py --build           # Build + save index
  python bm25_index.py --test "query"    # Test search
"""

import argparse
import os
import pickle
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 stdout (Windows console cp1252 default breaks on Vietnamese)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Try loading from local eval/.env first, then fallback to root .env
eval_env = Path(__file__).parent / ".env"
if eval_env.exists():
    load_dotenv(eval_env)
else:
    load_dotenv(Path(__file__).parent.parent / ".env")

HERE = Path(__file__).parent
INDEX_FILE = HERE / "bm25_index.pkl"


def connect_pg():
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DATABASE"),
    )


def tokenize_vi(text: str) -> list[str]:
    """Simple Vietnamese tokenization: lowercase, split on whitespace+punct.

    Keep diacritics. Strip markdown image markers (IMG_*, ![Image](...)).
    """
    if not text:
        return []
    # Strip markdown image refs
    text = re.sub(r"!\[Image\]\([^)]*\)", " ", text)
    text = re.sub(r"IMG_\w+", " ", text)
    # Lowercase + split on non-word chars (Unicode aware for VN)
    text = text.lower()
    tokens = re.findall(r"[\wÀ-ɏḀ-ỿ]+", text, re.UNICODE)
    return [t for t in tokens if len(t) > 1]


def load_chunks_from_pg():
    """Load all chunks from PG."""
    c = connect_pg()
    cur = c.cursor()
    cur.execute(
        "SELECT id, content, full_doc_id, file_path "
        "FROM lightrag_doc_chunks WHERE workspace='lightrag' "
        "ORDER BY id"
    )
    rows = cur.fetchall()
    c.close()
    chunks = []
    for row in rows:
        chunks.append({
            "id": row[0],
            "content": row[1] or "",
            "full_doc_id": row[2] or "",
            "file_path": row[3] or "",
        })
    return chunks


def build_index():
    """Build BM25 index from PG chunks, save to pickle."""
    from rank_bm25 import BM25Okapi

    print("[BM25] Loading chunks from PG...")
    chunks = load_chunks_from_pg()
    print(f"[BM25] Loaded {len(chunks)} chunks")

    print("[BM25] Tokenizing...")
    corpus_tokens = [tokenize_vi(c["content"]) for c in chunks]
    avg_len = sum(len(t) for t in corpus_tokens) / max(len(corpus_tokens), 1)
    print(f"[BM25] Avg tokens/chunk: {avg_len:.1f}")

    print("[BM25] Building BM25Okapi index...")
    bm25 = BM25Okapi(corpus_tokens)

    payload = {
        "bm25": bm25,
        "chunks": chunks,  # full metadata for retrieval
        "version": "1.0",
        "n_chunks": len(chunks),
    }
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"[BM25] Saved → {INDEX_FILE} ({INDEX_FILE.stat().st_size / 1024:.0f} KB)")


def load_index():
    """Load pickled index. Rebuild if missing."""
    if not INDEX_FILE.exists():
        print("[BM25] Index not found, building...")
        build_index()
    with open(INDEX_FILE, "rb") as f:
        return pickle.load(f)


def search(query: str, top_k: int = 10, index=None) -> list[dict]:
    """Return top-K chunks with score.

    Each result: {id, content, full_doc_id, file_path, score}
    """
    if index is None:
        index = load_index()
    bm25 = index["bm25"]
    chunks = index["chunks"]

    query_tokens = tokenize_vi(query)
    if not query_tokens:
        return []

    scores = bm25.get_scores(query_tokens)
    # Top-K indices
    import numpy as np
    top_idx = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] <= 0:
            continue
        c = chunks[int(idx)].copy()
        c["score"] = float(scores[idx])
        results.append(c)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build/rebuild index")
    parser.add_argument("--test", type=str, help="Test query string")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.build:
        build_index()
        return

    if args.test:
        sys.stdout.reconfigure(encoding="utf-8")
        results = search(args.test, top_k=args.top_k)
        print(f"\n[Query] {args.test}")
        print(f"[Results] {len(results)} chunks\n")
        for i, r in enumerate(results, 1):
            preview = r["content"][:150].replace("\n", " ")
            print(f"#{i} score={r['score']:.3f} | {r['file_path'][:40]}")
            print(f"   {preview}...")
            print()


if __name__ == "__main__":
    main()
