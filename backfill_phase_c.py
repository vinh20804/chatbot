"""Phase C backfill: extract entities/relations for old docs ingested before Phase E wiring.

Strategy:
- Fetch all doc IDs that have chunks but no full_entities row.
- For each doc, load chunks from lightrag_doc_chunks.
- Call rag.extract_and_merge_chunks(chunks, doc_id, persist=True).
- Per-doc flush = checkpoint. Crash-safe: idempotent skip on resume.

Prereq: backend server SHOULD BE STOPPED to avoid concurrent writes.

Run:
    venv/Scripts/python.exe backfill_phase_c.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

# Try loading from local eval/.env first, then fallback to root .env
eval_env = ROOT / "eval" / ".env"
if eval_env.exists():
    load_dotenv(eval_env)
else:
    load_dotenv(ROOT / ".env")

import asyncpg

import numpy as np
from sentence_transformers import SentenceTransformer

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc


PG_DSN = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
    f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DATABASE']}"
)
WORKSPACE = os.environ.get("POSTGRES_WORKSPACE", "lightrag")

# Backfill uses 9router pool (smart models for extraction).
# Reads ROUTER_* so it works when server's LLM_BINDING_* points to Ollama.
LLM_HOST = os.environ.get("ROUTER_HOST") or os.environ["LLM_BINDING_HOST"]
LLM_KEY = os.environ.get("ROUTER_API_KEY") or os.environ["LLM_BINDING_API_KEY"]
EMBED_MODEL = os.environ["EMBEDDING_MODEL"]
EMBED_DIM = int(os.environ.get("EMBEDDING_DIM", "640"))

# Round-robin pool — 3 quota pool độc lập (Copilot, Google Cloud, KiloCode).
# All non-reasoning OR reasoning-with-clean-content (verified via probe).
# Dropped: nvidia/abacusai/dracarys (400 Bad Request "Function id DEACTIVATED" on backfill run).
LLM_POOL = [
    "gh/gpt-4o-mini",                                         # Copilot
    "gc/gemini-2.5-flash",                                    # Google Cloud
    "kc/stepfun/step-3.5-flash:free",                         # KiloCode (OpenRouter)
    "kc/poolside/laguna-m.1:free",                            # KiloCode (OpenRouter)
]
_rr_counter = 0
_pool_lock = asyncio.Lock()
# Circuit breaker: when a model fails repeatedly, mark it cooled-down for N seconds.
# Avoids burning 360s timeout per chunk on a known-broken provider.
_cooldown: dict[str, float] = {}  # model -> unix_timestamp_when_usable_again
COOLDOWN_SECONDS = 120

# Lazy-load HF model (mirror server lightrag_server.py:852-871)
_hf_model = None


def _get_hf_model():
    global _hf_model
    if _hf_model is None:
        print(f"[backfill] loading HF embedding model: {EMBED_MODEL}")
        _hf_model = SentenceTransformer(EMBED_MODEL)
    return _hf_model


def _is_cooled(model: str) -> bool:
    """True if model still in cooldown window."""
    now = time.time()
    return _cooldown.get(model, 0) > now


def _mark_cooldown(model: str) -> None:
    _cooldown[model] = time.time() + COOLDOWN_SECONDS


async def _next_model() -> str:
    """Round-robin pick next model, skip cooled-down ones."""
    global _rr_counter
    async with _pool_lock:
        n = len(LLM_POOL)
        for _ in range(n):
            model = LLM_POOL[_rr_counter % n]
            _rr_counter += 1
            if not _is_cooled(model):
                return model
        # All cooled — pick first anyway (let it retry, may have recovered).
        return LLM_POOL[_rr_counter % n]


async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Round-robin across LLM_POOL with circuit breaker.
    On failure: mark model cooled, walk pool. After all fail, raise.
    """
    kwargs.pop("model", None)
    last_err: Exception | None = None
    n = len(LLM_POOL)
    start_model = await _next_model()
    candidates = [start_model] + [
        LLM_POOL[(LLM_POOL.index(start_model) + i) % n] for i in range(1, n)
    ]
    for model in candidates:
        if _is_cooled(model):
            continue  # skip without burning a request
        try:
            return await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key=LLM_KEY,
                base_url=LLM_HOST,
                **kwargs,
            )
        except Exception as e:
            last_err = e
            err_str = str(e)[:200]
            _mark_cooldown(model)
            print(f"  [llm-fail] {model} (cooldown {COOLDOWN_SECONDS}s): {err_str}")
            continue
    raise RuntimeError(f"All {n} providers failed/cooled. Last: {last_err}")


async def _hf_embed(texts):
    model = _get_hf_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.array(embeddings)


embedding_func = EmbeddingFunc(
    embedding_dim=EMBED_DIM,
    func=_hf_embed,
    model_name=EMBED_MODEL,  # required for VDB table suffix isolation
)


async def fetch_pending_docs(conn) -> list[str]:
    """Docs that have chunks but no full_entities row."""
    rows = await conn.fetch(
        """
        SELECT DISTINCT c.full_doc_id
        FROM lightrag_doc_chunks c
        LEFT JOIN lightrag_full_entities e
          ON e.workspace = c.workspace AND e.id = c.full_doc_id
        WHERE c.workspace = $1 AND e.id IS NULL AND c.full_doc_id IS NOT NULL
        ORDER BY c.full_doc_id
        """,
        WORKSPACE,
    )
    return [r["full_doc_id"] for r in rows]


async def fetch_chunks_for_doc(conn, doc_id: str) -> dict:
    rows = await conn.fetch(
        """
        SELECT id, content, tokens, chunk_order_index, full_doc_id, file_path
        FROM lightrag_doc_chunks
        WHERE workspace = $1 AND full_doc_id = $2
        ORDER BY chunk_order_index
        """,
        WORKSPACE,
        doc_id,
    )
    return {
        r["id"]: {
            "content": r["content"],
            "tokens": r["tokens"],
            "chunk_order_index": r["chunk_order_index"],
            "full_doc_id": r["full_doc_id"],
            "file_path": r["file_path"] or doc_id,
        }
        for r in rows
    }


async def main():
    working_dir = os.environ.get("WORKING_DIR", "./rag_storage")
    if not os.path.isabs(working_dir):
        working_dir = str((ROOT / working_dir).resolve())
    print(f"[backfill] working_dir={working_dir}")
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_func,
        embedding_func=embedding_func,
        kv_storage=os.environ.get("LIGHTRAG_KV_STORAGE", "PGKVStorage"),
        doc_status_storage=os.environ.get("LIGHTRAG_DOC_STATUS_STORAGE", "PGDocStatusStorage"),
        vector_storage=os.environ.get("LIGHTRAG_VECTOR_STORAGE", "PGVectorStorage"),
        graph_storage=os.environ.get("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    pg_conn = await asyncpg.connect(PG_DSN)

    pending = await fetch_pending_docs(pg_conn)
    print(f"[backfill] {len(pending)} docs need entity/relation extract")

    for i, doc_id in enumerate(pending, 1):
        chunks = await fetch_chunks_for_doc(pg_conn, doc_id)
        if not chunks:
            print(f"[{i}/{len(pending)}] {doc_id}: SKIP (no chunks)")
            continue

        print(f"[{i}/{len(pending)}] {doc_id}: {len(chunks)} chunks -> extract")
        t0 = time.time()
        try:
            await rag.extract_and_merge_chunks(
                chunks=chunks,
                doc_id=doc_id,
                file_path=next(iter(chunks.values()))["file_path"],
                persist=True,
            )
            elapsed = time.time() - t0
            print(f"  [OK] done in {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [FAIL] FAIL after {elapsed:.1f}s: {e}")
            # continue next doc

        # Politeness sleep
        await asyncio.sleep(1)

    await pg_conn.close()
    await rag.finalize_storages()
    print("[backfill] complete")


if __name__ == "__main__":
    asyncio.run(main())
