"""DATN Evaluation Script: Evaluate LightRAG answers using the 4 thesis metrics:
- Tính toàn diện (Comprehensiveness) [1-5]
- Tính đa dạng (Diversity) [1-5]
- Tính hữu ích (Empowerment) [1-5]
- Mức độ ảo giác (Hallucination Rate) [0-1]

Uses Ollama (via configured model in .env or fallback) to generate and judge.
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

# Load environment variables
eval_env = ROOT / "eval" / ".env"
if eval_env.exists():
    load_dotenv(eval_env)
else:
    load_dotenv(ROOT / ".env")

# Force using local qwen2.5:1.5b model for local execution
os.environ["LLM_MODEL"] = "qwen2.5:1.5b"

import httpx
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc

# TLU-specific & Common knowledge queries for evaluation
QUERIES = [
    # --- Common knowledge ---
    "Laravel sử dụng Eloquent ORM để làm gì?",
    "Cache L1 và L2 khác nhau thế nào?",
    "Booth Algorithm hoạt động ra sao?",
    "MVC gồm những thành phần nào trong Laravel?",
    "Register trong CPU dùng để làm gì?",
    # --- TLU-specific (Should have high accuracy from tlu_workspace) ---
    "Trong tài liệu TLU, Cache Replacement Algorithm nói về những thuật toán nào?",
    "Slide TLU dạy về MVC pattern dùng framework gì làm ví dụ?",
    "Trong slide TLU, các loại địa chỉ trong instruction là gì?",
    "Trong tài liệu TLU, Eloquent có những phương thức nào được giới thiệu?",
    "Slide TLU nêu các thành phần chính nào của một CPU?",
]

SYSTEM_JUDGE_PROMPT = """Bạn là một chuyên gia đánh giá hệ thống Hỏi - Đáp (RAG) cho Đồ án tốt nghiệp Khoa CNTT - Đại học Thủy Lợi.
Hãy phân tích câu hỏi (Query), câu trả lời (Answer), và các đoạn ngữ cảnh trích xuất từ tài liệu (Chunks) để đánh giá câu trả lời theo 4 tiêu chí nghiêm ngặt sau:

1. **Tính toàn diện (Comprehensiveness)** (Thang điểm 1-5): Đánh giá mức độ bao phủ thông tin. Câu trả lời có giải quyết được mọi khía cạnh của câu hỏi đầu vào hay không?
   - 5: Trả lời hoàn hảo mọi khía cạnh, chi tiết và đầy đủ.
   - 1: Trả lời thiếu sót hầu hết nội dung chính.

2. **Tính đa dạng (Diversity)** (Thang điểm 1-5): Đánh giá khả năng tổng hợp chéo thông tin. Câu trả lời có cung cấp được nhiều góc nhìn, phương pháp hoặc lập luận được trích xuất từ các ngữ cảnh khác nhau trong cơ sở dữ liệu hay không?
   - 5: Tổng hợp thông tin rất tốt từ nhiều phần ngữ cảnh khác nhau, đa chiều.
   - 1: Thông tin đơn điệu, chỉ lặp lại một ý hoặc một đoạn.

3. **Tính hữu ích (Empowerment)** (Thang điểm 1-5): Đánh giá giá trị thực tiễn. Câu trả lời có thực sự giúp người dùng hiểu rõ bản chất của vấn đề, hay chỉ là những đoạn văn bản chắp vá vô nghĩa?
   - 5: Rất hữu ích, rõ ràng, giúp người đọc nắm bắt được cốt lõi vấn đề và ứng dụng được.
   - 1: Văn bản chắp vá, rời rạc, không mang lại giá trị kiến thức.

4. **Mức độ ảo giác (Hallucination Rate)** (Thang điểm 0-1): Đánh giá tính xác thực. Tỷ lệ các khẳng định (claims) trong câu trả lời KHÔNG được hỗ trợ bởi các Chunks. 
   - 0.0: Không có ảo giác (mọi khẳng định đều được hỗ trợ bởi Chunks).
   - 1.0: Ảo giác hoàn toàn (câu trả lời tự bịa thông tin hoặc dùng kiến thức ngoài hoàn toàn mà không có trong Chunks).
   - Quy tắc tính: (Số claims không có bằng chứng trong Chunks) / (Tổng số claims có trong Answer).

Hãy trả về kết quả dưới định dạng JSON duy nhất như sau (không kèm mã markdown ```json):
{
  "comprehensiveness": <int 1-5>,
  "comprehensiveness_reason": "<lý do ngắn gọn bằng tiếng Việt>",
  "diversity": <int 1-5>,
  "diversity_reason": "<lý do ngắn gọn bằng tiếng Việt>",
  "empowerment": <int 1-5>,
  "empowerment_reason": "<lý do ngắn gọn bằng tiếng Việt>",
  "hallucination_rate": <float 0.0-1.0>,
  "hallucination_rate_reason": "<lý do ngắn gọn bằng tiếng Việt>"
}
"""

async def call_ollama_judge(client: httpx.AsyncClient, host: str, model: str, query: str, answer: str, chunks: list[str]) -> dict:
    chunks_text = "\n\n".join(f"[Chunk {i+1}]: {c}" for i, c in enumerate(chunks))
    prompt = f"### Query:\n{query}\n\n### Answer:\n{answer}\n\n### Chunks:\n{chunks_text}\n"
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_JUDGE_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "options": {
            "temperature": 0.0
        },
        "format": "json"
    }
    
    try:
        resp = await client.post(f"{host}/api/chat", json=payload, timeout=60.0)
        if resp.status_code == 200:
            content = resp.json()["message"]["content"]
            return json.loads(content)
        else:
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        return {"error": str(e)}

async def main():
    working_dir = os.environ.get("WORKING_DIR", "./tlu_workspace")
    if not os.path.isabs(working_dir):
        working_dir = str((ROOT / working_dir).resolve())
        
    print("=" * 80)
    print("📊 TLU-CHATBOT GRADUATION THESIS EVALUATION (4 METRICS)")
    print("=" * 80)
    print(f"Working Dir: {working_dir}")
    
    # Check if tlu_workspace exists
    if not Path(working_dir).exists():
        print(f"[ERR] working_dir {working_dir} does not exist. Please check your .env configuration.")
        return

    # Initialize RAG
    print("[init] Initializing LightRAG...")
    from functools import partial
    from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    from lightrag.utils import EmbeddingFunc
    try:
        rag = LightRAG(
            working_dir=working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name="qwen2.5:1.5b",
            llm_model_kwargs={
                "host": os.environ.get("LLM_BINDING_HOST", "http://localhost:11434"),
                "options": {"num_ctx": 4096},
                "timeout": 120,
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=int(os.environ.get("EMBEDDING_DIM", "768")),
                max_token_size=int(os.environ.get("MAX_EMBED_TOKENS", "8192")),
                func=partial(
                    ollama_embed.func,
                    embed_model=os.environ.get("EMBEDDING_MODEL", "nomic-embed-text:latest"),
                    host=os.environ.get("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
                ),
            ),
            kv_storage=os.environ.get("LIGHTRAG_KV_STORAGE", "JsonKVStorage"),
            doc_status_storage=os.environ.get("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage"),
            vector_storage=os.environ.get("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
            graph_storage=os.environ.get("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage"),
        )
        await rag.initialize_storages()
    except Exception as e:
        print(f"[ERR] Failed to initialize LightRAG: {e}")
        return

    ollama_host = os.environ.get("LLM_BINDING_HOST", "http://localhost:11434")
    ollama_model = os.environ.get("LLM_MODEL", "qwen2.5:7b")
    
    print(f"Ollama Host: {ollama_host}")
    print(f"Ollama Model: {ollama_model}")
    
    # Verify Ollama is running and has the model
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{ollama_host}/api/tags")
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                print(f"Installed models: {models}")
                if ollama_model not in models and (ollama_model + ":latest") not in models:
                    print(f"\n[WARNING] Model '{ollama_model}' is not installed in Ollama.")
                    print(f"To run this evaluation, please pull the model first:")
                    print(f"  ollama pull {ollama_model}")
                    print("\nAttempting to continue. If queries fail, please pull the model.")
            else:
                print(f"[WARN] Cannot connect to Ollama tags endpoint: HTTP {resp.status_code}")
        except Exception as e:
            print(f"[WARN] Cannot connect to Ollama at {ollama_host}: {e}")
            print("Please ensure Ollama is running.")
            return

        results = []
        print("\n[running] Starting evaluation across 10 queries...")
        for idx, query in enumerate(QUERIES, 1):
            print(f"\n[{idx}/{len(QUERIES)}] Query: {query}")
            for mode in ["naive", "hybrid"]:
                print(f"  → mode={mode}...", end="", flush=True)
                try:
                    # Query LightRAG
                    from lightrag import QueryParam
                    t0 = time.time()
                    ans = await rag.aquery(query, param=QueryParam(mode=mode))
                    elapsed = time.time() - t0
                    
                    # Fetch retrieved chunks for context
                    # Use internal retrieval to get chunks used
                    retrieved_chunks = []
                    if mode == "naive":
                        # Naive retrieval helper
                        chunks = await rag.chunks_vdb.query(query, top_k=6)
                        retrieved_chunks = [c.get("content", "") for c in chunks if c]
                    else:
                        # Hybrid retrieval helper
                        chunks = await rag.chunks_vdb.query(query, top_k=4)
                        retrieved_chunks = [c.get("content", "") for c in chunks if c]
                        
                    print(f" Answered ({elapsed:.1f}s). Judging...", end="", flush=True)
                    
                    # Call judge
                    judge_res = await call_ollama_judge(client, ollama_host, ollama_model, query, ans, retrieved_chunks)
                    
                    if "error" in judge_res:
                        print(f" Judge Failed: {judge_res['error']}")
                        entry = {
                            "query": query,
                            "mode": mode,
                            "answer": ans,
                            "latency_sec": elapsed,
                            "error": judge_res["error"]
                        }
                    else:
                        print(" OK.")
                        entry = {
                            "query": query,
                            "mode": mode,
                            "answer": ans,
                            "latency_sec": elapsed,
                            "comprehensiveness": judge_res.get("comprehensiveness"),
                            "comprehensiveness_reason": judge_res.get("comprehensiveness_reason"),
                            "diversity": judge_res.get("diversity"),
                            "diversity_reason": judge_res.get("diversity_reason"),
                            "empowerment": judge_res.get("empowerment"),
                            "empowerment_reason": judge_res.get("empowerment_reason"),
                            "hallucination_rate": judge_res.get("hallucination_rate"),
                            "hallucination_rate_reason": judge_res.get("hallucination_rate_reason"),
                        }
                except Exception as e:
                    print(f" Failed: {e}")
                    entry = {
                        "query": query,
                        "mode": mode,
                        "error": str(e)
                    }
                
                results.append(entry)
                await asyncio.sleep(0.5)

        # Save results
        out_file = ROOT / "eval_datn_results.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
            
        print("\n" + "=" * 80)
        print("📊 AGGREGATE RESULTS")
        print("=" * 80)
        for mode in ["naive", "hybrid"]:
            mode_entries = [r for r in results if r.get("mode") == mode and "error" not in r]
            if not mode_entries:
                print(f"Mode {mode}: No successful results.")
                continue
                
            comp = [r["comprehensiveness"] for r in mode_entries if r.get("comprehensiveness") is not None]
            div = [r["diversity"] for r in mode_entries if r.get("diversity") is not None]
            emp = [r["empowerment"] for r in mode_entries if r.get("empowerment") is not None]
            hall = [r["hallucination_rate"] for r in mode_entries if r.get("hallucination_rate") is not None]
            lat = [r["latency_sec"] for r in mode_entries]
            
            avg_comp = sum(comp) / len(comp) if comp else 0
            avg_div = sum(div) / len(div) if div else 0
            avg_emp = sum(emp) / len(emp) if emp else 0
            avg_hall = sum(hall) / len(hall) if hall else 0
            avg_lat = sum(lat) / len(lat) if lat else 0
            
            print(f"Mode: {mode}")
            print(f"  - Avg Comprehensiveness (Tính toàn diện) [1-5]: {avg_comp:.2f}")
            print(f"  - Avg Diversity (Tính đa dạng) [1-5]:           {avg_div:.2f}")
            print(f"  - Avg Empowerment (Tính hữu ích) [1-5]:         {avg_emp:.2f}")
            print(f"  - Avg Hallucination Rate (Mức độ ảo giác) [0-1]: {avg_hall:.2f}")
            print(f"  - Avg Latency (Thời gian phản hồi):             {avg_lat:.2f}s")
            print("-" * 50)
            
        print(f"\n[done] Detailed results saved to: {out_file}")

    await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())
