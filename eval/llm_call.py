"""LLM caller for BM25 mode in benchmark.

Calls 9router proxy (http://localhost:20128/v1) with same model as LightRAG server.
This ensures fair comparison: BM25 retrieval + same LLM = isolate retrieval quality.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Try loading from local eval/.env first, then fallback to root .env
eval_env = Path(__file__).parent / ".env"
if eval_env.exists():
    load_dotenv(eval_env)
else:
    load_dotenv(Path(__file__).parent.parent / ".env")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

LLM_HOST = os.getenv("LLM_BINDING_HOST", "http://localhost:20128/v1")
LLM_API_KEY = os.getenv("LLM_BINDING_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/google/gemma-3n-e4b-it")
TIMEOUT_SEC = 180


def build_rag_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for RAG with BM25-retrieved chunks.

    Mirrors LightRAG naive_rag_response prompt VERBATIM for fair apples-to-apples compare.
    Ensures identical LLM behavior (comprehensive, Vietnamese, markdown, citations) across all modes.
    """
    system = (
        "---Role---\n\n"
        "You are an expert AI assistant specializing in synthesizing information from a provided "
        "knowledge base. Your primary function is to answer user queries accurately by ONLY using "
        "the information within the provided **Context**.\n\n"
        "---Goal---\n\n"
        "Generate a comprehensive, well-structured answer to the user query.\n"
        "The answer must integrate relevant facts from the Document Chunks found in the **Context**.\n\n"
        "---Instructions---\n\n"
        "1. Step-by-Step Instruction:\n"
        "  - Carefully determine the user's query intent to fully understand the user's information need.\n"
        "  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of "
        "information that are directly relevant to answering the user query.\n"
        "  - Weave the extracted facts into a coherent and logical response. Your own knowledge must "
        "ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.\n"
        "  - Track the reference_id of the document chunk which directly support the facts presented "
        "in the response. Correlate reference_id with the entries in the Context to generate citations.\n"
        "  - Generate a **References** section at the end of the response. Each reference document "
        "must directly support the facts presented in the response.\n"
        "  - Do not generate anything after the reference section.\n\n"
        "2. Content & Grounding:\n"
        "  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, "
        "or infer any information not explicitly stated.\n"
        "  - If the answer cannot be found in the **Context**, state that you do not have enough "
        "information to answer. Do not attempt to guess.\n"
        "  - Provide a DETAILED and COMPREHENSIVE answer. Each key concept MUST be explained "
        "thoroughly with definitions, examples, and relationships to other concepts.\n"
        "  - Use ALL relevant information from the context. Do not skip or summarize important details.\n"
        "  - Aim for a thorough response of at least 500 words when the context provides sufficient information.\n\n"
        "3. Formatting & Language:\n"
        "  - The response MUST always be in Vietnamese.\n"
        "  - The response MUST utilize Markdown formatting for enhanced clarity and structure "
        "(e.g., headings, bold text, bullet points).\n\n"
        "4. Image Markers:\n"
        "  - The context may contain image markers in the format `[IMG_docname_N]`.\n"
        "  - You MUST preserve these markers exactly as they appear in your response.\n"
        "  - Do NOT remove, rename, translate, or modify any `[IMG_...]` marker.\n\n"
        "5. References Section Format:\n"
        "  - The References section should be under heading: `### References`\n"
        "  - Reference list entries should adhere to the format: `* [n] Document Title`.\n"
        "  - The Document Title in the citation must retain its original language.\n"
        "  - Output each citation on an individual line.\n"
        "  - Provide maximum of 5 most relevant citations.\n"
        "  - Do not generate footnotes section or any comment after the references.\n\n"
        "6. Reference Section Example:\n"
        "```\n"
        "### References\n\n"
        "- [1] Document Title One\n"
        "- [2] Document Title Two\n"
        "- [3] Document Title Three\n"
        "```"
    )

    context_parts = []
    for i, c in enumerate(chunks, 1):
        content = c.get("content", "").strip()
        if not content:
            continue
        fp = c.get("file_path", "?")
        context_parts.append(f"[{i}] (Nguồn: {fp})\n{content}")

    context_str = "\n\n---\n\n".join(context_parts)

    user = (
        f"### Context\n\n{context_str}\n\n"
        f"### Câu hỏi\n\n{query}\n\n"
        f"### Trả lời\n"
    )
    return system, user


def call_llm(query: str, chunks: list[dict]) -> dict:
    """POST to LLM, return {response, latency_sec, error, prompt_chars}."""
    import requests

    system, user = build_rag_prompt(query, chunks)

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    try:
        r = requests.post(
            f"{LLM_HOST}/chat/completions",
            json=payload,
            headers=headers,
            timeout=TIMEOUT_SEC,
        )
        r.raise_for_status()
        elapsed = time.time() - t0
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return {
            "response": content,
            "latency_sec": round(elapsed, 2),
            "error": None,
            "prompt_chars": len(system) + len(user),
        }
    except requests.HTTPError as e:
        return {
            "response": "",
            "latency_sec": round(time.time() - t0, 2),
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            "prompt_chars": 0,
        }
    except Exception as e:
        return {
            "response": "",
            "latency_sec": round(time.time() - t0, 2),
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "prompt_chars": 0,
        }


def main():
    """Smoke test."""
    import bm25_index
    query = "Hệ thống thông minh là gì?"
    chunks = bm25_index.search(query, top_k=5)
    print(f"[BM25] Retrieved {len(chunks)} chunks for: {query}")

    result = call_llm(query, chunks)
    print(f"\n[LLM] latency={result['latency_sec']}s prompt_chars={result['prompt_chars']}")
    if result["error"]:
        print(f"[ERROR] {result['error']}")
    else:
        print(f"\n[RESPONSE]\n{result['response']}")


if __name__ == "__main__":
    main()
