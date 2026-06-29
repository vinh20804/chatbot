"""Stat mode handler: route to Neo4j Text2Cypher for bibliography metadata queries.

Stat mode = bibliography statistics from TLU library Neo4j graph.
Pipeline:
  1. LLM generate Cypher from natural-language query
  2. Execute Cypher on Neo4j (TLU bibliography graph)
  3. On failure: feedback error → LLM regenerate (retry up to MAX_ATTEMPTS)
  4. Format records as facts text
  5. LLM compose natural answer using facts
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


async def handle_stat_query(
    query: str,
    knowledge_graph_inst,  # unused for stat (Neo4j path), kept for signature compat
    llm_func,
) -> str:
    """Stat: bibliography metadata via Text2Cypher → Neo4j with retry-on-error.

    Retry policy:
      - records is None (Cypher syntax/runtime error) → retry with error feedback
      - records is empty list (valid query, no matches) → return early, NOT retry
      - Cypher gen returns None (LLM/validator failure) → retry with hint

    Args:
        query: User question
        knowledge_graph_inst: NetworkX (unused — kept for kg_query signature)
        llm_func: LLM function for Cypher gen + final answer compose

    Returns:
        Natural-language answer based on Neo4j data, or fallback message.
    """
    from lightrag.text2cypher import (
        generate_cypher,
        Neo4jExecutor,
        format_cypher_results,
    )

    executor = Neo4jExecutor()
    if not executor.driver:
        executor.close()
        return "Neo4j không khả dụng. Stat mode cần Neo4j bibliography graph."

    last_err: str | None = None
    last_cypher: str | None = None

    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # Build query — original on first attempt, augmented with error context on retry.
            if attempt == 1:
                gen_input = query
            else:
                gen_input = (
                    f"{query}\n\n"
                    f"Previous Cypher attempt FAILED:\n```cypher\n{last_cypher or '(no cypher generated)'}\n```\n"
                    f"Error: {last_err}\n"
                    f"Generate a CORRECTED Cypher query."
                )
                logger.info(f"Cypher retry attempt {attempt}/{MAX_ATTEMPTS}")

            cypher = await generate_cypher(gen_input, llm_func)
            if not cypher:
                last_err = "LLM did not produce a valid Cypher query (validator rejected or empty)"
                last_cypher = None
                continue

            last_cypher = cypher
            records = executor.execute(cypher)

            if records is None:
                last_err = "Cypher execution failed (syntax error, unknown label/property, or runtime error)"
                continue

            if not records:
                # Valid Cypher, just zero matches — don't waste retries.
                return "Không tìm thấy kết quả phù hợp trong cơ sở dữ liệu bibliography."

            # Success.
            if attempt > 1:
                logger.info(f"Cypher succeeded on retry attempt {attempt}")
            facts_text = format_cypher_results(records)
            prompt = (
                f"Câu hỏi: {query}\n\n"
                f"Dữ liệu thực tế từ Neo4j bibliography:\n{facts_text}\n\n"
                f"Trả lời câu hỏi DỰA HOÀN TOÀN trên dữ liệu trên. "
                f"Nếu hỏi đếm thì cho số chính xác. Nếu hỏi liệt kê thì list ngắn gọn. "
                f"Không bịa thông tin ngoài data."
            )
            return await llm_func(prompt)

        # Exhausted all attempts.
        if last_cypher:
            return (
                f"Sau {MAX_ATTEMPTS} lần thử, không thực thi được Cypher. "
                f"Lỗi cuối: {last_err}. Cypher cuối: `{last_cypher}`"
            )
        return f"Sau {MAX_ATTEMPTS} lần thử, LLM không sinh được Cypher hợp lệ."
    finally:
        executor.close()
