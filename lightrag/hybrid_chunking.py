"""Custom chunking function using Docling's HybridChunker.

Replaces LightRAG's default token-based chunking with structure-aware
chunking that respects document headings, tables, and code blocks.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def build_hybrid_chunking_func(
    max_tokens: int = 512,
    embed_model_name: str = "microsoft/harrier-oss-v1-270m",
):
    """Factory that returns a chunking_func compatible with LightRAG's signature.

    Uses HuggingFace tokenizer of the embedding model to keep chunk-size
    accounting in sync with the embedder (avoids Tiktoken/WordPiece mismatch
    that caused "Token indices sequence length is longer than max" warnings).

    The returned function matches LightRAG's chunking_func signature:
        (tokenizer, content, split_by_character, split_by_character_only,
        chunk_overlap_token_size, chunk_token_size) -> List[Dict]

    Each dict has keys: tokens, content, chunk_order_index
    """
    from docling_core.transforms.chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import MarkdownParams
    from docling_core.types.doc.base import ImageRefMode

    class ImageRefSerializerProvider(ChunkingSerializerProvider):
        """Serializer that keeps image references in chunk text for UI rendering."""
        def get_serializer(self, doc):
            return ChunkingDocSerializer(
                doc=doc,
                params=MarkdownParams(
                    image_mode=ImageRefMode.REFERENCED,
                    escape_underscores=False,
                    escape_html=False,
                ),
            )

    # Sync tokenizer with embedding model to prevent token-count drift.
    hf_tokenizer = HuggingFaceTokenizer.from_pretrained(
        model_name=embed_model_name,
        max_tokens=max_tokens,
    )
    logger.info(
        f"HybridChunker using HF tokenizer '{embed_model_name}' max_tokens={max_tokens}"
    )

    chunker = HybridChunker(
        tokenizer=hf_tokenizer,
        merge_peers=True,
        repeat_table_header=True,
        serializer_provider=ImageRefSerializerProvider(),
    )

    def hybrid_chunking_func(
        tokenizer,
        content: str,
        split_by_character: Optional[str] = None,
        split_by_character_only: bool = False,
        chunk_overlap_token_size: int = 100,
        chunk_token_size: int = 1200,
    ) -> List[Dict[str, Any]]:
        """LightRAG-compatible chunking function using HybridChunker."""
        from lightrag.api.routers.document_routes import get_cached_docling_document

        doc = get_cached_docling_document(content)

        if doc is not None:
            logger.info("Using HybridChunker for structured document")
            doc_chunks = list(chunker.chunk(dl_doc=doc))

            result = []
            for i, chunk in enumerate(doc_chunks):
                chunk_text = chunker.contextualize(chunk)
                token_count = len(hf_tokenizer.encode(chunk_text))
                result.append(
                    {
                        "tokens": token_count,
                        "content": chunk_text,
                        "chunk_order_index": i,
                    }
                )

            logger.info(f"HybridChunker produced {len(result)} chunks")
            return result
        else:
            logger.info(
                "No cached DoclingDocument found, falling back to token-based chunking"
            )
            from lightrag.operate import chunking_by_token_size

            return chunking_by_token_size(
                tokenizer,
                content,
                split_by_character,
                split_by_character_only,
                chunk_overlap_token_size,
                chunk_token_size,
            )

    return hybrid_chunking_func
