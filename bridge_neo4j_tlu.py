import asyncio
import hashlib
import os
import logging
from dotenv import load_dotenv
from neo4j import AsyncGraphDatabase

load_dotenv(dotenv_path="D:/Document/RAG_Learning/.env")

from lightrag.lightrag import LightRAG
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

# Enable logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
WORKING_DIR = "./tlu_workspace"
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")


# nomic-embed-text runs local via Ollama, no rate limit needed
async def nomic_embed(texts, **kwargs):
    return await ollama_embed.func(texts, embed_model="nomic-embed-text", **kwargs)


def compute_mdhash_id(content, prefix=""):
    """LightRAG's standard ID generation function"""
    return prefix + hashlib.md5(content.encode()).hexdigest()


async def run_bridge():
    logging.info(f"Initializing LightRAG at {WORKING_DIR}...")
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=nomic_embed,
        ),
    )

    await rag.initialize_storages()

    # TEST: verify Ollama embedding works
    logging.info("Testing embed with 1 entity...")
    test_result = await nomic_embed(["test entity"])
    logging.info(f"Test embed OK: shape={test_result.shape}")

    logging.info(f"Connecting to Neo4j at {NEO4J_URI}...")
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    try:
        async with driver.session(default_access_mode="READ") as session:
            # ==========================================
            # PART 1: EXTRACT AND EMBED ENTITIES (NODES)
            # ==========================================
            logging.info(
                "Step 1: Extracting Entities (Person, Document, Topic, Major, Degree)..."
            )

            entity_query = """
            MATCH (n)
            WHERE n:Person OR n:Document OR n:Topic OR n:Major OR n:Degree
            WITH n,
                CASE
                    WHEN n:Person THEN 'Person'
                    WHEN n:Document THEN 'Document'
                    WHEN n:Topic THEN 'Topic'
                    WHEN n:Major THEN 'Major'
                    WHEN n:Degree THEN 'Degree'
                 END AS entity_type,
                 CASE
                    WHEN n:Person THEN n.name
                    WHEN n:Document THEN n.id
                    WHEN n:Topic THEN n.name
                    WHEN n:Major THEN n.name
                    WHEN n:Degree THEN n.name
                    ELSE n.entity_id
                 END AS entity_name
            WHERE entity_name IS NOT NULL
            RETURN entity_name, entity_type, properties(n) AS props
            """
            result = await session.run(entity_query)

            entities_payload = {}
            async for record in result:
                name = str(record["entity_name"])
                etype = str(record["entity_type"])
                props = record["props"]

                skip_keys = {"name", "id", "entity_id", "workspace_label"}

                description_parts = []
                for k, v in props.items():
                    if k not in skip_keys:
                        description_parts.append(f"{k}: {v}")

                description_text = (
                    " ".join(description_parts) if description_parts else "N/A"
                )
                content = f"{name}\n{description_text}"
                vdb_id = compute_mdhash_id(name, prefix="ent-")

                entities_payload[vdb_id] = {
                    "entity_name": name,
                    "entity_type": etype,
                    "content": content,
                    "source_id": "TLU_GRAPH_IMPORT",
                    "description": description_text,
                }

            if entities_payload:
                logging.info(f"Found {len(entities_payload)} entities. Upserting...")
                await rag.entities_vdb.upsert(entities_payload)
                logging.info("Entities upsert complete.")
            else:
                logging.warning("No TLU entities found in Neo4j.")

            # ==================================================
            # PART 2: EXTRACT AND EMBED RELATIONSHIPS (EDGES)
            # ==================================================
            logging.info("Step 2: Extracting Relationships...")

            rel_query = """
            MATCH (a)-[r]->(b)
            WHERE (a:Person OR a:Document OR a:Topic OR a:Major OR a:Degree)
            AND (b:Person OR b:Document OR b:Topic OR b:Major OR b:Degree)
            WITH a, r, b,
                CASE WHEN a:Person THEN a.name WHEN a:Document THEN a.id WHEN a:Topic THEN a.name WHEN a:Major THEN a.name WHEN a:Degree THEN a.name ELSE a.entity_id END AS src_name,
                CASE WHEN b:Person THEN b.name WHEN b:Document THEN b.id WHEN b:Topic THEN b.name WHEN b:Major THEN b.name WHEN b:Degree THEN b.name ELSE b.entity_id END AS tgt_name,
                type(r) as rel_type
            WHERE src_name IS NOT NULL AND tgt_name IS NOT NULL
            RETURN src_name, rel_type, tgt_name, properties(r) AS props
            """
            rel_result = await session.run(rel_query)

            rels_payload = {}
            async for record in rel_result:
                src = str(record["src_name"])
                tgt = str(record["tgt_name"])
                rel_type = str(record["rel_type"])
                props = record["props"]

                description = props.get(
                    "description", f"{src} is related to {tgt} by {rel_type}"
                )
                keywords = props.get("keywords", rel_type)
                try:
                    weight = float(props.get("weight", 1.0))
                except (ValueError, TypeError):
                    weight = 1.0

                content = f"{keywords}\t{src}\n{tgt}\n{description}"
                rel_vdb_id = compute_mdhash_id(src + tgt, prefix="rel-")

                rels_payload[rel_vdb_id] = {
                    "src_id": src,
                    "tgt_id": tgt,
                    "content": content,
                    "keywords": keywords,
                    "description": description,
                    "weight": weight,
                    "source_id": "TLU_GRAPH_IMPORT",
                }

            if rels_payload:
                logging.info(f"Found {len(rels_payload)} relationships. Upserting...")
                await rag.relationships_vdb.upsert(rels_payload)
                logging.info("Relationships upsert complete.")
            else:
                logging.warning("No TLU relationships found in Neo4j.")

            # ==========================================
            # PART 3: SAVE VECTOR DB TO DISK (CRITICAL)
            # ==========================================
            logging.info("Step 3: Flushing VDB changes to disk...")
            await rag.entities_vdb.index_done_callback()
            await rag.relationships_vdb.index_done_callback()

            logging.info(
                "SUCCESS! Graph data is fully embedded and migrated to LightRAG Vector DBs."
            )

    except Exception as e:
        logging.error(f"An error occurred during migration: {str(e)}")
        raise
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(run_bridge())
