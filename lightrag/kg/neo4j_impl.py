import os
import re
from dataclasses import dataclass
from typing import final
import configparser


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import logging
from ..utils import logger
from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
from ..kg.shared_storage import get_data_init_lock
import pipmaster as pm

if not pm.is_installed("neo4j"):
    pm.install("neo4j")

from neo4j import (  # type: ignore
    AsyncGraphDatabase,
    exceptions as neo4jExceptions,
    AsyncDriver,
    AsyncManagedTransaction,
)

from dotenv import load_dotenv

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


# Set neo4j logger level to ERROR to suppress warning logs
logging.getLogger("neo4j").setLevel(logging.ERROR)


READ_RETRY_EXCEPTIONS = (
    neo4jExceptions.ServiceUnavailable,
    neo4jExceptions.TransientError,
    neo4jExceptions.SessionExpired,
    ConnectionResetError,
    OSError,
    AttributeError,
)

READ_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(READ_RETRY_EXCEPTIONS),
    reraise=True,
)


@final
@dataclass
class Neo4JStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func, workspace=None):
        # Read env and override the arg if present
        neo4j_workspace = os.environ.get("NEO4J_WORKSPACE")
        original_workspace = workspace  # Save original value for logging
        if neo4j_workspace and neo4j_workspace.strip():
            workspace = neo4j_workspace

        # Default to 'base' when both arg and env are empty
        if not workspace or not str(workspace).strip():
            workspace = "base"

        super().__init__(
            namespace=namespace,
            workspace=workspace,
            global_config=global_config,
            embedding_func=embedding_func,
        )

        # Log after super().__init__() to ensure self.workspace is initialized
        if neo4j_workspace and neo4j_workspace.strip():
            logger.info(
                f"Using NEO4J_WORKSPACE environment variable: '{neo4j_workspace}' (overriding '{original_workspace}/{namespace}')"
            )

        self._driver = None

    def _get_workspace_label(self) -> str:
        """Return workspace label (guaranteed non-empty during initialization)"""
        return self.workspace

    def _normalize_index_suffix(self, workspace_label: str) -> str:
        """Normalize workspace label for safe use in index names."""
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", workspace_label).strip("_")
        if not normalized:
            normalized = "base"
        if not re.match(r"[A-Za-z_]", normalized[0]):
            normalized = f"ws_{normalized}"
        return normalized

    def _get_fulltext_index_name(self, workspace_label: str) -> str:
        """Return a full-text index name derived from the normalized workspace label."""
        suffix = self._normalize_index_suffix(workspace_label)
        return f"entity_id_fulltext_idx_{suffix}"

    def _is_chinese_text(self, text: str) -> bool:
        """Check if text contains Chinese/CJK characters.

        Covers:
        - CJK Unified Ideographs (U+4E00-U+9FFF)
        - CJK Extension A (U+3400-U+4DBF)
        - CJK Compatibility Ideographs (U+F900-U+FAFF)
        - CJK Extension B-F (U+20000-U+2FA1F) - supplementary planes
        """
        cjk_pattern = re.compile(
            r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|[\U00020000-\U0002fa1f]"
        )
        return bool(cjk_pattern.search(text))

    async def initialize(self):
        async with get_data_init_lock():
            URI = os.environ.get("NEO4J_URI", config.get("neo4j", "uri", fallback=None))
            USERNAME = os.environ.get(
                "NEO4J_USERNAME", config.get("neo4j", "username", fallback=None)
            )
            PASSWORD = os.environ.get(
                "NEO4J_PASSWORD", config.get("neo4j", "password", fallback=None)
            )
            MAX_CONNECTION_POOL_SIZE = int(
                os.environ.get(
                    "NEO4J_MAX_CONNECTION_POOL_SIZE",
                    config.get("neo4j", "connection_pool_size", fallback=100),
                )
            )
            CONNECTION_TIMEOUT = float(
                os.environ.get(
                    "NEO4J_CONNECTION_TIMEOUT",
                    config.get("neo4j", "connection_timeout", fallback=30.0),
                ),
            )
            CONNECTION_ACQUISITION_TIMEOUT = float(
                os.environ.get(
                    "NEO4J_CONNECTION_ACQUISITION_TIMEOUT",
                    config.get(
                        "neo4j", "connection_acquisition_timeout", fallback=30.0
                    ),
                ),
            )
            MAX_TRANSACTION_RETRY_TIME = float(
                os.environ.get(
                    "NEO4J_MAX_TRANSACTION_RETRY_TIME",
                    config.get("neo4j", "max_transaction_retry_time", fallback=30.0),
                ),
            )
            MAX_CONNECTION_LIFETIME = float(
                os.environ.get(
                    "NEO4J_MAX_CONNECTION_LIFETIME",
                    config.get("neo4j", "max_connection_lifetime", fallback=300.0),
                ),
            )
            LIVENESS_CHECK_TIMEOUT = float(
                os.environ.get(
                    "NEO4J_LIVENESS_CHECK_TIMEOUT",
                    config.get("neo4j", "liveness_check_timeout", fallback=30.0),
                ),
            )
            KEEP_ALIVE = os.environ.get(
                "NEO4J_KEEP_ALIVE",
                config.get("neo4j", "keep_alive", fallback="true"),
            ).lower() in ("true", "1", "yes", "on")
            DATABASE = os.environ.get(
                "NEO4J_DATABASE", re.sub(r"[^a-zA-Z0-9-]", "-", self.namespace)
            )
            """The default value approach for the DATABASE is only intended to maintain compatibility with legacy practices."""

            self._driver: AsyncDriver = AsyncGraphDatabase.driver(
                URI,
                auth=(USERNAME, PASSWORD),
                max_connection_pool_size=MAX_CONNECTION_POOL_SIZE,
                connection_timeout=CONNECTION_TIMEOUT,
                connection_acquisition_timeout=CONNECTION_ACQUISITION_TIMEOUT,
                max_transaction_retry_time=MAX_TRANSACTION_RETRY_TIME,
                max_connection_lifetime=MAX_CONNECTION_LIFETIME,
                liveness_check_timeout=LIVENESS_CHECK_TIMEOUT,
                keep_alive=KEEP_ALIVE,
            )

            # Try to connect to the database and create it if it doesn't exist
            for database in (DATABASE, None):
                self._DATABASE = database
                connected = False

                try:
                    async with self._driver.session(database=database) as session:
                        try:
                            result = await session.run("MATCH (n) RETURN n LIMIT 0")
                            await result.consume()  # Ensure result is consumed
                            logger.info(
                                f"[{self.workspace}] Connected to {database} at {URI}"
                            )
                            connected = True
                        except neo4jExceptions.ServiceUnavailable as e:
                            logger.error(
                                f"[{self.workspace}] "
                                + f"Database {database} at {URI} is not available"
                            )
                            raise e
                except neo4jExceptions.AuthError as e:
                    logger.error(
                        f"[{self.workspace}] Authentication failed for {database} at {URI}"
                    )
                    raise e
                except neo4jExceptions.ClientError as e:
                    if e.code == "Neo.ClientError.Database.DatabaseNotFound":
                        logger.info(
                            f"[{self.workspace}] "
                            + f"Database {database} at {URI} not found. Try to create specified database."
                        )
                        try:
                            async with self._driver.session() as session:
                                result = await session.run(
                                    f"CREATE DATABASE `{database}` IF NOT EXISTS"
                                )
                                await result.consume()  # Ensure result is consumed
                                logger.info(
                                    f"[{self.workspace}] "
                                    + f"Database {database} at {URI} created"
                                )
                                connected = True
                        except (
                            neo4jExceptions.ClientError,
                            neo4jExceptions.DatabaseError,
                        ) as e:
                            if (
                                e.code
                                == "Neo.ClientError.Statement.UnsupportedAdministrationCommand"
                            ) or (
                                e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                            ):
                                if database is not None:
                                    logger.warning(
                                        f"[{self.workspace}] This Neo4j instance does not support creating databases. Try to use Neo4j Desktop/Enterprise version or DozerDB instead. Fallback to use the default database."
                                    )
                            if database is None:
                                logger.error(
                                    f"[{self.workspace}] Failed to create {database} at {URI}"
                                )
                                raise e

                if connected:
                    workspace_label = self._get_workspace_label()
                    # Create B-Tree index for entity_id for faster lookups
                    try:
                        async with self._driver.session(database=database) as session:
                            await session.run(
                                f"CREATE INDEX IF NOT EXISTS FOR (n:`{workspace_label}`) ON (n.entity_id)"
                            )
                            logger.info(
                                f"[{self.workspace}] Ensured B-Tree index on entity_id for {workspace_label} in {database}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"[{self.workspace}] Failed to create B-Tree index: {str(e)}"
                        )

                    # Create full-text index for entity_id for faster text searches
                    await self._create_fulltext_index(
                        self._driver, self._DATABASE, workspace_label
                    )
                    break

    async def _create_fulltext_index(
        self, driver: AsyncDriver, database: str, workspace_label: str
    ):
        """Create a full-text index on the entity_id property with Chinese tokenizer support."""
        index_name = self._get_fulltext_index_name(workspace_label)
        legacy_index_name = "entity_id_fulltext_idx"
        try:
            async with driver.session(database=database) as session:
                # Check if the full-text index exists and get its configuration
                check_index_query = "SHOW FULLTEXT INDEXES"
                result = await session.run(check_index_query)
                indexes = await result.data()
                await result.consume()

                existing_index = None
                legacy_index = None
                for idx in indexes:
                    if idx["name"] == index_name:
                        existing_index = idx
                    elif idx["name"] == legacy_index_name:
                        legacy_index = idx
                    # Break early if we found both indexes
                    if existing_index and legacy_index:
                        break

                # Handle legacy index migration
                if legacy_index and not existing_index:
                    logger.info(
                        f"[{self.workspace}] Found legacy index '{legacy_index_name}'. Migrating to '{index_name}'."
                    )
                    try:
                        # Drop the legacy index (use IF EXISTS for safety)
                        drop_query = f"DROP INDEX {legacy_index_name} IF EXISTS"
                        result = await session.run(drop_query)
                        await result.consume()
                        logger.info(
                            f"[{self.workspace}] Dropped legacy index '{legacy_index_name}'"
                        )
                    except Exception as drop_error:
                        logger.warning(
                            f"[{self.workspace}] Failed to drop legacy index: {str(drop_error)}"
                        )

                # Check if index exists and is online
                if existing_index:
                    index_state = existing_index.get("state", "UNKNOWN")
                    logger.info(
                        f"[{self.workspace}] Found existing index '{index_name}' with state: {index_state}"
                    )

                    if index_state == "ONLINE":
                        logger.info(
                            f"[{self.workspace}] Full-text index '{index_name}' already exists and is online. Skipping recreation."
                        )
                        return
                    else:
                        logger.warning(
                            f"[{self.workspace}] Existing index '{index_name}' is not online (state: {index_state}). Will recreate."
                        )
                else:
                    logger.info(
                        f"[{self.workspace}] No existing index '{index_name}' found. Creating new index."
                    )

                # Create or recreate the index if needed
                needs_recreation = (
                    existing_index is not None
                    and existing_index.get("state") != "ONLINE"
                )
                needs_creation = existing_index is None

                if needs_recreation or needs_creation:
                    # Drop existing index if it needs recreation (use IF EXISTS for safety)
                    if needs_recreation:
                        try:
                            drop_query = f"DROP INDEX {index_name} IF EXISTS"
                            result = await session.run(drop_query)
                            await result.consume()
                            logger.info(
                                f"[{self.workspace}] Dropped existing index '{index_name}'"
                            )
                        except Exception as drop_error:
                            logger.warning(
                                f"[{self.workspace}] Failed to drop existing index: {str(drop_error)}"
                            )

                    # Create new index with CJK analyzer
                    logger.info(
                        f"[{self.workspace}] Creating full-text index '{index_name}' with Chinese tokenizer support."
                    )

                    try:
                        create_index_query = f"""
                        CREATE FULLTEXT INDEX {index_name}
                        FOR (n:`{workspace_label}`) ON EACH [n.entity_id]
                        OPTIONS {{
                            indexConfig: {{
                                `fulltext.analyzer`: 'cjk',
                                `fulltext.eventually_consistent`: true
                            }}
                        }}
                        """
                        result = await session.run(create_index_query)
                        await result.consume()
                        logger.info(
                            f"[{self.workspace}] Successfully created full-text index '{index_name}' with CJK analyzer."
                        )
                    except Exception as cjk_error:
                        # Fallback to standard analyzer if CJK is not supported
                        logger.warning(
                            f"[{self.workspace}] CJK analyzer not supported: {str(cjk_error)}. "
                            "Falling back to standard analyzer."
                        )
                        create_index_query = f"""
                        CREATE FULLTEXT INDEX {index_name}
                        FOR (n:`{workspace_label}`) ON EACH [n.entity_id]
                        """
                        result = await session.run(create_index_query)
                        await result.consume()
                        logger.info(
                            f"[{self.workspace}] Successfully created full-text index '{index_name}' with standard analyzer."
                        )

        except Exception as e:
            # Handle cases where the command might not be supported
            if "Unknown command" in str(e) or "invalid syntax" in str(e).lower():
                logger.warning(
                    f"[{self.workspace}] Could not create or verify full-text index '{index_name}'. "
                    "This might be because you are using a Neo4j version that does not support it. "
                    "Search functionality will fall back to slower, non-indexed queries."
                )
            else:
                logger.error(
                    f"[{self.workspace}] Failed to create or verify full-text index '{index_name}': {str(e)}"
                )

    async def finalize(self):
        """Close the Neo4j driver and release all resources"""
        if self._driver:
            await self._driver.close()
            self._driver = None

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure driver is closed when context manager exits"""
        await self.finalize()

    async def index_done_callback(self) -> None:
        # Neo4J handles persistence automatically
        pass

    @READ_RETRY
    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given id exists in the database.
        Searches across all TLU node types: Person, Document, Topic, Major, Degree.

        Args:
            node_id: The node identifier to check (matches entity_id, id, or name)

        Returns:
            bool: True if node exists, False otherwise

        Raises:
            Exception: If there is an error executing the query
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                query = """
                MATCH (n)
                WHERE (n:Person AND n.name = $node_id)
                   OR (n:Document AND n.id = $node_id)
                   OR (n:Topic AND n.name = $node_id)
                   OR (n:Major AND n.name = $node_id)
                   OR (n:Degree AND n.name = $node_id)
                   OR n.entity_id = $node_id
                RETURN count(n) > 0 AS node_exists
                """
                result = await session.run(query, node_id=node_id)
                single_result = await result.single()
                await result.consume()
                return single_result["node_exists"] if single_result else False
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking node existence for {node_id}: {str(e)}"
                )
                if result is not None:
                    await result.consume()
                raise

    @READ_RETRY
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes.
        Searches across all edge types: WROTE, ADVISED, HAS_TOPIC, BELONGS_TO_MAJOR, HAS_DEGREE.

        Args:
            source_node_id: Identifier of the source node (entity_id, id, or name)
            target_node_id: Identifier of the target node (entity_id, id, or name)

        Returns:
            bool: True if edge exists, False otherwise

        Raises:
            Exception: If there is an error executing the query
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                query = """
                MATCH (a)-[r]-(b)
                WHERE ((a:Person AND a.name = $source_node_id)
                    OR (a:Document AND a.id = $source_node_id)
                    OR (a:Topic AND a.name = $source_node_id)
                    OR (a:Major AND a.name = $source_node_id)
                    OR (a:Degree AND a.name = $source_node_id)
                    OR a.entity_id = $source_node_id)
                  AND ((b:Person AND b.name = $target_node_id)
                    OR (b:Document AND b.id = $target_node_id)
                    OR (b:Topic AND b.name = $target_node_id)
                    OR (b:Major AND b.name = $target_node_id)
                    OR (b:Degree AND b.name = $target_node_id)
                    OR b.entity_id = $target_node_id)
                RETURN COUNT(r) > 0 AS edgeExists
                """
                result = await session.run(
                    query,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                )
                single_result = await result.single()
                await result.consume()
                return single_result["edgeExists"] if single_result else False
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
                )
                if result is not None:
                    await result.consume()
                raise

    @READ_RETRY
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its identifier, return only node properties.
        Searches across all TLU node types: Person, Document, Topic, Major, Degree.

        Args:
            node_id: The node identifier to look up (matches entity_id, id, or name)

        Returns:
            dict: Node properties if found, with 'entity_id' added for LightRAG compatibility
            None: If node not found

        Raises:
            Exception: If there is an error executing the query
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = """
                MATCH (n)
                WHERE (n:Person AND n.name = $node_id)
                   OR (n:Document AND n.id = $node_id)
                   OR (n:Topic AND n.name = $node_id)
                   OR (n:Major AND n.name = $node_id)
                   OR (n:Degree AND n.name = $node_id)
                   OR n.entity_id = $node_id
                RETURN n
                """
                result = await session.run(query, node_id=node_id)
                try:
                    records = await result.fetch(2)

                    if len(records) > 1:
                        logger.warning(
                            f"[{self.workspace}] Multiple nodes found with id '{node_id}'. Using first node."
                        )
                    if records:
                        node = records[0]["n"]
                        node_dict = dict(node)
                        # Add entity_id for LightRAG compatibility
                        if "entity_id" not in node_dict:
                            node_dict["entity_id"] = node_id
                        # logger.debug(f"Neo4j query node {query} return: {node_dict}")
                        return node_dict
                    return None
                finally:
                    await result.consume()
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node for {node_id}: {str(e)}"
                )
                raise

    @READ_RETRY
    async def get_nodes_batch(self, node_ids: list[str]) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using UNWIND.

        Args:
            node_ids: List of node entity IDs to fetch.

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            UNWIND $node_ids AS id
            MATCH (n)
            WHERE (n:Person AND n.name = id)
            OR (n:Document AND n.id = id)
            OR (n:Topic AND n.name = id)
            OR (n:Major AND n.name = id)
            OR (n:Degree AND n.name = id)
            OR (n:`{workspace_label}` AND n.entity_id = id)
            RETURN id AS requested_id, n
            """
            result = await session.run(query, node_ids=node_ids)
            nodes = {}
            async for record in result:
                entity_id = record["requested_id"]
                node = record["n"]
                node_dict = dict(node)
                # Remove the workspace label if present in a 'labels' property
                if "labels" in node_dict:
                    node_dict["labels"] = [
                        label
                        for label in node_dict["labels"]
                        if label != workspace_label
                    ]
                nodes[entity_id] = node_dict
            await result.consume()  # Make sure to consume the result fully
            return nodes

    @READ_RETRY
    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node.
        Searches across all TLU node types.

        Args:
            node_id: The node identifier (entity_id, id, or name)

        Returns:
            int: The number of relationships the node has, or 0 if no node found

        Raises:
            Exception: If there is an error executing the query
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = """
                    MATCH (n)-[r]-()
                    WHERE (n:Person AND n.name = $node_id)
                       OR (n:Document AND n.id = $node_id)
                       OR (n:Topic AND n.name = $node_id)
                       OR (n:Major AND n.name = $node_id)
                       OR (n:Degree AND n.name = $node_id)
                       OR n.entity_id = $node_id
                    RETURN COUNT(DISTINCT r) AS degree
                """
                result = await session.run(query, node_id=node_id)
                try:
                    record = await result.single()

                    if not record:
                        return 0

                    degree = record["degree"]
                    return degree
                finally:
                    await result.consume()
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting node degree for {node_id}: {str(e)}"
                )
                raise

    @READ_RETRY
    async def node_degrees_batch(self, node_ids: list[str]) -> dict[str, int]:
        """
        Retrieve the degree for multiple nodes in a single query using UNWIND.
        Searches across all TLU node types.

        Args:
            node_ids: List of node identifiers (entity_id, id, or name values).

        Returns:
            A dictionary mapping each node_id to its degree (number of relationships).
            If a node is not found, its degree will be set to 0.
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
                UNWIND $node_ids AS id
                MATCH (n)-[r]-()
                WHERE (n:Person AND n.name = id)
                   OR (n:Document AND n.id = id)
                   OR (n:Topic AND n.name = id)
                   OR (n:Major AND n.name = id)
                   OR (n:Degree AND n.name = id)
                   OR n.entity_id = id
                WITH id, COUNT(DISTINCT r) AS degree
                RETURN id, degree
            """
            result = await session.run(query, node_ids=node_ids)
            degrees = {}
            async for record in result:
                degrees[record["id"]] = record["degree"]
            await result.consume()

            for nid in node_ids:
                if nid not in degrees:
                    degrees[nid] = 0

            return degrees

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes.

        Args:
            src_id: Label of the source node
            tgt_id: Label of the target node

        Returns:
            int: Sum of the degrees of both nodes
        """
        src_degree = await self.node_degree(src_id)
        trg_degree = await self.node_degree(tgt_id)

        # Convert None to 0 for addition
        src_degree = 0 if src_degree is None else src_degree
        trg_degree = 0 if trg_degree is None else trg_degree

        degrees = int(src_degree) + int(trg_degree)
        return degrees

    @READ_RETRY
    async def edge_degrees_batch(
        self, edge_pairs: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.

        Args:
            edge_pairs: List of (src, tgt) tuples.

        Returns:
            A dictionary mapping each (src, tgt) tuple to the sum of their degrees.
        """
        # Collect unique node IDs from all edge pairs.
        unique_node_ids = {src for src, _ in edge_pairs}
        unique_node_ids.update({tgt for _, tgt in edge_pairs})

        # Get degrees for all nodes in one go.
        degrees = await self.node_degrees_batch(list(unique_node_ids))

        # Sum up degrees for each edge pair.
        edge_degrees = {}
        for src, tgt in edge_pairs:
            edge_degrees[(src, tgt)] = degrees.get(src, 0) + degrees.get(tgt, 0)
        return edge_degrees

    @READ_RETRY
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes.
        Returns edge type (WROTE, ADVISED, HAS_TOPIC, BELONGS_TO_MAJOR, HAS_DEGREE) in the result.

        Args:
            source_node_id: Identifier of the source node (entity_id, id, or name)
            target_node_id: Identifier of the target node (entity_id, id, or name)

        Returns:
            dict: Edge properties including 'type' (relationship type) if found
            None: If no edge found

        Raises:
            Exception: If there is an error executing the query
        """
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                query = """
                MATCH (a)-[r]-(b)
                WHERE ((a:Person AND a.name = $source_node_id)
                    OR (a:Document AND a.id = $source_node_id)
                    OR (a:Topic AND a.name = $source_node_id)
                    OR (a:Major AND a.name = $source_node_id)
                    OR (a:Degree AND a.name = $source_node_id)
                    OR a.entity_id = $source_node_id)
                  AND ((b:Person AND b.name = $target_node_id)
                    OR (b:Document AND b.id = $target_node_id)
                    OR (b:Topic AND b.name = $target_node_id)
                    OR (b:Major AND b.name = $target_node_id)
                    OR (b:Degree AND b.name = $target_node_id)
                    OR b.entity_id = $target_node_id)
                RETURN type(r) as edge_type, properties(r) as edge_properties
                """
                result = await session.run(
                    query,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                )
                try:
                    records = await result.fetch(2)

                    if len(records) > 1:
                        logger.warning(
                            f"[{self.workspace}] Multiple edges found between '{source_node_id}' and '{target_node_id}'. Using first edge."
                        )
                    if records:
                        try:
                            edge_type = records[0]["edge_type"]
                            edge_result = dict(records[0]["edge_properties"])
                            # Add edge type to result
                            edge_result["type"] = edge_type
                            # Ensure required keys exist with defaults
                            required_keys = {
                                "weight": 1.0,
                                "source_id": None,
                                "description": None,
                                "keywords": None,
                            }
                            for key, default_value in required_keys.items():
                                if key not in edge_result:
                                    edge_result[key] = default_value
                                    logger.warning(
                                        f"[{self.workspace}] Edge between {source_node_id} and {target_node_id} "
                                        f"missing {key}, using default: {default_value}"
                                    )
                            return edge_result
                        except (KeyError, TypeError, ValueError) as e:
                            logger.error(
                                f"[{self.workspace}] Error processing edge properties between {source_node_id} "
                                f"and {target_node_id}: {str(e)}"
                            )
                            return {
                                "weight": 1.0,
                                "source_id": None,
                                "description": None,
                                "keywords": None,
                                "type": None,
                            }

                    return None
                finally:
                    await result.consume()

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error in get_edge between {source_node_id} and {target_node_id}: {str(e)}"
            )
            raise

    @READ_RETRY
    async def get_edges_batch(
        self, pairs: list[dict[str, str]]
    ) -> dict[tuple[str, str], dict]:
        """
        Retrieve edge properties for multiple (src, tgt) pairs in one query.

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            UNWIND $pairs AS pair
            MATCH (start)-[r]-(end)
            WHERE ((start:Person AND start.name = pair.src) OR (start:Document AND start.id = pair.src) OR (start:Topic AND start.name = pair.src) OR (start:Major AND start.name = pair.src) OR (start:Degree AND start.name = pair.src) OR (start:`{workspace_label}` AND start.entity_id = pair.src))
              AND ((end:Person AND end.name = pair.tgt) OR (end:Document AND end.id = pair.tgt) OR (end:Topic AND end.name = pair.tgt) OR (end:Major AND end.name = pair.tgt) OR (end:Degree AND end.name = pair.tgt) OR (end:`{workspace_label}` AND end.entity_id = pair.tgt))
            RETURN pair.src AS src_id, pair.tgt AS tgt_id, collect({{props: properties(r), type: type(r)}}) AS edges_with_type
            """
            result = await session.run(query, pairs=pairs)
            edges_dict = {}
            async for record in result:
                src = record["src_id"]
                tgt = record["tgt_id"]
                edges_list = record["edges_with_type"]
                if edges_list and len(edges_list) > 0:
                    first_edge = edges_list[0]
                    edge_props = first_edge["props"]
                    edge_props["type"] = first_edge["type"]
                    # Ensure required keys exist with defaults
                    for key, default in {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }.items():
                        if key not in edge_props:
                            edge_props[key] = default
                    edges_dict[(src, tgt)] = edge_props
                else:
                    # No edge found – set default edge properties
                    edges_dict[(src, tgt)] = {
                        "weight": 1.0,
                        "source_id": None,
                        "description": None,
                        "keywords": None,
                    }
            await result.consume()
            return edges_dict

    @READ_RETRY
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node.
        Returns edges with relationship types.

        Args:
            source_node_id: Identifier of the node to get edges for (entity_id, id, or name)

        Returns:
            list[tuple[str, str]]: List of (source_id, target_id) tuples representing edges
            None: If no edges found

        Raises:
            Exception: If there is an error executing the query
        """
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                results = None
                try:
                    query = """
                    MATCH (n)-[r]-(connected)
                    WHERE (n:Person AND n.name = $source_node_id)
                       OR (n:Document AND n.id = $source_node_id)
                       OR (n:Topic AND n.name = $source_node_id)
                       OR (n:Major AND n.name = $source_node_id)
                       OR (n:Degree AND n.name = $source_node_id)
                       OR n.entity_id = $source_node_id
                    RETURN n, r, connected
                    """
                    results = await session.run(query, source_node_id=source_node_id)

                    edges = []
                    async for record in results:
                        source_node = record["n"]
                        connected_node = record["connected"]

                        if not source_node or not connected_node:
                            continue

                        source_id = (
                            source_node.get("entity_id")
                            or source_node.get("id")
                            or source_node.get("name")
                        )
                        target_id = (
                            connected_node.get("entity_id")
                            or connected_node.get("id")
                            or connected_node.get("name")
                        )

                        if source_id and target_id:
                            edges.append((source_id, target_id))

                    await results.consume()
                    return edges if edges else None
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error getting edges for node {source_node_id}: {str(e)}"
                    )
                    if results is not None:
                        await results.consume()
                    raise
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error in get_node_edges for {source_node_id}: {str(e)}"
            )
            raise

    @READ_RETRY
    async def get_nodes_edges_batch(
        self, node_ids: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Batch retrieve edges for multiple nodes in one query using UNWIND.
        For each node, returns both outgoing and incoming edges to properly represent
        the undirected graph nature.

        Args:
            node_ids: List of node IDs (entity_id) for which to retrieve edges.

        Returns:
            A dictionary mapping each node ID to its list of edge tuples (source, target).
            For each node, the list includes both:
            - Outgoing edges: (queried_node, connected_node)
            - Incoming edges: (connected_node, queried_node)
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            # Query to get both outgoing and incoming edges
            workspace_label = self._get_workspace_label()
            query = f"""
                UNWIND $node_ids AS id
                MATCH (n)
                WHERE (n:Person AND n.name = id) OR (n:Document AND n.id = id) OR (n:Topic AND n.name = id) OR (n:Major AND n.name = id) OR (n:Degree AND n.name = id) OR (n:`{workspace_label}` AND n.entity_id = id)
                OPTIONAL MATCH (n)-[r]-(connected)
                WHERE connected:Person OR connected:Document OR connected:Topic OR connected:Major OR connected:Degree OR connected:`{workspace_label}`
                WITH id, n, r, connected,
                     CASE WHEN connected:Person THEN connected.name WHEN connected:Document THEN connected.id WHEN connected:Topic THEN connected.name WHEN connected:Major THEN connected.name WHEN connected:Degree THEN connected.name ELSE connected.entity_id END AS connected_entity_id
                RETURN id AS queried_id, 
                       id AS node_entity_id,
                       connected_entity_id,
                       CASE WHEN startNode(r) = n THEN id ELSE connected_entity_id END AS start_entity_id
            """
            result = await session.run(query, node_ids=node_ids)

            # Initialize the dictionary with empty lists for each node ID
            edges_dict = {node_id: [] for node_id in node_ids}

            # Process results to include both outgoing and incoming edges
            async for record in result:
                queried_id = record["queried_id"]
                node_entity_id = record["node_entity_id"]
                connected_entity_id = record["connected_entity_id"]
                start_entity_id = record["start_entity_id"]

                # Skip if either node is None
                if not node_entity_id or not connected_entity_id:
                    continue

                # Determine the actual direction of the edge
                # If the start node is the queried node, it's an outgoing edge
                # Otherwise, it's an incoming edge
                if start_entity_id == node_entity_id:
                    # Outgoing edge: (queried_node -> connected_node)
                    edges_dict[queried_id].append((node_entity_id, connected_entity_id))
                else:
                    # Incoming edge: (connected_node -> queried_node)
                    edges_dict[queried_id].append((connected_entity_id, node_entity_id))

            await result.consume()  # Ensure results are fully consumed
            return edges_dict

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
                neo4jExceptions.SessionExpired,
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        workspace_label = self._get_workspace_label()
        properties = node_data
        entity_type = properties["entity_type"]
        if "entity_id" not in properties:
            raise ValueError("Neo4j: node properties must contain an 'entity_id' field")

        try:
            async with self._driver.session(database=self._DATABASE) as session:

                async def execute_upsert(tx: AsyncManagedTransaction):
                    query = f"""
                    MERGE (n:`{workspace_label}` {{entity_id: $entity_id}})
                    SET n += $properties
                    SET n:`{entity_type}`
                    """
                    result = await tx.run(
                        query, entity_id=node_id, properties=properties
                    )
                    await result.consume()  # Ensure result is fully consumed

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during upsert: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
                neo4jExceptions.SessionExpired,
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.
        Ensures both source and target nodes exist and are unique before creating the edge.
        Uses entity_id property to uniquely identify nodes.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): Dictionary of properties to set on the edge

        Raises:
            ValueError: If either source or target node does not exist or is not unique
        """
        try:
            edge_properties = edge_data
            async with self._driver.session(database=self._DATABASE) as session:

                async def execute_upsert(tx: AsyncManagedTransaction):
                    workspace_label = self._get_workspace_label()
                    query = f"""
                    MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})
                    WITH source
                    MATCH (target:`{workspace_label}` {{entity_id: $target_entity_id}})
                    MERGE (source)-[r:DIRECTED]-(target)
                    SET r += $properties
                    RETURN r, source, target
                    """
                    result = await tx.run(
                        query,
                        source_entity_id=source_node_id,
                        target_entity_id=target_node_id,
                        properties=edge_properties,
                    )
                    try:
                        await result.fetch(2)
                    finally:
                        await result.consume()  # Ensure result is consumed

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during edge upsert: {str(e)}")
            raise

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maxiumu nodes to return by BFS, Defaults to 1000

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # Get max_nodes from global_config if not provided
        if max_nodes is None:
            max_nodes = self.global_config.get("max_graph_nodes", 1000)
        else:
            # Limit max_nodes to not exceed global_config max_graph_nodes
            max_nodes = min(max_nodes, self.global_config.get("max_graph_nodes", 1000))

        workspace_label = self._get_workspace_label()
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                if node_label == "*":
                    # First check total node count to determine if graph is truncated
                    count_query = (
                        f"MATCH (n:`{workspace_label}`) RETURN count(n) as total"
                    )
                    count_result = None
                    try:
                        count_result = await session.run(count_query)
                        count_record = await count_result.single()

                        if count_record and count_record["total"] > max_nodes:
                            result.is_truncated = True
                            logger.info(
                                f"[{self.workspace}] Graph truncated: {count_record['total']} nodes found, limited to {max_nodes}"
                            )
                    finally:
                        if count_result:
                            await count_result.consume()

                    # Run main query to get nodes with highest degree
                    main_query = f"""
                    MATCH (n:`{workspace_label}`)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COALESCE(count(r), 0) AS degree
                    ORDER BY degree DESC
                    LIMIT $max_nodes
                    WITH collect({{node: n}}) AS filtered_nodes
                    UNWIND filtered_nodes AS node_info
                    WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                    OPTIONAL MATCH (a)-[r]-(b)
                    WHERE a IN kept_nodes AND b IN kept_nodes
                    RETURN filtered_nodes AS node_info,
                           collect(DISTINCT r) AS relationships
                    """
                    result_set = None
                    try:
                        result_set = await session.run(
                            main_query,
                            {"max_nodes": max_nodes},
                        )
                        record = await result_set.single()
                    finally:
                        if result_set:
                            await result_set.consume()

                else:
                    # return await self._robust_fallback(node_label, max_depth, max_nodes)
                    # First try without limit to check if we need to truncate
                    full_query = f"""
                    MATCH (start:`{workspace_label}`)
                    WHERE start.entity_id = $entity_id
                    WITH start
                    CALL apoc.path.subgraphAll(start, {{
                        relationshipFilter: '',
                        labelFilter: '{workspace_label}',
                        minLevel: 0,
                        maxLevel: $max_depth,
                        bfs: true
                    }})
                    YIELD nodes, relationships
                    WITH nodes, relationships, size(nodes) AS total_nodes
                    UNWIND nodes AS node
                    WITH collect({{node: node}}) AS node_info, relationships, total_nodes
                    RETURN node_info, relationships, total_nodes
                    """

                    # Try to get full result
                    full_result = None
                    try:
                        full_result = await session.run(
                            full_query,
                            {
                                "entity_id": node_label,
                                "max_depth": max_depth,
                            },
                        )
                        full_record = await full_result.single()

                        # If no record found, return empty KnowledgeGraph
                        if not full_record:
                            logger.debug(
                                f"[{self.workspace}] No nodes found for entity_id: {node_label}"
                            )
                            return result

                        # If record found, check node count
                        total_nodes = full_record["total_nodes"]

                        if total_nodes <= max_nodes:
                            # If node count is within limit, use full result directly
                            logger.debug(
                                f"[{self.workspace}] Using full result with {total_nodes} nodes (no truncation needed)"
                            )
                            record = full_record
                        else:
                            # If node count exceeds limit, set truncated flag and run limited query
                            result.is_truncated = True
                            logger.info(
                                f"[{self.workspace}] Graph truncated: {total_nodes} nodes found, breadth-first search limited to {max_nodes}"
                            )

                            # Run limited query
                            limited_query = f"""
                            MATCH (start:`{workspace_label}`)
                            WHERE start.entity_id = $entity_id
                            WITH start
                            CALL apoc.path.subgraphAll(start, {{
                                relationshipFilter: '',
                                labelFilter: '{workspace_label}',
                                minLevel: 0,
                                maxLevel: $max_depth,
                                limit: $max_nodes,
                                bfs: true
                            }})
                            YIELD nodes, relationships
                            UNWIND nodes AS node
                            WITH collect({{node: node}}) AS node_info, relationships
                            RETURN node_info, relationships
                            """
                            result_set = None
                            try:
                                result_set = await session.run(
                                    limited_query,
                                    {
                                        "entity_id": node_label,
                                        "max_depth": max_depth,
                                        "max_nodes": max_nodes,
                                    },
                                )
                                record = await result_set.single()
                            finally:
                                if result_set:
                                    await result_set.consume()
                    finally:
                        if full_result:
                            await full_result.consume()

                if record:
                    # Handle nodes (compatible with multi-label cases)
                    for node_info in record["node_info"]:
                        node = node_info["node"]
                        node_id = node.id
                        if node_id not in seen_nodes:
                            result.nodes.append(
                                KnowledgeGraphNode(
                                    id=f"{node_id}",
                                    labels=[node.get("entity_id")],
                                    properties=dict(node),
                                )
                            )
                            seen_nodes.add(node_id)

                    # Handle relationships (including direction information)
                    for rel in record["relationships"]:
                        edge_id = rel.id
                        if edge_id not in seen_edges:
                            start = rel.start_node
                            end = rel.end_node
                            result.edges.append(
                                KnowledgeGraphEdge(
                                    id=f"{edge_id}",
                                    type=rel.type,
                                    source=f"{start.id}",
                                    target=f"{end.id}",
                                    properties=dict(rel),
                                )
                            )
                            seen_edges.add(edge_id)

                    logger.info(
                        f"[{self.workspace}] Subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
                    )

            except neo4jExceptions.ClientError as e:
                logger.warning(f"[{self.workspace}] APOC plugin error: {str(e)}")
                if node_label != "*":
                    logger.warning(
                        f"[{self.workspace}] Neo4j: falling back to basic Cypher recursive search..."
                    )
                    return await self._robust_fallback(node_label, max_depth, max_nodes)
                else:
                    logger.warning(
                        f"[{self.workspace}] Neo4j: APOC plugin error with wildcard query, returning empty result"
                    )

        return result

    async def _robust_fallback(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """
        Fallback implementation when APOC plugin is not available or incompatible.
        This method implements the same functionality as get_knowledge_graph but uses
        only basic Cypher queries and true breadth-first traversal instead of APOC procedures.
        """
        from collections import deque

        result = KnowledgeGraph()
        visited_nodes = set()
        visited_edges = set()
        visited_edge_pairs = set()

        # Get the starting node's data
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
            RETURN id(n) as node_id, n
            """
            node_result = await session.run(query, entity_id=node_label)
            try:
                node_record = await node_result.single()
                if not node_record:
                    return result

                # Create initial KnowledgeGraphNode
                start_node = KnowledgeGraphNode(
                    id=f"{node_record['n'].get('entity_id')}",
                    labels=[node_record["n"].get("entity_id")],
                    properties=dict(node_record["n"]._properties),
                )
            finally:
                await node_result.consume()  # Ensure results are consumed

        # Initialize queue for BFS with (node, edge, depth) tuples
        # edge is None for the starting node
        queue = deque([(start_node, None, 0)])

        # True BFS implementation using a queue
        while queue and len(visited_nodes) < max_nodes:
            # Dequeue the next node to process
            current_node, current_edge, current_depth = queue.popleft()

            # Skip if already visited or exceeds max depth
            if current_node.id in visited_nodes:
                continue

            if current_depth > max_depth:
                logger.debug(
                    f"[{self.workspace}] Skipping node at depth {current_depth} (max_depth: {max_depth})"
                )
                continue

            # Add current node to result
            result.nodes.append(current_node)
            visited_nodes.add(current_node.id)

            # Add edge to result if it exists and not already added
            if current_edge and current_edge.id not in visited_edges:
                result.edges.append(current_edge)
                visited_edges.add(current_edge.id)

            # Stop if we've reached the node limit
            if len(visited_nodes) >= max_nodes:
                result.is_truncated = True
                logger.info(
                    f"[{self.workspace}] Graph truncated: breadth-first search limited to: {max_nodes} nodes"
                )
                break

            # Get all edges and target nodes for the current node (even at max_depth)
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (a:`{workspace_label}` {{entity_id: $entity_id}})-[r]-(b)
                WITH r, b, id(r) as edge_id, id(b) as target_id
                RETURN r, b, edge_id, target_id
                """
                results = await session.run(query, entity_id=current_node.id)

                # Get all records and release database connection
                records = await results.fetch(1000)  # Max neighbor nodes we can handle
                await results.consume()  # Ensure results are consumed

                # Process all neighbors - capture all edges but only queue unvisited nodes
                for record in records:
                    rel = record["r"]
                    edge_id = str(record["edge_id"])

                    if edge_id not in visited_edges:
                        b_node = record["b"]
                        target_id = b_node.get("entity_id")

                        if target_id:  # Only process if target node has entity_id
                            # Create KnowledgeGraphNode for target
                            target_node = KnowledgeGraphNode(
                                id=f"{target_id}",
                                labels=[target_id],
                                properties=dict(b_node._properties),
                            )

                            # Create KnowledgeGraphEdge
                            target_edge = KnowledgeGraphEdge(
                                id=f"{edge_id}",
                                type=rel.type,
                                source=f"{current_node.id}",
                                target=f"{target_id}",
                                properties=dict(rel),
                            )

                            # Sort source_id and target_id to ensure (A,B) and (B,A) are treated as the same edge
                            sorted_pair = tuple(sorted([current_node.id, target_id]))

                            # Check if the same edge already exists (considering undirectedness)
                            if sorted_pair not in visited_edge_pairs:
                                # Only add the edge if the target node is already in the result or will be added
                                if target_id in visited_nodes or (
                                    target_id not in visited_nodes
                                    and current_depth < max_depth
                                ):
                                    result.edges.append(target_edge)
                                    visited_edges.add(edge_id)
                                    visited_edge_pairs.add(sorted_pair)

                            # Only add unvisited nodes to the queue for further expansion
                            if target_id not in visited_nodes:
                                # Only add to queue if we're not at max depth yet
                                if current_depth < max_depth:
                                    # Add node to queue with incremented depth
                                    # Edge is already added to result, so we pass None as edge
                                    queue.append((target_node, None, current_depth + 1))
                                else:
                                    # At max depth, we've already added the edge but we don't add the node
                                    # This prevents adding nodes beyond max_depth to the result
                                    logger.debug(
                                        f"[{self.workspace}] Node {target_id} beyond max depth {max_depth}, edge added but node not included"
                                    )
                            else:
                                # If target node already exists in result, we don't need to add it again
                                logger.debug(
                                    f"[{self.workspace}] Node {target_id} already visited, edge added but node not queued"
                                )
                        else:
                            logger.warning(
                                f"[{self.workspace}] Skipping edge {edge_id} due to missing entity_id on target node"
                            )

        logger.info(
            f"[{self.workspace}] BFS subgraph query successful | Node count: {len(result.nodes)} | Edge count: {len(result.edges)}"
        )
        return result

    async def get_all_labels(self) -> list[str]:
        """
        Get all existing entity identifiers in the database.
        Returns all entity_id, id, and name values across all TLU node types.

        Returns:
            list[str]: Alphabetically sorted list of all entity identifiers
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            MATCH (n:Person|Document|Topic|Major|Degree)
            WITH CASE 
                WHEN n.entity_id IS NOT NULL THEN n.entity_id
                WHEN n.id IS NOT NULL THEN n.id
                WHEN n.name IS NOT NULL THEN n.name
            END AS label
            WHERE label IS NOT NULL
            RETURN DISTINCT label
            ORDER BY label
            """
            result = await session.run(query)
            labels = []
            try:
                async for record in result:
                    labels.append(record["label"])
            finally:
                await result.consume()
            return labels

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
                neo4jExceptions.SessionExpired,
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete
        """

        async def _do_delete(tx: AsyncManagedTransaction):
            workspace_label = self._get_workspace_label()
            query = f"""
            MATCH (n:`{workspace_label}` {{entity_id: $entity_id}})
            DETACH DELETE n
            """
            result = await tx.run(query, entity_id=node_id)
            logger.debug(f"[{self.workspace}] Deleted node with label '{node_id}'")
            await result.consume()  # Ensure result is fully consumed

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_delete)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during node deletion: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
                neo4jExceptions.SessionExpired,
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node labels to be deleted
        """
        for node in nodes:
            await self.delete_node(node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
                neo4jExceptions.SessionExpired,
                ConnectionResetError,
                OSError,
            )
        ),
    )
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:

            async def _do_delete_edge(tx: AsyncManagedTransaction):
                workspace_label = self._get_workspace_label()
                query = f"""
                MATCH (source:`{workspace_label}` {{entity_id: $source_entity_id}})-[r]-(target:`{workspace_label}` {{entity_id: $target_entity_id}})
                DELETE r
                """
                result = await tx.run(
                    query, source_entity_id=source, target_entity_id=target
                )
                logger.debug(
                    f"[{self.workspace}] Deleted edge from '{source}' to '{target}'"
                )
                await result.consume()  # Ensure result is fully consumed

            try:
                async with self._driver.session(database=self._DATABASE) as session:
                    await session.execute_write(_do_delete_edge)
            except Exception as e:
                logger.error(f"[{self.workspace}] Error during edge deletion: {str(e)}")
                raise

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            MATCH (n)
            WHERE (n:Person OR n:Document OR n:Topic OR n:Major OR n:Degree OR n:`{workspace_label}`)
            RETURN n
            """
            result = await session.run(query)
            nodes = []
            async for record in result:
                node = record["n"]
                node_dict = dict(node)
                # Add node id (entity_id) to the dictionary for easier access
                node_dict["id"] = (
                    node_dict.get("entity_id")
                    or node_dict.get("id")
                    or node_dict.get("name")
                )
                nodes.append(node_dict)
            await result.consume()
            return nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
        """
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = f"""
            MATCH (a)-[r]-(b)
            WHERE (a:Person OR a:Document OR a:Topic OR a:Major OR a:Degree OR a:`{workspace_label}`)
              AND (b:Person OR b:Document OR b:Topic OR b:Major OR b:Degree OR b:`{workspace_label}`)
            RETURN DISTINCT 
                COALESCE(a.entity_id, a.id, a.name) AS source, 
                COALESCE(b.entity_id, b.id, b.name) AS target, 
                properties(r) AS properties
            """
            result = await session.run(query)
            edges = []
            async for record in result:
                edge_properties = record["properties"]
                edge_properties["source"] = record["source"]
                edge_properties["target"] = record["target"]
                edges.append(edge_properties)
            await result.consume()
            return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels(entity names) by node degree (most connected entities)

        Args:
            limit: Maximum number of labels to return

        Returns:
            List of labels(entity names) sorted by degree (highest first)
        """
        workspace_label = self._get_workspace_label()
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            result = None
            try:
                query = f"""
                MATCH (n:`{workspace_label}`)
                WHERE n.entity_id IS NOT NULL
                OPTIONAL MATCH (n)-[r]-()
                WITH n.entity_id AS label, count(r) AS degree
                ORDER BY degree DESC, label ASC
                LIMIT $limit
                RETURN label
                """
                result = await session.run(query, limit=limit)
                labels = []
                async for record in result:
                    labels.append(record["label"])
                await result.consume()

                logger.debug(
                    f"[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})"
                )
                return labels
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error getting popular labels: {str(e)}"
                )
                if result is not None:
                    await result.consume()
                raise

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """
        Search labels(entity names) with fuzzy matching, using a full-text index for performance if available.
        Enhanced with Chinese text support using CJK analyzer.
        Falls back to a slower CONTAINS search if the index is not available or fails.
        """
        workspace_label = self._get_workspace_label()
        query_strip = query.strip()
        if not query_strip:
            return []

        query_lower = query_strip.lower()
        is_chinese = self._is_chinese_text(query_strip)
        index_name = self._get_fulltext_index_name(workspace_label)

        # Attempt to use the full-text index first
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                if is_chinese:
                    # For Chinese text, use different search strategies
                    cypher_query = f"""
                    CALL db.index.fulltext.queryNodes($index_name, $search_query) YIELD node, score
                    WITH node, score
                    WHERE node:`{workspace_label}`
                    WITH node.entity_id AS label, score
                    WITH label, score,
                         CASE
                             WHEN label = $query_strip THEN score + 1000
                             WHEN label CONTAINS $query_strip THEN score + 500
                             ELSE score
                         END AS final_score
                    RETURN label
                    ORDER BY final_score DESC, label ASC
                    LIMIT $limit
                    """
                    # For Chinese, don't add wildcard as it may not work properly with CJK analyzer
                    search_query = query_strip
                else:
                    # For non-Chinese text, use the original logic with wildcard
                    cypher_query = f"""
                    CALL db.index.fulltext.queryNodes($index_name, $search_query) YIELD node, score
                    WITH node, score
                    WHERE node:`{workspace_label}`
                    WITH node.entity_id AS label, toLower(node.entity_id) AS label_lower, score
                    WITH label, label_lower, score,
                         CASE
                             WHEN label_lower = $query_lower THEN score + 1000
                             WHEN label_lower STARTS WITH $query_lower THEN score + 500
                             WHEN label_lower CONTAINS ' ' + $query_lower OR label_lower CONTAINS '_' + $query_lower THEN score + 50
                             ELSE score
                         END AS final_score
                    RETURN label
                    ORDER BY final_score DESC, label ASC
                    LIMIT $limit
                    """
                    search_query = f"{query_strip}*"

                result = await session.run(
                    cypher_query,
                    index_name=index_name,
                    search_query=search_query,
                    query_lower=query_lower,
                    query_strip=query_strip,
                    limit=limit,
                )
                labels = [record["label"] async for record in result]
                await result.consume()

                logger.debug(
                    f"[{self.workspace}] Full-text search ({'Chinese' if is_chinese else 'Latin'}) for '{query}' returned {len(labels)} results (limit: {limit})"
                )
                return labels

        except Exception as e:
            # If the full-text search fails, fall back to CONTAINS search
            logger.warning(
                f"[{self.workspace}] Full-text search failed with error: {str(e)}. "
                "Falling back to slower, non-indexed search."
            )

            # Enhanced fallback implementation
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                if is_chinese:
                    # For Chinese text, use direct CONTAINS without case conversion
                    cypher_query = f"""
                    MATCH (n:`{workspace_label}`)
                    WHERE n.entity_id IS NOT NULL
                    WITH n.entity_id AS label
                    WHERE label CONTAINS $query_strip
                    WITH label,
                         CASE
                             WHEN label = $query_strip THEN 1000
                             WHEN label STARTS WITH $query_strip THEN 500
                             ELSE 100 - size(label)
                         END AS score
                    ORDER BY score DESC, label ASC
                    LIMIT $limit
                    RETURN label
                    """
                    result = await session.run(
                        cypher_query, query_strip=query_strip, limit=limit
                    )
                else:
                    # For non-Chinese text, use the original fallback logic
                    cypher_query = f"""
                    MATCH (n:`{workspace_label}`)
                    WHERE n.entity_id IS NOT NULL
                    WITH n.entity_id AS label, toLower(n.entity_id) AS label_lower
                    WHERE label_lower CONTAINS $query_lower
                    WITH label, label_lower,
                         CASE
                             WHEN label_lower = $query_lower THEN 1000
                             WHEN label_lower STARTS WITH $query_lower THEN 500
                             ELSE 100 - size(label)
                         END AS score
                    ORDER BY score DESC, label ASC
                    LIMIT $limit
                    RETURN label
                    """
                    result = await session.run(
                        cypher_query, query_lower=query_lower, limit=limit
                    )

                labels = [record["label"] async for record in result]
                await result.consume()
                logger.debug(
                    f"[{self.workspace}] Fallback search ({'Chinese' if is_chinese else 'Latin'}) for '{query}' returned {len(labels)} results (limit: {limit})"
                )
                return labels

    async def drop(self) -> dict[str, str]:
        """Drop all data from current workspace storage and clean up resources

        This method will delete all nodes and relationships in the current workspace only.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "workspace data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        workspace_label = self._get_workspace_label()
        try:
            async with self._driver.session(database=self._DATABASE) as session:
                # Delete all nodes and relationships in current workspace only
                query = f"MATCH (n:`{workspace_label}`) DETACH DELETE n"
                result = await session.run(query)
                await result.consume()  # Ensure result is fully consumed

                # logger.debug(
                #     f"[{self.workspace}] Process {os.getpid()} drop Neo4j workspace '{workspace_label}' in database {self._DATABASE}"
                # )
                return {
                    "status": "success",
                    "message": f"workspace '{workspace_label}' data dropped",
                }
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping Neo4j workspace '{workspace_label}' in database {self._DATABASE}: {e}"
            )
            return {"status": "error", "message": str(e)}
