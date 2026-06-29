import json
import sys
from pathlib import Path
from neo4j import GraphDatabase, Query
from collections import defaultdict


class Neo4jImporter:
    def __init__(self, uri, user, password):
        """Initialize connection to Neo4j Aura"""
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✅ Connected to Neo4j Aura successfully!")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")
            sys.exit(1)

    def close(self):
        """Close database connection"""
        self.driver.close()

    def clear_database(self):
        """Clear all existing data (use with caution!)"""
        print("\n⚠️  Clearing existing data...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("✅ Database cleared")

    def create_constraints_and_indexes(self):
        """Create unique constraints and indexes"""
        print("\n📐 Creating constraints and indexes...")

        constraints = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT major_name IF NOT EXISTS FOR (m:Major) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT degree_name IF NOT EXISTS FOR (deg:Degree) REQUIRE deg.name IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX doc_type IF NOT EXISTS FOR (d:Document) ON (d.type)",
            "CREATE INDEX doc_year IF NOT EXISTS FOR (d:Document) ON (d.year)",
        ]

        with self.driver.session() as session:
            # Constraints
            try:
                session.run(
                    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
                )
                print("  ✓ doc_id (Document)")
            except Exception as e:
                print(f"  ⚠ doc_id constraint may already exist: {e}")

            try:
                session.run(
                    "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE"
                )
                print("  ✓ person_name (Person)")
            except Exception as e:
                print(f"  ⚠ person_name constraint may already exist: {e}")

            try:
                session.run(
                    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE"
                )
                print("  ✓ topic_name (Topic)")
            except Exception as e:
                print(f"  ⚠ topic_name constraint may already exist: {e}")

            try:
                session.run(
                    "CREATE CONSTRAINT major_name IF NOT EXISTS FOR (m:Major) REQUIRE m.name IS UNIQUE"
                )
                print("  ✓ major_name (Major)")
            except Exception as e:
                print(f"  ⚠ major_name constraint may already exist: {e}")

            try:
                session.run(
                    "CREATE CONSTRAINT degree_name IF NOT EXISTS FOR (deg:Degree) REQUIRE deg.name IS UNIQUE"
                )
                print("  ✓ degree_name (Degree)")
            except Exception as e:
                print(f"  ⚠ degree_name constraint may already exist: {e}")

            # Indexes
            try:
                session.run(
                    "CREATE INDEX doc_type IF NOT EXISTS FOR (d:Document) ON (d.type)"
                )
                print("  ✓ doc_type index")
            except Exception as e:
                print(f"  ⚠ doc_type index may already exist: {e}")

            try:
                session.run(
                    "CREATE INDEX doc_year IF NOT EXISTS FOR (d:Document) ON (d.year)"
                )
                print("  ✓ doc_year index")
            except Exception as e:
                print(f"  ⚠ doc_year index may already exist: {e}")

    def import_documents(self, data, batch_size=500):
        """Import Document nodes"""
        print(f"\n📄 Importing {len(data):,} Document nodes...")

        query = """
        UNWIND $batch AS doc
        MERGE (d:Document {id: doc.id})
        SET d.type = doc.type,
            d.raw_type = doc.raw_type,
            d.title = doc.title,
            d.abstract = doc.abstract,
            d.year = doc.year,
            d.language = doc.language,
            d.publisher = doc.publisher,
            d.uri = doc.uri,
            d.oai_id = doc.oai_id,
            d.ddc = doc.ddc,
            d.date = doc.date
        """

        count = 0
        with self.driver.session() as session:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                session.run(query, batch=batch)
                count += len(batch)
                print(
                    f"  Progress: {count:,} / {len(data):,} ({count / len(data) * 100:.1f}%)",
                    end="\r",
                )

        print(f"\n✅ Imported {count:,} Document nodes")

    def import_persons_and_relationships(self, data, batch_size=500):
        """Import Person nodes and create WROTE/ADVISED relationships"""
        print(f"\n👤 Importing Person nodes and relationships...")

        # Extract unique persons
        all_persons = set()
        for d in data:
            all_persons.update(d.get("authors", []))
            all_persons.update(d.get("advisors", []))
        all_persons = list(all_persons)

        # Create Person nodes
        print(f"  Creating {len(all_persons):,} Person nodes...")
        person_query = """
        UNWIND $batch AS name
        MERGE (p:Person {name: name})
        """

        with self.driver.session() as session:
            for i in range(0, len(all_persons), batch_size):
                batch = all_persons[i : i + batch_size]
                session.run(person_query, batch=batch)

        # Create WROTE relationships
        print(f"  Creating WROTE relationships...")
        wrote_query = """
        UNWIND $batch AS item
        MATCH (p:Person {name: item.author})
        MATCH (d:Document {id: item.doc_id})
        MERGE (p)-[:WROTE]->(d)
        """

        wrote_rels = []
        for d in data:
            for author in d.get("authors", []):
                wrote_rels.append({"author": author, "doc_id": d["id"]})

        with self.driver.session() as session:
            for i in range(0, len(wrote_rels), batch_size):
                batch = wrote_rels[i : i + batch_size]
                session.run(wrote_query, batch=batch)

        print(f"  ✅ Created {len(wrote_rels):,} WROTE relationships")

        # Create ADVISED relationships
        print(f"  Creating ADVISED relationships...")
        advised_query = """
        UNWIND $batch AS item
        MATCH (p:Person {name: item.advisor})
        MATCH (d:Document {id: item.doc_id})
        MERGE (p)-[:ADVISED]->(d)
        """

        advised_rels = []
        for d in data:
            for advisor in d.get("advisors", []):
                advised_rels.append({"advisor": advisor, "doc_id": d["id"]})

        with self.driver.session() as session:
            for i in range(0, len(advised_rels), batch_size):
                batch = advised_rels[i : i + batch_size]
                session.run(advised_query, batch=batch)

        print(f"  ✅ Created {len(advised_rels):,} ADVISED relationships")

    def import_topics(self, data, batch_size=500):
        """Import Topic nodes and create HAS_TOPIC relationships"""
        print(f"\n🏷️  Importing Topic nodes and relationships...")

        # Extract unique topics
        all_topics = set()
        for d in data:
            all_topics.update(d.get("subjects", []))
        all_topics = list(all_topics)

        # Create Topic nodes
        print(f"  Creating {len(all_topics):,} Topic nodes...")
        topic_query = """
        UNWIND $batch AS name
        MERGE (t:Topic {name: name})
        """

        with self.driver.session() as session:
            for i in range(0, len(all_topics), batch_size):
                batch = all_topics[i : i + batch_size]
                session.run(topic_query, batch=batch)

        # Create HAS_TOPIC relationships
        print(f"  Creating HAS_TOPIC relationships...")
        has_topic_query = """
        UNWIND $batch AS item
        MATCH (d:Document {id: item.doc_id})
        MATCH (t:Topic {name: item.topic})
        MERGE (d)-[:HAS_TOPIC]->(t)
        """

        topic_rels = []
        for d in data:
            for topic in d.get("subjects", []):
                topic_rels.append({"doc_id": d["id"], "topic": topic})

        with self.driver.session() as session:
            for i in range(0, len(topic_rels), batch_size):
                batch = topic_rels[i : i + batch_size]
                session.run(has_topic_query, batch=batch)

        print(f"  ✅ Created {len(topic_rels):,} HAS_TOPIC relationships")

    def import_majors_and_degrees(self, data, batch_size=500):
        """Import Major and Degree nodes with relationships"""
        print(f"\n🎓 Importing Major and Degree nodes...")

        # Extract unique majors (filter empty)
        all_majors = set(
            d["major"] for d in data if d.get("major") and d["major"].strip()
        )
        all_majors = list(all_majors)

        # Create Major nodes
        if all_majors:
            print(f"  Creating {len(all_majors):,} Major nodes...")
            major_query = """
            UNWIND $batch AS name
            MERGE (m:Major {name: name})
            """

            with self.driver.session() as session:
                for i in range(0, len(all_majors), batch_size):
                    batch = all_majors[i : i + batch_size]
                    session.run(major_query, batch=batch)

            # Create BELONGS_TO_MAJOR relationships
            print(f"  Creating BELONGS_TO_MAJOR relationships...")
            major_rel_query = """
            UNWIND $batch AS item
            MATCH (d:Document {id: item.doc_id})
            MATCH (m:Major {name: item.major})
            MERGE (d)-[:BELONGS_TO_MAJOR]->(m)
            """

            major_rels = []
            for d in data:
                if d.get("major") and d["major"].strip():
                    major_rels.append({"doc_id": d["id"], "major": d["major"]})

            with self.driver.session() as session:
                for i in range(0, len(major_rels), batch_size):
                    batch = major_rels[i : i + batch_size]
                    session.run(major_rel_query, batch=batch)

            print(f"  ✅ Created {len(major_rels):,} BELONGS_TO_MAJOR relationships")

        # Extract unique degrees (filter null)
        all_degrees = set(d["degree"] for d in data if d.get("degree"))
        all_degrees = list(all_degrees)

        # Create Degree nodes
        if all_degrees:
            print(f"  Creating {len(all_degrees):,} Degree nodes...")
            degree_query = """
            UNWIND $batch AS name
            MERGE (deg:Degree {name: name})
            """

            with self.driver.session() as session:
                session.run(degree_query, batch=all_degrees)

            # Create HAS_DEGREE relationships
            print(f"  Creating HAS_DEGREE relationships...")
            degree_rel_query = """
            UNWIND $batch AS item
            MATCH (d:Document {id: item.doc_id})
            MATCH (deg:Degree {name: item.degree})
            MERGE (d)-[:HAS_DEGREE]->(deg)
            """

            degree_rels = []
            for d in data:
                if d.get("degree"):
                    degree_rels.append({"doc_id": d["id"], "degree": d["degree"]})

            with self.driver.session() as session:
                for i in range(0, len(degree_rels), batch_size):
                    batch = degree_rels[i : i + batch_size]
                    session.run(degree_rel_query, batch=batch)

            print(f"  ✅ Created {len(degree_rels):,} HAS_DEGREE relationships")

    def get_count(self, session, query):
        result = session.run(query)
        record = result.single()
        return record["count"] if record else 0

    def verify_import(self):
        """Verify import by counting nodes and relationships"""
        print("\n📊 Verifying import...")

        with self.driver.session() as session:
            # Count nodes
            node_counts = {
                "Document": self.get_count(
                    session, "MATCH (d:Document) RETURN count(d) AS count"
                ),
                "Person": self.get_count(
                    session, "MATCH (p:Person) RETURN count(p) AS count"
                ),
                "Topic": self.get_count(
                    session, "MATCH (t:Topic) RETURN count(t) AS count"
                ),
                "Major": self.get_count(
                    session, "MATCH (m:Major) RETURN count(m) AS count"
                ),
                "Degree": self.get_count(
                    session, "MATCH (deg:Degree) RETURN count(deg) AS count"
                ),
            }

            # Count relationships
            rel_counts = {
                "WROTE": self.get_count(
                    session, "MATCH ()-[r:WROTE]->() RETURN count(r) AS count"
                ),
                "ADVISED": self.get_count(
                    session, "MATCH ()-[r:ADVISED]->() RETURN count(r) AS count"
                ),
                "HAS_TOPIC": self.get_count(
                    session, "MATCH ()-[r:HAS_TOPIC]->() RETURN count(r) AS count"
                ),
                "BELONGS_TO_MAJOR": self.get_count(
                    session,
                    "MATCH ()-[r:BELONGS_TO_MAJOR]->() RETURN count(r) AS count",
                ),
                "HAS_DEGREE": self.get_count(
                    session, "MATCH ()-[r:HAS_DEGREE]->() RETURN count(r) AS count"
                ),
            }

        print("\n  Nodes:")
        for label, count in node_counts.items():
            print(f"    {label:15s}: {count:6,}")

        print("\n  Relationships:")
        for rel_type, count in rel_counts.items():
            print(f"    {rel_type:20s}: {count:6,}")

        total_nodes = sum(node_counts.values())
        total_rels = sum(rel_counts.values())

        print(f"\n  Total Nodes:         {total_nodes:6,}")
        print(f"  Total Relationships: {total_rels:6,}")

    def run(self, input_file=None, clear_existing=False):
        """Main import workflow"""
        print("=" * 80)
        print("🚀 NEO4J IMPORT - TLU METADATA")
        print("=" * 80)

        # Use script directory if no input_file provided
        if input_file is None:
            script_dir = Path(__file__).parent
            input_file = script_dir / "tlu_metadata_clean_v2.jsonl"
        elif not Path(input_file).is_absolute():
            # If relative path, make it relative to script directory
            script_dir = Path(__file__).parent
            input_file = script_dir / input_file

        # Convert to string for open()
        input_file_str = str(input_file)

        # Load data
        print(f"\n📂 Loading data from {input_file_str}...")
        data = []
        with open(input_file_str, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        print(f"✅ Loaded {len(data):,} records")

        # Clear existing data if requested
        if clear_existing:
            confirm = input(
                "\n⚠️  Are you sure you want to clear existing data? (yes/no): "
            )
            if confirm.lower() == "yes":
                self.clear_database()
            else:
                print("Skipping database clear...")

        # Execute import steps
        self.create_constraints_and_indexes()
        self.import_documents(data)
        self.import_persons_and_relationships(data)
        self.import_topics(data)
        self.import_majors_and_degrees(data)
        self.verify_import()

        print("✅ IMPORT COMPLETED SUCCESSFULLY!")


def load_config():
    import os
    from dotenv import load_dotenv
    from pathlib import Path

    # Get script directory and .env path
    script_dir = Path(__file__).parent
    env_file = script_dir / ".env"

    # Debug info
    print(f"🔍 Looking for .env at: {env_file}")
    print(f"   File exists: {env_file.exists()}")

    # Load with explicit path
    load_dotenv(dotenv_path=env_file)

    uri = os.getenv("NEO4J_URI")
    username = os.getenv("USERNAME_NEO4J")
    password = os.getenv("PASSWORD_NEO4J")

    # Debug what we got
    print(f"   NEO4J_URI: {uri}")
    print(f"   USERNAME_NEO4J:ư {username}")
    print(f"   PASSWORD_NEO4J: {'***' if password else None}")

    if not uri or not username or not password:
        print("❌ Missing environment variables!")
        print("\nPlease set the following in your .env file:")
        print("NEO4J_URI=neo4j+s://xxxxx.databases.neo4j.io")
        print("USERNAME_NEO4J=neo4j")
        print("PASSWORD_NEO4J=your-password")
        sys.exit(1)

    return {"uri": uri, "user": username, "password": password}


if __name__ == "__main__":
    config = load_config()

    importer = Neo4jImporter(
        uri=config["uri"], user=config["user"], password=config["password"]
    )

    try:
        importer.run(input_file="tlu_metadata_clean_v2.jsonl", clear_existing=False)
    finally:
        importer.close()
        print("\n👋 Connection closed")
