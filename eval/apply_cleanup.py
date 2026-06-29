"""Apply cleanup plan to BOTH graphml file AND Postgres entity storages.

Reads eval/cleanup_plan.json (produced by audit_graph.py) and:
  1. Backups graphml + dumps current entity row counts
  2. For each DELETE node: drop from graph + Postgres entity tables
  3. For each MERGE group: redirect dups → canonical, sum edge weights,
     update Postgres rows from dup_name to canonical_name

Postgres tables touched:
  - lightrag_vdb_entity                                  (vector store, entity_name col)
  - lightrag_vdb_entity_microsoft_harrier_oss_v1_270m_640d (vector store, entity_name col)
  - lightrag_full_entities                                (per-doc entities, JSONB)
  - lightrag_entity_chunks                                (entity ↔ chunk mapping, id col)
  - lightrag_vdb_relation*                                (relation vector store, source_id/target_id)
  - lightrag_full_relations                               (per-doc relations, JSONB)

DRY-RUN by default. Pass --apply to actually mutate.
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import networkx as nx
from dotenv import load_dotenv

load_dotenv("/home/tts/AI/aiQuoc/TLU-Chatbot/.env")

GRAPHML = "/home/tts/AI/aiQuoc/tlu_workspace/graph_chunk_entity_relation.graphml"
PLAN = "/home/tts/AI/aiQuoc/TLU-Chatbot/eval/cleanup_plan.json"

# Postgres entity tables to update (name column).
# Check actual schemas first before applying.
ENTITY_TABLES = [
    ("lightrag_vdb_entity", "entity_name"),
    ("lightrag_vdb_entity_microsoft_harrier_oss_v1_270m_640d", "entity_name"),
    ("lightrag_full_entities", None),   # JSONB content, special handling
    ("lightrag_entity_chunks", "id"),   # column is "id" not "entity_name"
]

# Relation tables: delete rows where source_id OR target_id is a deleted entity
RELATION_TABLES = [
    "lightrag_vdb_relation",
    "lightrag_vdb_relation_microsoft_harrier_oss_v1_270m_640d",
]


def connect_pg():
    import psycopg2
    return psycopg2.connect(
        host=os.environ['POSTGRES_HOST'],
        port=os.environ['POSTGRES_PORT'],
        user=os.environ['POSTGRES_USER'],
        password=os.environ['POSTGRES_PASSWORD'],
        dbname=os.environ['POSTGRES_DATABASE'],
    )


def inspect_schemas():
    """Print actual columns of each entity table."""
    conn = connect_pg()
    cur = conn.cursor()
    print("\n=== Entity table schemas (for sanity) ===")
    for table, _ in ENTITY_TABLES:
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
        """, (table,))
        cols = cur.fetchall()
        if cols:
            print(f"  {table}: {cols}")
        else:
            print(f"  {table}: (table not found)")
    conn.close()


def apply_graph_cleanup(plan, dry_run: bool):
    print(f"\n=== Graph cleanup (dry_run={dry_run}) ===")
    G = nx.read_graphml(GRAPHML)
    before_nodes = G.number_of_nodes()
    before_edges = G.number_of_edges()

    # DELETE
    delete_names = [d["name"] for d in plan["delete"]]
    actually_deleted = 0
    for name in delete_names:
        if G.has_node(name):
            G.remove_node(name)
            actually_deleted += 1
    print(f"  DELETE: dropped {actually_deleted} nodes (plan: {len(delete_names)})")

    # MERGE
    merged_groups = 0
    merged_edges = 0
    for group in plan["merge"]:
        canonical = group["canonical"]
        if not G.has_node(canonical):
            # If canonical doesn't exist (e.g. was in delete), pick first dup
            available = [d for d in group["duplicates"] if G.has_node(d)]
            if not available:
                continue
            canonical = available[0]
            dups = available[1:]
        else:
            dups = [d for d in group["duplicates"] if G.has_node(d) and d != canonical]
        if not dups:
            continue
        merged_groups += 1
        for dup in dups:
            # Redirect all edges of dup → canonical
            for nb in list(G.neighbors(dup)):
                if nb == canonical:
                    continue
                edge_attr = G.edges[dup, nb].copy() if G.has_edge(dup, nb) else {}
                w_dup = float(edge_attr.get("weight", 1.0))
                if G.has_edge(canonical, nb):
                    w_existing = float(G.edges[canonical, nb].get("weight", 1.0))
                    G.edges[canonical, nb]["weight"] = w_existing + w_dup
                    # Concat descriptions
                    desc_old = G.edges[canonical, nb].get("description", "")
                    desc_new = edge_attr.get("description", "")
                    if desc_new and desc_new not in desc_old:
                        G.edges[canonical, nb]["description"] = (desc_old + "<SEP>" + desc_new).strip("<SEP>")
                else:
                    G.add_edge(canonical, nb, **edge_attr)
                merged_edges += 1
            # Merge node attributes (description, type)
            for attr in ("description", "entity_type"):
                v_old = G.nodes[canonical].get(attr, "")
                v_new = G.nodes[dup].get(attr, "")
                if v_new and v_new not in v_old:
                    G.nodes[canonical][attr] = (v_old + "<SEP>" + v_new).strip("<SEP>")
            G.remove_node(dup)
    print(f"  MERGE: consolidated {merged_groups} groups, redirected {merged_edges} edges")

    after_nodes = G.number_of_nodes()
    after_edges = G.number_of_edges()
    print(f"  Nodes: {before_nodes} → {after_nodes}  ({after_nodes-before_nodes:+d})")
    print(f"  Edges: {before_edges} → {after_edges}  ({after_edges-before_edges:+d})")

    if not dry_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = GRAPHML.replace(".graphml", f"_backup_{ts}.graphml")
        shutil.copy2(GRAPHML, backup_path)
        nx.write_graphml(G, GRAPHML)
        print(f"  [APPLIED] backup={backup_path}, wrote new graphml")
    else:
        print(f"  [DRY-RUN] no file written")


def apply_pg_cleanup(plan, dry_run: bool):
    print(f"\n=== Postgres cleanup (dry_run={dry_run}) ===")
    conn = connect_pg()
    cur = conn.cursor()

    delete_names = [d["name"] for d in plan["delete"]]
    merge_map = {dup: g["canonical"] for g in plan["merge"] for dup in g["duplicates"]}

    for table, name_col in ENTITY_TABLES:
        # Check table exists
        cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
        if cur.fetchone()[0] is None:
            print(f"  {table}: skip (not exist)")
            continue

        if name_col:
            # DELETE rows for invalid/malformed names
            if delete_names:
                placeholders = ",".join(["%s"] * len(delete_names))
                cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {name_col} IN ({placeholders})", tuple(delete_names))
                n_del = cur.fetchone()[0]
                print(f"  {table}.{name_col}: would DELETE {n_del} rows")
                if not dry_run and n_del > 0:
                    cur.execute(f"DELETE FROM {table} WHERE {name_col} IN ({placeholders})", tuple(delete_names))

            # MERGE: update dup → canonical, then delete dup rows whose canonical also exists
            n_updated = 0
            n_dropped = 0
            for dup, canonical in merge_map.items():
                # If canonical already has a row, just delete dup; else rename dup to canonical
                cur.execute(f"SELECT 1 FROM {table} WHERE {name_col}=%s LIMIT 1", (canonical,))
                has_canon = cur.fetchone() is not None
                if has_canon:
                    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {name_col}=%s", (dup,))
                    n = cur.fetchone()[0]
                    if n > 0:
                        if not dry_run:
                            cur.execute(f"DELETE FROM {table} WHERE {name_col}=%s", (dup,))
                        n_dropped += n
                else:
                    cur.execute(f"SELECT COUNT(*) FROM {table} WHERE {name_col}=%s", (dup,))
                    n = cur.fetchone()[0]
                    if n > 0:
                        if not dry_run:
                            cur.execute(f"UPDATE {table} SET {name_col}=%s WHERE {name_col}=%s", (canonical, dup))
                        n_updated += n
            print(f"  {table}.{name_col}: would UPDATE {n_updated} rows, DELETE {n_dropped} dup rows")
        else:
            # lightrag_full_entities has JSONB structure; skip for now
            print(f"  {table}: SKIP JSONB-based table (manual cleanup needed if necessary)")

    # Clean relation tables: remove relations involving deleted entities
    if delete_names:
        placeholders = ",".join(["%s"] * len(delete_names))
        for rel_table in RELATION_TABLES:
            cur.execute("SELECT to_regclass(%s)", (f"public.{rel_table}",))
            if cur.fetchone()[0] is None:
                print(f"  {rel_table}: skip (not exist)")
                continue
            cur.execute(
                f"SELECT COUNT(*) FROM {rel_table} WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                tuple(delete_names) * 2,
            )
            n_rel = cur.fetchone()[0]
            print(f"  {rel_table}: would DELETE {n_rel} relation rows")
            if not dry_run and n_rel > 0:
                cur.execute(
                    f"DELETE FROM {rel_table} WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
                    tuple(delete_names) * 2,
                )

    if not dry_run:
        conn.commit()
        print(f"  [APPLIED] Postgres changes committed")
    else:
        conn.rollback()
        print(f"  [DRY-RUN] no changes committed")
    conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually mutate (default is dry-run)")
    parser.add_argument("--inspect", action="store_true", help="Just print PG schemas")
    args = parser.parse_args()

    if args.inspect:
        inspect_schemas()
        return

    plan = json.load(open(PLAN, encoding="utf-8"))
    print(f"Loaded plan: {plan['stats']}")

    apply_graph_cleanup(plan, dry_run=not args.apply)
    apply_pg_cleanup(plan, dry_run=not args.apply)

    if not args.apply:
        print("\n[NOTE] Dry-run only. Re-run with --apply to commit changes.")
        print("       Don't forget to restart server after applying:  bash siu.sh")


if __name__ == "__main__":
    main()
