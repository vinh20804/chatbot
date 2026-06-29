"""End-to-end cleanup: refined audit + apply to graphml + Postgres (entity + relation tables).

Workflow:
  1. Read .graphml
  2. Refined audit:
     - DELETE candidates: IMG markers, low-utility single letters, exercise numbers, fragments
     - KEEP anchors: C (language), 1957, 1970, 127.0.0.1, strong identifiers
     - MERGE groups: case-fold duplicates
  3. Apply to graph (NetworkX): delete + merge with edge weight summation + desc dedup
  4. Apply to Postgres: 4 entity tables + 4 relation tables
  5. Output dry-run report or commit changes

Usage:
  python cleanup_graph.py              # dry-run, show counts
  python cleanup_graph.py --apply      # commit changes
"""
import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import networkx as nx
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

GRAPHML = "/home/tts/AI/aiQuoc/tlu_workspace/graph_chunk_entity_relation.graphml"
SEP = "<SEP>"

# --- Refined classification rules ---

# Always delete (clear noise, no query value)
ALWAYS_DELETE_PATTERNS = [
    (re.compile(r"\[?IMG_|^IMG_", re.I), "img-marker"),
    # Note: NOT deleting pure-punctuation nodes — many are valid operators like ==, ===, &&, regex ^, $, .*
    (re.compile(r"^(unknown|n/a|none|null|tbd|undefined)$", re.I), "placeholder"),
]

# Single letters that are NOT useful as graph anchors (their info is in chunks)
LOW_UTILITY_LETTERS = {"X", "Y", "a", "p", "E", "g", "i", "m", "x", "y", "Z"}

# Keep these despite single letter (real concepts)
KEEP_SINGLE_LETTERS = {"C"}  # programming language

# Numbers: keep if anchor (year with degree>=2 or strong identifier), else delete
STRONG_NUMBER_ANCHORS = {"127.0.0.1"}

# Fragment patterns: definitely delete (re-parented to canonical neighbor via merge)
FRAGMENT_PATTERNS = [
    (re.compile(r"[\.;]$"), "ends-with-period"),
    (re.compile(r"^(các|một|vài|nhiều)\s", re.I), "starts-with-quantifier"),
]


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip())


def is_number(name: str) -> bool:
    return bool(re.match(r"^[\d\.]+$", name))


def is_ip(name: str) -> bool:
    return bool(re.match(r"^\d+\.\d+\.\d+\.\d+$", name))


def is_year(name: str) -> bool:
    return bool(re.match(r"^(19|20)\d{2}$", name))


def classify_node(name: str, G) -> tuple[str, str]:
    """Return ('keep'|'delete'|'merge'-candidate, reason)."""
    # Always-delete patterns
    for pat, reason in ALWAYS_DELETE_PATTERNS:
        if pat.search(name):
            return "delete", reason

    # Single letters
    if len(name) == 1 and name.isalpha():
        if name in KEEP_SINGLE_LETTERS:
            return "keep", "single-letter-concept"
        if name in LOW_UTILITY_LETTERS:
            return "delete", "low-utility-letter"
        # Other single letters: default delete
        return "delete", "single-letter-other"

    # Strong number anchors
    if name in STRONG_NUMBER_ANCHORS:
        return "keep", "strong-number-anchor"

    # Years: keep if degree >= 2
    if is_year(name):
        deg = G.degree(name) if G.has_node(name) else 0
        if deg >= 2:
            return "keep", "year-anchor"
        return "keep", "year-singleton"  # keep yearly anchors even singletons

    # IPs: keep
    if is_ip(name):
        return "keep", "ip-anchor"

    # Pure numbers (decimals, binary): delete as exercise values
    if is_number(name):
        return "delete", "exercise-number"

    # Fragments
    if len(name) <= 80:
        for pat, reason in FRAGMENT_PATTERNS:
            if pat.search(name):
                return "delete", f"fragment-{reason}"

    # Too long
    if len(name) > 80:
        return "delete", "too-long"

    return "keep", "normal-entity"


def pick_canonical(names: list[str]) -> str:
    """For case-fold dup group, pick the most 'named-looking' variant."""
    return sorted(names, key=lambda n: (-sum(1 for c in n if c.isupper()), -len(n), n))[0]


def build_plan(G):
    """Build delete + merge plan from graph."""
    delete_list = []
    by_norm = defaultdict(list)
    for node in G.nodes():
        name = str(node).strip()
        decision, reason = classify_node(name, G)
        if decision == "delete":
            delete_list.append({"name": name, "reason": reason})
        by_norm[normalize(name)].append(name)

    delete_set = {d["name"] for d in delete_list}

    # Merge groups (excluding nodes already in delete)
    merge_list = []
    for norm, names in by_norm.items():
        valid = [n for n in names if n not in delete_set]
        if len(valid) <= 1:
            continue
        canonical = pick_canonical(valid)
        dups = [n for n in valid if n != canonical]
        merge_list.append({"canonical": canonical, "duplicates": dups})

    return delete_list, merge_list


# --- Description helpers ---

def merge_desc(existing: str, new: str) -> str:
    """Concat descriptions only if not duplicate."""
    existing = (existing or "").strip()
    new = (new or "").strip()
    if not new:
        return existing
    if not existing:
        return new
    # Split existing by <SEP>, check if new already inside
    parts = [p.strip() for p in existing.split(SEP) if p.strip()]
    if new in parts:
        return existing
    parts.append(new)
    return SEP.join(parts)


# --- Graph operations ---

def apply_graph(G, delete_list, merge_list):
    """Mutate G in-place. Return stats."""
    deleted = 0
    for d in delete_list:
        if G.has_node(d["name"]):
            G.remove_node(d["name"])
            deleted += 1

    merged_groups = 0
    edges_redirected = 0
    for group in merge_list:
        canonical = group["canonical"]
        if not G.has_node(canonical):
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
            # Merge desc
            G.nodes[canonical]["description"] = merge_desc(
                G.nodes[canonical].get("description"), G.nodes[dup].get("description")
            )
            # Redirect edges
            for nb in list(G.neighbors(dup)):
                if nb == canonical:
                    continue
                edata = G.edges[dup, nb].copy()
                w_dup = float(edata.get("weight", 1.0))
                if G.has_edge(canonical, nb):
                    G.edges[canonical, nb]["weight"] = float(G.edges[canonical, nb].get("weight", 1.0)) + w_dup
                    G.edges[canonical, nb]["description"] = merge_desc(
                        G.edges[canonical, nb].get("description"), edata.get("description")
                    )
                else:
                    G.add_edge(canonical, nb, **edata)
                edges_redirected += 1
            G.remove_node(dup)
    return {"deleted": deleted, "merged_groups": merged_groups, "edges_redirected": edges_redirected}


# --- Postgres operations ---

def connect_pg():
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"], port=os.environ["POSTGRES_PORT"],
        user=os.environ["POSTGRES_USER"], password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DATABASE"],
    )


ENTITY_VDB_TABLES = ["lightrag_vdb_entity", "lightrag_vdb_entity_microsoft_harrier_oss_v1_270m_640d"]
RELATION_VDB_TABLES = ["lightrag_vdb_relation", "lightrag_vdb_relation_microsoft_harrier_oss_v1_270m_640d"]


def apply_pg(cur, delete_names, merge_map, dry_run):
    """delete_names: set of node names to remove.
    merge_map: {dup_name: canonical_name}
    Returns stats dict.
    """
    stats = defaultdict(int)

    # ============ ENTITIES ============
    # 1. vdb_entity (small and embedding tables)
    for tbl in ENTITY_VDB_TABLES:
        # Delete
        if delete_names:
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE entity_name = ANY(%s)", (list(delete_names),))
            n = cur.fetchone()[0]
            stats[f"{tbl}_delete"] = n
            if not dry_run and n:
                cur.execute(f"DELETE FROM {tbl} WHERE entity_name = ANY(%s)", (list(delete_names),))

        # Merge: dedup-aware
        for dup, canonical in merge_map.items():
            cur.execute(f"SELECT 1 FROM {tbl} WHERE entity_name=%s LIMIT 1", (canonical,))
            has_canon = cur.fetchone() is not None
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE entity_name=%s", (dup,))
            n_dup = cur.fetchone()[0]
            if n_dup == 0:
                continue
            if has_canon:
                stats[f"{tbl}_merge_drop"] += n_dup
                if not dry_run:
                    cur.execute(f"DELETE FROM {tbl} WHERE entity_name=%s", (dup,))
            else:
                stats[f"{tbl}_merge_rename"] += n_dup
                if not dry_run:
                    cur.execute(f"UPDATE {tbl} SET entity_name=%s WHERE entity_name=%s", (canonical, dup))

    # 2. lightrag_entity_chunks (id = entity name)
    tbl = "lightrag_entity_chunks"
    if delete_names:
        cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE id = ANY(%s)", (list(delete_names),))
        n = cur.fetchone()[0]
        stats[f"{tbl}_delete"] = n
        if not dry_run and n:
            cur.execute(f"DELETE FROM {tbl} WHERE id = ANY(%s)", (list(delete_names),))

    for dup, canonical in merge_map.items():
        cur.execute(f"SELECT chunk_ids FROM {tbl} WHERE id=%s", (canonical,))
        canon_row = cur.fetchone()
        cur.execute(f"SELECT chunk_ids FROM {tbl} WHERE id=%s", (dup,))
        dup_row = cur.fetchone()
        if not dup_row:
            continue
        if canon_row:
            merged_chunks = list({*canon_row[0], *dup_row[0]})
            stats[f"{tbl}_merge_combine"] += 1
            if not dry_run:
                cur.execute(f"UPDATE {tbl} SET chunk_ids=%s, count=%s WHERE id=%s",
                            (Json(merged_chunks), len(merged_chunks), canonical))
                cur.execute(f"DELETE FROM {tbl} WHERE id=%s", (dup,))
        else:
            stats[f"{tbl}_merge_rename"] += 1
            if not dry_run:
                cur.execute(f"UPDATE {tbl} SET id=%s WHERE id=%s", (canonical, dup))

    # 3. lightrag_full_entities (JSONB entity_names per doc)
    tbl = "lightrag_full_entities"
    cur.execute(f"SELECT id, entity_names FROM {tbl}")
    for doc_id, names_json in cur.fetchall():
        names = names_json if isinstance(names_json, list) else (names_json or [])
        new_names = []
        changed = False
        for n in names:
            if n in delete_names:
                changed = True
                continue
            if n in merge_map:
                canonical = merge_map[n]
                if canonical not in new_names:
                    new_names.append(canonical)
                changed = True
            else:
                if n not in new_names:
                    new_names.append(n)
        if changed:
            stats[f"{tbl}_doc_updated"] += 1
            if not dry_run:
                cur.execute(f"UPDATE {tbl} SET entity_names=%s, count=%s WHERE id=%s",
                            (Json(new_names), len(new_names), doc_id))

    # ============ RELATIONS ============
    # 4. vdb_relation tables (source_id, target_id)
    for tbl in RELATION_VDB_TABLES:
        # Delete: any relation involving a deleted entity
        if delete_names:
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE source_id = ANY(%s) OR target_id = ANY(%s)",
                        (list(delete_names), list(delete_names)))
            n = cur.fetchone()[0]
            stats[f"{tbl}_delete"] = n
            if not dry_run and n:
                cur.execute(f"DELETE FROM {tbl} WHERE source_id = ANY(%s) OR target_id = ANY(%s)",
                            (list(delete_names), list(delete_names)))

        # Merge: rename src/tgt. Collisions deduplicated by keeping first.
        for dup, canonical in merge_map.items():
            # source_id rename
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE source_id=%s", (dup,))
            n_src = cur.fetchone()[0]
            if n_src:
                stats[f"{tbl}_rename_src"] += n_src
                if not dry_run:
                    # Avoid PK collision: delete dup rows where canonical already paired same target
                    cur.execute(
                        f"DELETE FROM {tbl} WHERE source_id=%s AND target_id IN "
                        f"(SELECT target_id FROM {tbl} WHERE source_id=%s)",
                        (dup, canonical))
                    cur.execute(f"UPDATE {tbl} SET source_id=%s WHERE source_id=%s", (canonical, dup))
            # target_id rename
            cur.execute(f"SELECT COUNT(*) FROM {tbl} WHERE target_id=%s", (dup,))
            n_tgt = cur.fetchone()[0]
            if n_tgt:
                stats[f"{tbl}_rename_tgt"] += n_tgt
                if not dry_run:
                    cur.execute(
                        f"DELETE FROM {tbl} WHERE target_id=%s AND source_id IN "
                        f"(SELECT source_id FROM {tbl} WHERE target_id=%s)",
                        (dup, canonical))
                    cur.execute(f"UPDATE {tbl} SET target_id=%s WHERE target_id=%s", (canonical, dup))

    # 5. lightrag_relation_chunks (id = 'src<SEP>tgt')
    tbl = "lightrag_relation_chunks"
    cur.execute(f"SELECT id FROM {tbl}")
    all_ids = [r[0] for r in cur.fetchall()]
    deletes_in_relchunks = []
    renames_in_relchunks = []
    for rid in all_ids:
        if SEP not in rid:
            continue
        src, tgt = rid.split(SEP, 1)
        if src in delete_names or tgt in delete_names:
            deletes_in_relchunks.append(rid)
            continue
        new_src = merge_map.get(src, src)
        new_tgt = merge_map.get(tgt, tgt)
        if new_src != src or new_tgt != tgt:
            renames_in_relchunks.append((rid, f"{new_src}{SEP}{new_tgt}"))
    stats[f"{tbl}_delete"] = len(deletes_in_relchunks)
    stats[f"{tbl}_rename"] = len(renames_in_relchunks)
    if not dry_run:
        for rid in deletes_in_relchunks:
            cur.execute(f"DELETE FROM {tbl} WHERE id=%s", (rid,))
        for old_rid, new_rid in renames_in_relchunks:
            cur.execute(f"SELECT chunk_ids FROM {tbl} WHERE id=%s", (new_rid,))
            target_row = cur.fetchone()
            cur.execute(f"SELECT chunk_ids FROM {tbl} WHERE id=%s", (old_rid,))
            src_row = cur.fetchone()
            if not src_row:
                continue
            if target_row:
                merged = list({*target_row[0], *src_row[0]})
                cur.execute(f"UPDATE {tbl} SET chunk_ids=%s, count=%s WHERE id=%s",
                            (Json(merged), len(merged), new_rid))
                cur.execute(f"DELETE FROM {tbl} WHERE id=%s", (old_rid,))
            else:
                cur.execute(f"UPDATE {tbl} SET id=%s WHERE id=%s", (new_rid, old_rid))

    # 6. lightrag_full_relations (JSONB relation_pairs per doc)
    tbl = "lightrag_full_relations"
    cur.execute(f"SELECT id, relation_pairs FROM {tbl}")
    for doc_id, pairs_json in cur.fetchall():
        pairs = pairs_json if isinstance(pairs_json, list) else (pairs_json or [])
        seen = set()
        new_pairs = []
        changed = False
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                new_pairs.append(pair)
                continue
            src, tgt = pair
            if src in delete_names or tgt in delete_names:
                changed = True
                continue
            new_src = merge_map.get(src, src)
            new_tgt = merge_map.get(tgt, tgt)
            if new_src != src or new_tgt != tgt:
                changed = True
            key = (new_src, new_tgt)
            if key in seen:
                changed = True
                continue
            seen.add(key)
            new_pairs.append([new_src, new_tgt])
        if changed:
            stats[f"{tbl}_doc_updated"] += 1
            if not dry_run:
                cur.execute(f"UPDATE {tbl} SET relation_pairs=%s, count=%s WHERE id=%s",
                            (Json(new_pairs), len(new_pairs), doc_id))

    return dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--merge-only", action="store_true", help="Phase 1: only MERGE duplicates, skip DELETE")
    parser.add_argument("--delete-only", action="store_true", help="Phase 2: only DELETE, skip MERGE (assumes Phase 1 already done)")
    args = parser.parse_args()

    print(f"Loading {GRAPHML}...")
    G = nx.read_graphml(GRAPHML)
    before_n, before_e = G.number_of_nodes(), G.number_of_edges()
    print(f"  Nodes: {before_n} | Edges: {before_e}\n")

    delete_list, merge_list = build_plan(G)
    if args.merge_only:
        delete_list = []
        print(f"[MODE] --merge-only: skip DELETE\n")
    elif args.delete_only:
        merge_list = []
        print(f"[MODE] --delete-only: skip MERGE\n")
    print(f"=== AUDIT PLAN ===")
    print(f"  DELETE: {len(delete_list)} nodes")
    by_reason = defaultdict(int)
    for d in delete_list:
        by_reason[d["reason"]] += 1
    for r, c in sorted(by_reason.items(), key=lambda x: -x[1]):
        print(f"    {r:<25} {c}")
    print(f"  MERGE groups: {len(merge_list)} ({sum(len(g['duplicates']) for g in merge_list)} dup nodes)")
    print()

    # Sample shown
    print("--- Sample DELETE ---")
    for d in delete_list[:10]:
        print(f"  [{d['reason']}] {d['name']!r}")
    print("\n--- Sample MERGE (top 5) ---")
    for g in merge_list[:5]:
        print(f"  '{g['canonical']}' ← {g['duplicates']}")

    # Apply graph
    print("\n=== GRAPH MUTATION (in-memory) ===")
    graph_stats = apply_graph(G, delete_list, merge_list)
    print(f"  Deleted nodes:     {graph_stats['deleted']}")
    print(f"  Merged groups:     {graph_stats['merged_groups']}")
    print(f"  Edges redirected:  {graph_stats['edges_redirected']}")
    print(f"  After: nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

    # Apply PG
    delete_names = {d["name"] for d in delete_list}
    merge_map = {dup: g["canonical"] for g in merge_list for dup in g["duplicates"]}

    print(f"\n=== POSTGRES MUTATION (dry_run={not args.apply}) ===")
    conn = connect_pg()
    cur = conn.cursor()
    pg_stats = apply_pg(cur, delete_names, merge_map, dry_run=not args.apply)
    for k, v in sorted(pg_stats.items()):
        if v > 0:
            print(f"  {k:<60} {v}")

    if args.apply:
        # Backup graphml
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = GRAPHML.replace(".graphml", f"_backup_{ts}.graphml")
        shutil.copy2(GRAPHML, backup)
        # Write graph
        nx.write_graphml(G, GRAPHML)
        # Commit PG
        conn.commit()
        print(f"\n[APPLIED]")
        print(f"  Graph backup: {backup}")
        print(f"  Graph saved:  {GRAPHML}")
        print(f"  PG committed.")
        print(f"  RESTART SERVER: bash siu.sh")
    else:
        conn.rollback()
        print(f"\n[DRY-RUN] No changes. Re-run with --apply to commit.")
    conn.close()


if __name__ == "__main__":
    main()
