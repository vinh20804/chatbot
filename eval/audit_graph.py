"""Audit knowledge graph for dirty nodes and output a cleanup plan.

Categories (per rebut.md framework):
  - DELETE: Invalid (numbers, IPs, IMG markers) + Malformed (single letters) + Fragments
  - MERGE:  Duplicates (case-insensitive duplicates, group to canonical)
  - KEEP:   Overbroad (Data, System) — flagged but kept (may be valid in context)

Output: eval/cleanup_plan.json with two sections:
  - delete: list of node names to remove from graph + Postgres
  - merge:  list of {canonical, duplicates} groups to consolidate
"""
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
import networkx as nx

GRAPHML = "/home/tts/AI/aiQuoc/tlu_workspace/graph_chunk_entity_relation.graphml"
PLAN_OUT = Path("/home/tts/AI/aiQuoc/TLU-Chatbot/eval/cleanup_plan.json")

INVALID_PATTERNS = [
    (re.compile(r"^[\d\s\.\-]+$"), "pure-numbers/punct"),
    (re.compile(r"^(slide|page|trang|chương|chapter|hình|figure)\s*\d+$", re.I), "page/slide/figure marker"),
    (re.compile(r"\[IMG_|IMG_\w+", re.I), "image marker"),
    (re.compile(r"^(unknown|n/a|none|null|tbd|undefined)$", re.I), "placeholder/empty"),
    (re.compile(r"^[\W_]+$"), "punctuation-only"),
    # Rule 2 — file path nodes are not concepts
    (re.compile(r"\.(php|js|css|html|md|min)\b", re.I), "file-path-node"),
    (re.compile(r"^(routes|config|app)/", re.I), "file-path-node"),
]

# Rule 1 — allowed entity types per LightRAG schema
# NOTE: "unknown" is LightRAG's legit fallback for valid concepts the LLM
# couldn't classify (e.g. "Big Data Storage", "Phụ thuộc đa trị"). Whitelist it
# — do NOT flag as junk (an earlier run over-deleted ~15-20 valid concepts).
ALLOWED_ENTITY_TYPES = {
    "concept", "algorithm", "data_structure", "language",
    "framework", "tool", "architecture", "metric", "person", "organization",
    "unknown",
    "unknown",
}


def normalize(name: str) -> str:
    return re.sub(r"\s+", " ", name.lower().strip())


def pick_canonical(names: list[str]) -> str:
    """Among case-variant duplicates, pick the most 'named-looking' one.
    Heuristic: prefer the one with most uppercase letters (more like proper noun),
    tie-break by longest, then alphabetical."""
    return sorted(names, key=lambda n: (-sum(1 for c in n if c.isupper()), -len(n), n))[0]


def audit(graphml: str = GRAPHML, plan_out: Path = PLAN_OUT):
    print(f"Loading {graphml}...")
    G = nx.read_graphml(graphml)
    total = G.number_of_nodes()
    print(f"Nodes: {total} | Edges: {G.number_of_edges()}\n")

    delete_list = []  # [(name, category, reason)]
    by_norm = defaultdict(list)
    for node, attrs in G.nodes(data=True):
        name = str(node).strip()
        norm = normalize(name)
        by_norm[norm].append(name)

        # Check invalid
        invalid_reason = None
        for pat, reason in INVALID_PATTERNS:
            if pat.search(name):
                invalid_reason = reason
                break
        if invalid_reason:
            delete_list.append({"name": name, "category": "invalid", "reason": invalid_reason})
            continue

        # Rule 1 — entity_type must be in allowed schema
        entity_type = str(attrs.get("entity_type", "")).strip().lower()
        if entity_type and entity_type not in ALLOWED_ENTITY_TYPES:
            delete_list.append({
                "name": name,
                "category": "invalid",
                "reason": f"invalid-entity-type({entity_type})",
            })
            continue

        # Malformed
        if len(name) < 2:
            delete_list.append({"name": name, "category": "malformed", "reason": "too-short"})
        elif len(name) > 80:
            delete_list.append({"name": name, "category": "malformed", "reason": f"too-long({len(name)})"})
        elif re.search(r"[\.;]$|^(các|một|vài|nhiều)\s", name.lower()):
            delete_list.append({"name": name, "category": "fragment", "reason": "sentence-fragment"})

    # Build merge groups (multiple raw names → canonical)
    merge_list = []
    delete_set = {d["name"] for d in delete_list}
    for norm, names in by_norm.items():
        if len(names) <= 1:
            continue
        # Exclude any name already marked delete
        valid_names = [n for n in names if n not in delete_set]
        if len(valid_names) <= 1:
            continue
        canonical = pick_canonical(valid_names)
        dups = [n for n in valid_names if n != canonical]
        merge_list.append({"canonical": canonical, "duplicates": dups})

    plan = {
        "stats": {
            "total_nodes": total,
            "delete_count": len(delete_list),
            "merge_groups": len(merge_list),
            "merge_node_count": sum(len(g["duplicates"]) for g in merge_list),
        },
        "delete": delete_list,
        "merge": merge_list,
    }

    plan_out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== CLEANUP PLAN ===")
    print(f"  DELETE:        {len(delete_list)} nodes")
    print(f"    invalid:     {sum(1 for d in delete_list if d['category']=='invalid')}")
    print(f"      file-path: {sum(1 for d in delete_list if d.get('reason')=='file-path-node')}")
    print(f"      bad-type:  {sum(1 for d in delete_list if str(d.get('reason','')).startswith('invalid-entity-type'))}")
    print(f"    malformed:   {sum(1 for d in delete_list if d['category']=='malformed')}")
    print(f"    fragment:    {sum(1 for d in delete_list if d['category']=='fragment')}")
    print(f"  MERGE groups:  {len(merge_list)} groups → consolidate {sum(len(g['duplicates']) for g in merge_list)} duplicate nodes")
    print(f"\n[SAVED] {plan_out}")
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml", default=GRAPHML, help="GraphML file to audit")
    parser.add_argument("--out", default=str(PLAN_OUT), help="Cleanup plan output path")
    args = parser.parse_args()
    audit(args.graphml, Path(args.out))
