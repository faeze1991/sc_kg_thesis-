"""
build_graph.py

Constructs the supply chain Knowledge Graph (NetworkX MultiDiGraph) from
data/interim/extraction_results_normalized_FINAL.jsonl.

Node types : company, product, location
Edge types (relations):
    manufactures     : company  -> product   (from manufactures == "yes")
    distributed_by   : company  -> product   (distributor company -> product)
    made_in          : product  -> location
    is_component_of  : product  -> product   (component -> parent product)

Design decisions (documented here for the methodology chapter):

1. Product node identity is GLOBAL by cleaned product string (no company
   scoping). This matches the "light cleanup only, no merging" decision
   already made for product/is_component_of during normalization. Risk:
   generic product names (e.g. "Q3", "Furniture") can collide across
   unrelated companies. This is monitored, not silently accepted -- see
   `diagnose_manufactures_collisions()` below, which reports how often a
   single product node is claimed as "manufactures" by more than one
   distinct company. If that number is high, switching to company-scoped
   product IDs (company::product) is the fix; the hook is left in place
   in `make_product_id()` to make that change a one-line edit later.

2. `product` and `is_component_of` values have no _cleaned field in the
   source file (unlike company/distributed_by), so light cleaning
   (whitespace, case-preserving strip, punctuation trim) is applied here
   at graph-build time via `light_clean()`. This was NOT done in the
   normalization stage -- flagging so it's documented as a graph-
   construction-time step, not a normalization-stage step, in the thesis.

3. `made_in` has no separate cleaned field either. This script attempts to
   import your existing location alias map from
   src/normalization/location_aliases.py. If the import fails (e.g. the
   variable name differs from what's guessed below), it falls back to
   using the raw made_in string and prints a warning -- fix the import
   line if you see that warning.

4. Every edge carries provenance as attributes: source_count, domains
   (list), sample_urls (capped sample), and for manufactures edges,
   company_is_domain_fallback_rate (fraction of supporting records where
   the company field was just the domain restated -- useful for CQ4/CQ5
   plausibility checks).

Usage:
    python3 src/graph/build_graph.py

Outputs:
    data/interim/kg_graph.gpickle   (NetworkX MultiDiGraph, for reloading)
    data/interim/kg_graph.graphml   (for Gephi / external visualization)
    Prints summary stats + CQ1-CQ8 demonstration queries.
"""

import json
import pickle
import re
from collections import Counter, defaultdict

import networkx as nx

INPUT_PATH = "data/interim/extraction_results_normalized_FINAL.jsonl"
OUTPUT_PICKLE = "data/interim/kg_graph.gpickle"
OUTPUT_GRAPHML = "data/interim/kg_graph.graphml"

SAMPLE_URL_CAP = 5


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def light_clean(s):
    """Light cleanup for product / is_component_of strings: whitespace and
    stray punctuation only -- no fuzzy merging, no case changes."""
    if not s or s == "unknown":
        return None
    s = re.sub(r"\s+", " ", s).strip()
    s = s.strip(" -_/\\|,;:")
    return s if s else None


def load_location_aliases():
    """Try to import the existing location alias map from
    src/normalization/location_aliases.py under a few likely variable
    names. Falls back to an empty dict (i.e. no aliasing) with a warning
    if none are found -- fix the import below if this fires."""
    candidate_names = [
        "LOCATION_ALIASES",
        "ALIASES",
        "ALIAS_MAP",
        "LOCATION_ALIAS_MAP",
        "location_aliases",
    ]
    try:
        import src.normalization.location_aliases as loc_mod
    except ImportError:
        print("[WARN] Could not import src.normalization.location_aliases "
              "-- made_in values will be used raw (no alias normalization). "
              "Check that this script is run from the project root "
              "(/data/faezeh/sc_kg_thesis-).")
        return {}

    for name in candidate_names:
        if hasattr(loc_mod, name):
            aliases = getattr(loc_mod, name)
            if isinstance(aliases, dict):
                print(f"[INFO] Loaded location alias map "
                      f"'{name}' ({len(aliases)} entries).")
                return aliases

    print("[WARN] src.normalization.location_aliases imported, but no dict "
          f"found under expected names {candidate_names}. "
          "Edit load_location_aliases() with the correct variable name. "
          "made_in values will be used raw for now.")
    return {}


LOCATION_ALIASES = load_location_aliases()


def normalize_location(s):
    if not s or s == "unknown":
        return None
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    return LOCATION_ALIASES.get(s.lower(), s)


# ---------------------------------------------------------------------------
# Node ID resolution
# ---------------------------------------------------------------------------

def make_company_id(rec):
    name = (rec.get("company_canonical")
            or rec.get("company_cleaned")
            or rec.get("company"))
    name = light_clean(name) if name else None
    return f"company:{name}" if name else None, name


def make_distributor_id(rec):
    if rec.get("distributed_by", "unknown") == "unknown":
        return None, None
    name = (rec.get("distributed_by_canonical")
            or rec.get("distributed_by_cleaned")
            or rec.get("distributed_by"))
    name = light_clean(name) if name else None
    return f"company:{name}" if name else None, name


def make_product_id(rec, field="product"):
    """Global product identity by cleaned string.
    To switch to company-scoped product IDs later (fix for generic-name
    collisions, see module docstring point 1), change this to prefix the
    id with the company id, e.g. f"product:{company_name}::{name}"."""
    raw = rec.get(field)
    name = light_clean(raw)
    return f"product:{name}" if name else None, name


def make_location_id(rec):
    name = normalize_location(rec.get("made_in"))
    return f"location:{name}" if name else None, name


# Generic/category-level product terms that are too ambiguous to serve as an
# is_component_of PARENT node -- seeded from real collisions found via
# diagnose_manufactures_collisions() and CQ8 output (e.g. "vehicles" linking
# Kia/Nissan/Toyota/Lexus via unrelated dealership listings, "refrigerator"
# linking Samsung/Whirlpool/LG via unrelated appliance listings). Extend this
# list as you find more cases during review.
GENERIC_PARENT_STOPLIST = {
    "vehicles", "vehicle", "furniture", "refrigerator", "face mask",
    "hoodie", "car", "cars", "appliance", "appliances", "electronics", "new vehicle", "new vehicles",
    "accessories", "accessory", "parts", "part", "product", "products",
    "jewelry", "bracelet", "watch", "watches", "jacket", "apparel", "bag", "bags",
    "clutch", "chest of drawers", "used cars", "used vehicles", "clothing",
}


def is_generic_product_name(name):
    """Returns True if this product name is too generic/ambiguous to be
    trusted as an is_component_of parent (2-hop traversal target) OR as a
    shared-product anchor in CQ8's hidden-multihop check. Filters out
    stoplist matches only -- NOT single-word names in general, since many
    legitimate specific products are single tokens (e.g. 'iWALK2.0',
    model numbers). A blanket word-count rule was tried and reverted:
    it wrongly filtered real single-token product names. Extend
    GENERIC_PARENT_STOPLIST as you find more true generic-collision cases."""
    if not name:
        return True
    normalized = name.strip().lower()
    return normalized in GENERIC_PARENT_STOPLIST


# Backwards-compatible alias (old name used elsewhere in this file)
is_generic_parent_name = is_generic_product_name


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(input_path=INPUT_PATH):
    G = nx.MultiDiGraph()

    # edge_key = (source_id, target_id, relation) -> aggregation dict
    edge_agg = defaultdict(lambda: {
        "source_count": 0,
        "domains": set(),
        "sample_urls": [],
        "domain_fallback_flags": [],  # only meaningful for manufactures edges
    })

    n_records = 0
    n_skipped_no_product = 0
    n_skipped_generic_parent = [0]  # list for closure-style mutation

    with open(input_path) as f:
        for line in f:
            n_records += 1
            rec = json.loads(line)

            domain = rec.get("domain", "")
            url = rec.get("url", "")

            product_id, product_name = make_product_id(rec, "product")
            if not product_id:
                n_skipped_no_product += 1
                continue
            G.add_node(product_id, type="product", name=product_name)

            company_id, company_name = make_company_id(rec)
            if company_id:
                G.add_node(company_id, type="company", name=company_name)

            # --- manufactures: company -> product ---
            if company_id and str(rec.get("manufactures", "")).strip().lower() == "yes":
                key = (company_id, product_id, "manufactures")
                agg = edge_agg[key]
                agg["source_count"] += 1
                agg["domains"].add(domain)
                if len(agg["sample_urls"]) < SAMPLE_URL_CAP:
                    agg["sample_urls"].append(url)
                agg["domain_fallback_flags"].append(
                    bool(rec.get("company_is_domain_fallback", False))
                )

            # --- distributed_by: distributor_company -> product ---
            dist_id, dist_name = make_distributor_id(rec)
            if dist_id:
                G.add_node(dist_id, type="company", name=dist_name)
                key = (dist_id, product_id, "distributed_by")
                agg = edge_agg[key]
                agg["source_count"] += 1
                agg["domains"].add(domain)
                if len(agg["sample_urls"]) < SAMPLE_URL_CAP:
                    agg["sample_urls"].append(url)

            # --- made_in: product -> location ---
            loc_id, loc_name = make_location_id(rec)
            if loc_id:
                G.add_node(loc_id, type="location", name=loc_name)
                key = (product_id, loc_id, "made_in")
                agg = edge_agg[key]
                agg["source_count"] += 1
                agg["domains"].add(domain)
                if len(agg["sample_urls"]) < SAMPLE_URL_CAP:
                    agg["sample_urls"].append(url)

            # --- is_component_of: product -> parent product ---
            parent_id, parent_name = make_product_id(rec, "is_component_of")
            if parent_id and parent_id != product_id and not is_generic_parent_name(parent_name):
                G.add_node(parent_id, type="product", name=parent_name)
                key = (product_id, parent_id, "is_component_of")
                agg = edge_agg[key]
                agg["source_count"] += 1
                agg["domains"].add(domain)
            elif parent_id and is_generic_parent_name(parent_name):
                n_skipped_generic_parent[0] += 1
                if len(agg["sample_urls"]) < SAMPLE_URL_CAP:
                    agg["sample_urls"].append(url)

    # materialize aggregated edges
    for (src, tgt, relation), agg in edge_agg.items():
        edge_attrs = {
            "relation": relation,
            "source_count": agg["source_count"],
            "domains": sorted(agg["domains"]),
            "n_domains": len(agg["domains"]),
            "sample_urls": agg["sample_urls"],
        }
        if agg["domain_fallback_flags"]:
            edge_attrs["company_is_domain_fallback_rate"] = round(
                sum(agg["domain_fallback_flags"]) / len(agg["domain_fallback_flags"]), 3
            )
        G.add_edge(src, tgt, key=relation, **edge_attrs)

    print(f"[INFO] Processed {n_records} records "
          f"({n_skipped_no_product} skipped: no usable product string)")
    print(f"[INFO] Skipped {n_skipped_generic_parent[0]} is_component_of edges "
          f"where the parent product name was too generic "
          f"(single-word or in GENERIC_PARENT_STOPLIST) -- see "
          f"GENERIC_PARENT_STOPLIST in this script to review/extend.")
    return G


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def print_summary(G):
    print("\n=== GRAPH SUMMARY ===")
    node_types = Counter(nx.get_node_attributes(G, "type").values())
    print("Nodes by type:", dict(node_types))
    print("Total nodes:", G.number_of_nodes())

    relation_counts = Counter(
        d["relation"] for _, _, d in G.edges(data=True)
    )
    print("Edges by relation:", dict(relation_counts))
    print("Total edges (aggregated, deduped):", G.number_of_edges())


def diagnose_manufactures_collisions(G, min_companies=2):
    """For each product node, count distinct companies with a 'manufactures'
    edge pointing to it. >1 distinct manufacturer for the same product name
    is a plausibility red flag (generic-name collision or extraction noise) --
    unlike distributed_by, where many distributors per product is expected
    and normal."""
    product_manufacturers = defaultdict(set)
    for u, v, d in G.edges(data=True):
        if d["relation"] == "manufactures":
            product_manufacturers[v].add(u)

    collisions = {
        p: companies for p, companies in product_manufacturers.items()
        if len(companies) >= min_companies
    }
    print(f"\n[DIAGNOSTIC] {len(collisions)} / {len(product_manufacturers)} "
          f"products have >= {min_companies} distinct companies claiming "
          f"'manufactures'.")
    if collisions:
        print("Examples (first 5):")
        for p, companies in list(collisions.items())[:5]:
            pname = G.nodes[p]["name"]
            cnames = [G.nodes[c]["name"] for c in companies]
            print(f"  '{pname}' <- manufactured by: {cnames}")
    return collisions


# ---------------------------------------------------------------------------
# Competency Question (CQ) query functions
# ---------------------------------------------------------------------------

def cq1_products_manufactured_by(G, company_name):
    """CQ1: For a given company, what products does it manufacture,
    and from which domains was that relation extracted?"""
    cid = f"company:{company_name}"
    if cid not in G:
        return []
    results = []
    for _, v, d in G.out_edges(cid, data=True):
        if d["relation"] == "manufactures":
            results.append({
                "product": G.nodes[v]["name"],
                "source_count": d["source_count"],
                "domains": d["domains"],
            })
    return results


def cq2_manufacturer_and_origin(G, product_name):
    """CQ2: For a given product, who manufactures it, and where (made_in)?"""
    pid = f"product:{product_name}"
    if pid not in G:
        return {"manufacturers": [], "made_in": []}
    manufacturers = [
        G.nodes[u]["name"] for u, _, d in G.in_edges(pid, data=True)
        if d["relation"] == "manufactures"
    ]
    made_in = [
        G.nodes[v]["name"] for _, v, d in G.out_edges(pid, data=True)
        if d["relation"] == "made_in"
    ]
    return {"manufacturers": manufacturers, "made_in": made_in}


def cq3_relation_distribution(G):
    """CQ3: Distribution of relation types across the graph."""
    return Counter(d["relation"] for _, _, d in G.edges(data=True))


def cq4_high_out_degree_manufacturers(G, top_n=10):
    """CQ4: Companies with implausibly high 'manufactures' out-degree
    (possible normalization miss or hallucination)."""
    out_deg = Counter()
    for u, _, d in G.edges(data=True):
        if d["relation"] == "manufactures":
            out_deg[u] += 1
    top = out_deg.most_common(top_n)
    return [(G.nodes[c]["name"], count) for c, count in top]


def cq5_made_in_conflicts(G):
    """CQ5: Products claimed to be made_in more than one distinct location."""
    product_locations = defaultdict(set)
    for u, v, d in G.edges(data=True):
        if d["relation"] == "made_in":
            product_locations[u].add(v)
    conflicts = {p: locs for p, locs in product_locations.items() if len(locs) > 1}
    return {
        G.nodes[p]["name"]: [G.nodes[l]["name"] for l in locs]
        for p, locs in conflicts.items()
    }


def cq6_component_chain(G, component_product_name):
    """CQ6: component -> is_component_of -> parent product -> manufactures <- company.
    Returns the 2-hop chain revealing a hidden manufacturer relationship."""
    cid = f"product:{component_product_name}"
    if cid not in G:
        return []
    chains = []
    for _, parent, d in G.out_edges(cid, data=True):
        if d["relation"] != "is_component_of":
            continue
        parent_name = G.nodes[parent]["name"]
        for manufacturer, _, d2 in G.in_edges(parent, data=True):
            if d2["relation"] == "manufactures":
                chains.append({
                    "component": component_product_name,
                    "parent_product": parent_name,
                    "manufacturer": G.nodes[manufacturer]["name"],
                })
    return chains


def cq7_component_to_distributor_chain(G, component_product_name):
    """CQ7: component -> parent product -> distributed_by <- distributor."""
    cid = f"product:{component_product_name}"
    if cid not in G:
        return []
    chains = []
    for _, parent, d in G.out_edges(cid, data=True):
        if d["relation"] != "is_component_of":
            continue
        parent_name = G.nodes[parent]["name"]
        for distributor, _, d2 in G.in_edges(parent, data=True):
            if d2["relation"] == "distributed_by":
                chains.append({
                    "component": component_product_name,
                    "parent_product": parent_name,
                    "distributor": G.nodes[distributor]["name"],
                })
    return chains


def cq8_hidden_multihop_companies(G, max_pairs_to_check=None):
    """CQ: Companies connected only through >=2 hops via shared products
    (never co-occurring on the same source domain directly) -- i.e. cases
    where the KG reveals a dependency no single page states explicitly.

    Approach: for each product, collect the set of companies linked to it
    via manufactures/distributed_by. Any two companies sharing a product
    but never appearing together in the same record's domain set are
    flagged as a "hidden" (multi-source-inferred) connection.
    """
    product_companies = defaultdict(set)  # product -> {(company_id, domains_frozenset)}
    for u, v, d in G.edges(data=True):
        if d["relation"] in ("manufactures", "distributed_by"):
            product_name = G.nodes[v].get("name", "")
            if is_generic_product_name(product_name):
                continue
            product_companies[v].add((u, frozenset(d["domains"])))

    hidden_links = []
    checked = 0
    for product, companies in product_companies.items():
        companies = list(companies)
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                c1, domains1 = companies[i]
                c2, domains2 = companies[j]
                if c1 == c2:
                    continue
                checked += 1
                if max_pairs_to_check and checked > max_pairs_to_check:
                    return hidden_links
                if domains1.isdisjoint(domains2):
                    hidden_links.append({
                        "product": G.nodes[product]["name"],
                        "company_a": G.nodes[c1]["name"],
                        "company_b": G.nodes[c2]["name"],
                    })
    return hidden_links


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    G = build_graph()
    print_summary(G)
    diagnose_manufactures_collisions(G)

    with open(OUTPUT_PICKLE, "wb") as f:
        pickle.dump(G, f)
    print(f"\n[INFO] Saved graph to {OUTPUT_PICKLE}")

    try:
        nx.write_graphml(G, OUTPUT_GRAPHML)
        print(f"[INFO] Saved graph to {OUTPUT_GRAPHML}")
    except Exception as e:
        print(f"[WARN] Could not write GraphML (likely a non-primitive "
              f"attribute type, e.g. list): {e}")

    print("\n=== CQ3: relation type distribution ===")
    print(cq3_relation_distribution(G))

    print("\n=== CQ4: top 10 companies by manufactures out-degree ===")
    for name, count in cq4_high_out_degree_manufacturers(G):
        print(f"  {name}: {count}")

    print("\n=== CQ5: made_in conflicts (first 5) ===")
    conflicts = cq5_made_in_conflicts(G)
    for p, locs in list(conflicts.items())[:5]:
        print(f"  {p}: {locs}")
    print(f"  Total products with conflicting made_in: {len(conflicts)}")
