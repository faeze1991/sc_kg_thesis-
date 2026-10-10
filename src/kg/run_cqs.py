"""
run_cqs.py

Loads the pickled graph built by build_graph.py and runs CQ1, CQ2, CQ6, CQ7, CQ8
against real, automatically-selected examples from the graph itself (no need to
know company/product names in advance).

Usage (from the project root):
    python3 -m src.kg.run_cqs qwen
    python3 -m src.kg.run_cqs llama
(replaces the duplicate run_cqs_llama.py)
"""

import pickle
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.kg.build_graph import (
    cq1_products_manufactured_by,
    cq2_manufacturer_and_origin,
    cq6_component_chain,
    cq7_component_to_brand_chain,
    cq8_hidden_multihop_companies,
)

GRAPHS = {"qwen": "data/interim/kg_graph.gpickle",
          "llama": "data/interim/kg_graph_llama.gpickle"}
GRAPH_PATH = GRAPHS["qwen"]


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def pick_example_companies(G, n=3):
    """Pick companies with the most 'manufactures' out-edges as examples for CQ1."""
    from collections import Counter
    out_deg = Counter()
    for u, _, d in G.edges(data=True):
        if d["relation"] == "manufactures":
            out_deg[u] += 1
    top = out_deg.most_common(n)
    return [G.nodes[c]["name"] for c, _ in top]


def pick_example_products_with_manufacturer(G, n=3):
    """Pick products that have an incoming manufactures edge, for CQ2."""
    products = set()
    for u, v, d in G.edges(data=True):
        if d["relation"] == "manufactures":
            products.add(v)
        if len(products) >= n:
            break
    return [G.nodes[p]["name"] for p in list(products)[:n]]


def pick_example_components(G, n=5):
    """Pick products that are components (source of an is_component_of edge)."""
    components = set()
    for u, v, d in G.edges(data=True):
        if d["relation"] == "is_component_of":
            components.add(u)
        if len(components) >= n:
            break
    return [G.nodes[c]["name"] for c in list(components)[:n]]


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen"
    if model not in GRAPHS:
        sys.exit(f"usage: python3 -m src.kg.run_cqs [{'|'.join(GRAPHS)}]")
    GRAPH_PATH = GRAPHS[model]
    print(f"[INFO] Loading graph ({model}): {GRAPH_PATH}")
    G = load_graph()
    print(f"[INFO] Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- CQ1 ---
    print("\n" + "=" * 70)
    print("CQ1: For a given company, what products does it manufacture?")
    print("=" * 70)
    example_companies = pick_example_companies(G, n=3)
    for company in example_companies:
        print(f"\n-- {company} --")
        results = cq1_products_manufactured_by(G, company)
        for r in results:
            print(f"  manufactures: {r['product']}  "
                  f"(source_count={r['source_count']}, domains={r['domains'][:3]}{'...' if len(r['domains']) > 3 else ''})")

    # --- CQ2 ---
    print("\n" + "=" * 70)
    print("CQ2: For a given product, who manufactures it and where?")
    print("=" * 70)
    example_products = pick_example_products_with_manufacturer(G, n=3)
    for product in example_products:
        print(f"\n-- {product} --")
        result = cq2_manufacturer_and_origin(G, product)
        print(f"  manufacturers: {result['manufacturers']}")
        print(f"  made_in: {result['made_in']}")

    # --- CQ6 ---
    print("\n" + "=" * 70)
    print("CQ6: component -> parent product -> manufactures <- company")
    print("(2-hop chain: hidden manufacturer relationship via a component)")
    print("=" * 70)
    example_components = pick_example_components(G, n=10)
    found_cq6 = 0
    for comp in example_components:
        chains = cq6_component_chain(G, comp)
        if chains:
            found_cq6 += 1
            print(f"\n-- component: {comp} --")
            for c in chains:
                print(f"  {c['component']} -> is_component_of -> {c['parent_product']} "
                      f"<- manufactures <- {c['manufacturer']}")
    if found_cq6 == 0:
        print("\n[INFO] None of the first 10 sampled components had a manufactures "
              "edge on their parent product. Trying a larger sample...")
        example_components = pick_example_components(G, n=200)
        for comp in example_components:
            chains = cq6_component_chain(G, comp)
            if chains:
                found_cq6 += 1
                print(f"\n-- component: {comp} --")
                for c in chains:
                    print(f"  {c['component']} -> is_component_of -> {c['parent_product']} "
                          f"<- manufactures <- {c['manufacturer']}")
                if found_cq6 >= 5:
                    break
        if found_cq6 == 0:
            print("[INFO] No CQ6 chains found in the sampled components. This means "
                  "component and manufacturer relations rarely co-occur on the same "
                  "parent product in this graph -- worth noting as a coverage limitation.")

    # --- CQ7 ---
    print("\n" + "=" * 70)
    print("CQ7: component -> is_component_of -> parent product -> has_brand -> brand")
    print("=" * 70)
    example_components = pick_example_components(G, n=200)
    found_cq7 = 0
    for comp in example_components:
        chains = cq7_component_to_brand_chain(G, comp)
        if chains:
            found_cq7 += 1
            print(f"\n-- component: {comp} --")
            for c in chains:
                print(f"  {c['component']} -> is_component_of -> {c['parent_product']} "
                      f"-> has_brand -> {c['brand']}")
            if found_cq7 >= 5:
                break
    if found_cq7 == 0:
        print("\n[INFO] No CQ7 chains found in the sampled components.")

    # --- CQ8 ---
    print("\n" + "=" * 70)
    print("CQ8: Companies connected only via a shared product, never co-occurring")
    print("on the same source domain (hidden multi-source-inferred dependency)")
    print("=" * 70)
    hidden = cq8_hidden_multihop_companies(G, max_pairs_to_check=200000)
    print(f"\n[INFO] Found {len(hidden)} hidden company-pair connections "
          f"(checked up to 200,000 pairs).")
    for h in hidden[:10]:
        print(f"  via product '{h['product']}': {h['company_a']}  <-->  {h['company_b']}")
    if len(hidden) > 10:
        print(f"  ... and {len(hidden) - 10} more.")
