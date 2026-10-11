"""
build_kg.py -- supply chain knowledge graphs with evidence-based product identity.
Replaces build_graph.py + run_cqs.py. No LLM calls.

Run from the project root:
    python3 -m src.kg.build_kg                 # all 4 configurations x {full, common}
    python3 -m src.kg.build_kg qwen_single     # one configuration, both page sets

Configurations (normalized extraction outputs, see normalize_entities.py):
    qwen_single  extraction_results_normalized_FINAL.jsonl
    qwen_iter    extraction_results_iterative_normalized.jsonl
    llama_single extraction_results_llama70b_normalized.jsonl
    llama_iter   extraction_results_iterative_llama_normalized.jsonl
Page sets:
    full    every page the configuration produced a record for
    common  only pages for which ALL FOUR configurations produced a record
            (fair comparison for RQ3; full-output statistics are reported separately)

IDENTITY RULES (identical for every configuration)
  Every page's product is a SOURCE RECORD. Source records are linked to a shared
  CANONICAL product only when the evidence supports it:
    route 1  identifier: same WDC cluster_id on >= 2 of our pages, no conflicting
             brand evidence among the members (missing brand is not a conflict),
             AND every merged member has a sufficiently specific product name.
             Members with a generic / unspecific name or with internal brand
             conflict stay separate source nodes (their cluster_id is kept as
             metadata); if fewer than 2 members remain, no merge. Added after the
             graph audit: a shared cluster must not bypass the identity checks.
             This WITHHOLDS merges with insufficient evidence -- it does not claim
             the WDC cluster itself is wrong.
    route 2  same sufficiently specific normalized product name AND the same
             brand evidence (exact normalized brand key), generic names excluded.
    "Sufficiently specific" (routes 1 and 2): after removing category/descriptor
             words (GENERIC_WORDS) and brand tokens, a model-like token or at least
             two distinctive words remain. Tightened after the graph audit.
    route 3  same specific name containing a model-like token but NO brand
             evidence: logged as a CANDIDATE only, never merged.
  Missing brand information = insufficient evidence, not permission to merge.
  Brand evidence = the extracted has_brand (stored as distributed_by) and the WDC
  schema.org brand, recorded separately; spelling variants are normalized and a
  substantive disagreement between the two is flagged as a conflict.
  The operator with manufactures = yes is MANUFACTURER evidence, not brand evidence.

  Component parents (is_component_of): a parent is merged across pages only if
  its name is a SPECIFIC identity (contains a model-like token, >= 2 tokens, not
  generic). It is linked to an existing product node only if exactly one product
  node has that exact normalized name (otherwise it stays a shared parent node,
  ambiguity logged). Product families / unspecific parents (e.g. "Nissan Patrol")
  stay page-scoped and are never equated with a model. Generic parents
  (GENERIC_STOPLIST) produce no edge.

LOCATIONS
  made_in is resolved to a country (pycountry + project alias map), a
  subdivision (e.g. California) or a region (e.g. Europe, EU). Anything else
  ("Locally", descriptive phrases, unknown cities) creates NO location node; the
  raw value is kept on the product as made_in_unresolved and counted.
  Several made_in locations for one product are reported as MULTIPLE ORIGIN
  CLAIMS, not as contradictions.

RELATIONS
  manufactures     company -> product   (LLM, manufactures == yes)
  has_brand        product -> company   (LLM, stored as distributed_by)
  made_in          product -> location  (LLM, resolved locations only)
  is_component_of  product -> product   (LLM)
  offers           company -> product   (PAGE-CONTEXT-DERIVED, not LLM-extracted:
                   the site operator offers the page's product; created only when a
                   product was extracted and the operator is identified)

OUTPUTS (data/interim/kg/)
  kg_<config>_<set>.gpickle         graph
  entity_matches_<config>_<set>.csv accepted merges, rejected clusters, candidates
  cq_<config>_<set>.txt             CQ1-CQ8 results
  summary.csv                       one row per configuration and page set
"""

import csv
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache

import networkx as nx

try:
    import pycountry
except ImportError:  # pragma: no cover
    sys.exit("pycountry is required: pip install pycountry")

from src.normalization.location_aliases import LOCATION_ALIASES

D = "data/interim"
OUT = f"{D}/kg"
WDC_FIELDS = f"{D}/wdc_fields.jsonl"
CONFIGS = {
    "qwen_single": f"{D}/extraction_results_normalized_FINAL.jsonl",
    "qwen_iter": f"{D}/extraction_results_iterative_normalized.jsonl",
    "llama_single": f"{D}/extraction_results_llama70b_normalized.jsonl",
    "llama_iter": f"{D}/extraction_results_iterative_llama_normalized.jsonl",
}
SAMPLE_URL_CAP = 5
NEG = {"", "unknown", "none", "null", "n/a", "na"}

GENERIC_STOPLIST = {
    "vehicles", "vehicle", "furniture", "refrigerator", "face mask", "hoodie", "car",
    "cars", "appliance", "appliances", "electronics", "new vehicle", "new vehicles",
    "accessories", "accessory", "parts", "part", "product", "products", "jewelry",
    "jewellery", "bracelet", "watch", "watches", "jacket", "apparel", "bag", "bags",
    "clutch", "chest of drawers", "used cars", "used vehicles", "clothing", "laptop",
    "smartphone", "phone", "guitar", "printer", "camera", "tablet", "bike", "bicycle",
    "washing machine", "washing machines", "sewing machine", "candle", "painting",
    "book", "t-shirt", "shirt", "shoes", "pillow", "mask", "toy", "toys",
}
LEGAL = r"\b(inc|ltd|llc|gmbh|co|corp|corporation|company|ag|sa|bv|plc|pty|srl|spa|sas|kg|limited)\b"
REGIONS = {
    "europe": "Europe", "europe (unspecified)": "Europe", "eu": "European Union",
    "the eu": "European Union", "european union": "European Union", "asia": "Asia",
    "scandinavia": "Scandinavia", "north america": "North America",
    "south america": "South America", "africa": "Africa", "middle east": "Middle East",
}


# --------------------------------------------------------------------- helpers
def val(v):
    s = "" if v is None else str(v).strip()
    return None if s.lower() in NEG else s


def norm_name(s):
    s = (s or "").lower().translate(str.maketrans("", "", "®™©"))
    s = re.sub(r"[^a-z0-9äöüßéèàçñ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def has_model_token(n):
    return any(re.search(r"\d", t) and len(t) >= 2 for t in n.split())


def is_generic(n):
    return (not n) or n in GENERIC_STOPLIST


# Category / descriptor words that do not identify a product on their own
# (added after the graph audit: names like "Honda vehicles", "premium luxury car",
# "Plush Toy" or "vehicle (specific model not specified)" passed the old 2-token test).
GENERIC_WORDS = {
    "vehicle", "vehicles", "car", "cars", "auto", "autos", "automobile", "truck", "trucks",
    "suv", "suvs", "motorcycle", "motorcycles", "bike", "bikes", "new", "used", "certified",
    "pre", "owned", "preowned", "premium", "luxury", "specific", "model", "models", "not",
    "specified", "unspecified", "product", "products", "item", "items", "toy", "toys",
    "plush", "module", "modules", "jersey", "jerseys", "tv", "tvs", "television", "the",
    "and", "of", "for", "with", "a", "an", "in", "range", "line", "collection", "series",
    "accessories", "parts", "brand", "official", "genuine", "original",
}


def distinctive_tokens(n, brand=None):
    """Tokens that can identify a product: not a category/descriptor word and not
    part of the brand name."""
    return [t for t in n.split()
            if t not in GENERIC_WORDS and not (brand and len(t) >= 2 and t in brand)]


def is_specific(n, brand=None):
    """Sufficiently specific product name (routes 1 and 2): not generic, and after
    removing category words and brand tokens either a model-like token (letters/digits)
    or at least two distinctive words remain."""
    if is_generic(n) or len(n) < 6:
        return False
    d = distinctive_tokens(n, brand)
    return any(re.search(r"\d", t) and len(t) >= 2 for t in d) or len(d) >= 2


def is_specific_parent(n):
    """Specific parent identity: model-like token, >= 2 tokens, not generic."""
    return (not is_generic(n)) and len(n.split()) >= 2 and has_model_token(n)


def brand_key(s):
    s = (s or "").lower().translate(str.maketrans("", "", "®™©"))
    s = re.sub(LEGAL, " ", s)
    return re.sub(r"[^a-z0-9]", "", s)


def brands_compatible(a, b):
    if not a or not b:
        return True  # missing information is not a conflict
    return a == b or (min(len(a), len(b)) >= 3 and (a in b or b in a))


# --------------------------------------------------------------------- locations
_SUBDIV = {}
for sd in pycountry.subdivisions:
    _SUBDIV.setdefault(sd.name.lower(), sd.name)


@lru_cache(maxsize=None)
def _resolve_location_cached(raw):
    """Returns (canonical_name, kind) or (None, 'unresolved')."""
    s = val(raw)
    if not s:
        return None, "none"
    low = s.lower().strip(" .")
    if low in ("imported (unspecified)", "imported", "locally", "local"):
        return None, "unresolved"
    candidates = [low] + [p.strip(" .") for p in re.split(r"[,()/;]| - ", low) if p.strip(" .")]
    best = None
    for c in candidates:
        if c in LOCATION_ALIASES and "unspecified" not in LOCATION_ALIASES[c].lower():
            return LOCATION_ALIASES[c], "country"
        try:
            return pycountry.countries.lookup(c).name, "country"
        except LookupError:
            pass
        if c in REGIONS and best is None:
            best = (REGIONS[c], "region")
        if c in _SUBDIV and (best is None or best[1] == "region"):
            best = (_SUBDIV[c], "subdivision")
    return best if best else (None, "unresolved")


def resolve_location(raw):
    return _resolve_location_cached(None if raw is None else str(raw))


# --------------------------------------------------------------------- loading
def load_records(path):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            d = str(r.get("domain", "")).strip()
            if d and d not in recs:
                recs[d] = r
    return recs


def load_wdc():
    w = {}
    if not os.path.exists(WDC_FIELDS):
        print(f"[WARN] {WDC_FIELDS} missing: identifier route and WDC brand evidence disabled")
        return w
    with open(WDC_FIELDS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            w[r["domain"]] = r
    return w


def company_name(rec):
    return val(rec.get("company_canonical") or rec.get("company_cleaned") or rec.get("company"))


def brand_name(rec):
    if not val(rec.get("distributed_by")):
        return None
    return val(rec.get("distributed_by_canonical") or rec.get("distributed_by_cleaned")
               or rec.get("distributed_by"))


# --------------------------------------------------------------------- identity
def resolve_products(recs, wdc):
    """Assign every page with a product to a node id. Returns (node_of, rows, stats)."""
    pages = {}
    for d, r in recs.items():
        p = val(r.get("product"))
        if not p:
            continue
        llm_b = brand_key(brand_name(r) or "")
        wdc_b = brand_key(val((wdc.get(d) or {}).get("wdc_brand")) or "")
        conflict = bool(llm_b and wdc_b and not brands_compatible(llm_b, wdc_b))
        pages[d] = {"name": p, "norm": norm_name(p), "llm_brand": llm_b, "wdc_brand": wdc_b,
                    "brand": None if conflict else (llm_b or wdc_b), "conflict": conflict,
                    "cluster": (wdc.get(d) or {}).get("cluster_id")}
    node_of, rows = {}, []
    st = Counter(pages_with_product=len(pages),
                 page_brand_conflicts=sum(p["conflict"] for p in pages.values()))

    # route 1: shared WDC cluster, no conflicting brand evidence
    clusters = defaultdict(list)
    for d, p in pages.items():
        if p["cluster"] is not None:
            clusters[p["cluster"]].append(d)
    for cid, members in clusters.items():
        if len(members) < 2:
            continue
        unspecific = [d for d in members if not is_specific(pages[d]["norm"], pages[d]["brand"])]
        ok = [d for d in members if not pages[d]["conflict"] and d not in unspecific]
        brands = {pages[d]["brand"] for d in ok if pages[d]["brand"]}
        conflict = any(not brands_compatible(a, b) for a in brands for b in brands)
        if unspecific:
            st["route1_members_withheld_unspecific"] += len(unspecific)
        if conflict or len(ok) < 2:
            st["route1_clusters_rejected"] += 1
            if conflict:
                reason, key = "conflicting brand evidence", "route1_rejected_brand_conflict"
            elif unspecific and len([d for d in members if not pages[d]["conflict"]]) >= 2:
                reason, key = "insufficiently specific product names", "route1_rejected_unspecific"
            else:
                reason, key = "members with internal brand conflict", "route1_rejected_member_conflict"
            st[key] += 1
            rows.append({"decision": "rejected", "route": "1_identifier", "key": f"cluster:{cid}",
                         "reason": reason, "n_pages": len(members),
                         "domains": " | ".join(members[:20]),
                         "names": " | ".join(pages[d]["name"] for d in members[:5]),
                         "brands": " | ".join(sorted(brands))})
            continue
        if len(ok) < len(members):
            rows.append({"decision": "partial", "route": "1_identifier", "key": f"cluster:{cid}",
                         "reason": "members with unspecific name or brand conflict kept separate",
                         "n_pages": len(members) - len(ok),
                         "domains": " | ".join(d for d in members if d not in ok)[:400],
                         "names": " | ".join(pages[d]["name"] for d in members if d not in ok)[:400],
                         "brands": ""})
        nid = f"product:canon:cluster:{cid}"
        for d in ok:
            node_of[d] = nid
        st["route1_clusters_accepted"] += 1
        st["route1_pages_merged"] += len(ok)
        rows.append({"decision": "accepted", "route": "1_identifier", "key": f"cluster:{cid}",
                     "reason": "same WDC cluster_id, no brand conflict", "n_pages": len(ok),
                     "domains": " | ".join(ok[:20]),
                     "names": " | ".join(pages[d]["name"] for d in ok[:5]),
                     "brands": " | ".join(sorted(brands))})

    # route 2: same specific name + same brand evidence
    groups = defaultdict(list)
    name_brands = defaultdict(set)
    for d, p in pages.items():
        if d in node_of or not is_specific(p["norm"], p["brand"]):
            continue
        if p["brand"]:
            groups[(p["norm"], p["brand"])].append(d)
            name_brands[p["norm"]].add(p["brand"])
    for (n, b), members in groups.items():
        if len(members) < 2:
            continue
        nid = f"product:canon:name:{n}|brand:{b}"
        for d in members:
            node_of[d] = nid
        st["route2_groups_accepted"] += 1
        st["route2_pages_merged"] += len(members)
        rows.append({"decision": "accepted", "route": "2_name_brand", "key": f"{n} | {b}",
                     "reason": "same specific name and same brand evidence", "n_pages": len(members),
                     "domains": " | ".join(members[:20]),
                     "names": " | ".join(pages[d]["name"] for d in members[:5]), "brands": b})
    for n, bs in name_brands.items():
        if len(bs) >= 2:
            st["route2_names_with_different_brands"] += 1
            rows.append({"decision": "rejected", "route": "2_name_brand", "key": n,
                         "reason": "same name, different brands", "n_pages": "",
                         "domains": "", "names": n, "brands": " | ".join(sorted(bs))})

    # route 3: model-like token without brand -> candidate only
    cand = defaultdict(list)
    for d, p in pages.items():
        if d in node_of or p["brand"] or not is_specific(p["norm"]) or not has_model_token(p["norm"]):
            continue
        cand[p["norm"]].append(d)
    for n, members in cand.items():
        if len(members) >= 2:
            st["route3_candidates"] += 1
            rows.append({"decision": "candidate", "route": "3_model_token_no_brand", "key": n,
                         "reason": "model-like token without brand evidence: not merged",
                         "n_pages": len(members), "domains": " | ".join(members[:20]),
                         "names": " | ".join(pages[d]["name"] for d in members[:5]), "brands": ""})

    for d in pages:
        node_of.setdefault(d, f"product:src:{d}")
    return node_of, pages, rows, st


# --------------------------------------------------------------------- graph
def add_node(G, nid, ntype, name, **attrs):
    if nid not in G:
        G.add_node(nid, type=ntype, name=name, **attrs)


def build(recs, wdc, domains):
    recs = {d: r for d, r in recs.items() if d in domains}
    node_of, pages, rows, st = resolve_products(recs, wdc)
    G = nx.MultiDiGraph()
    agg = defaultdict(lambda: {"source_count": 0, "domains": set(), "sample_urls": []})

    def edge(u, v, rel, d, url, **extra):
        a = agg[(u, v, rel)]
        a["source_count"] += 1
        a["domains"].add(d)
        if url and len(a["sample_urls"]) < SAMPLE_URL_CAP:
            a["sample_urls"].append(url)
        a.update(extra)

    # product nodes
    for d, nid in node_of.items():
        p = pages[d]
        add_node(G, nid, "product", p["name"], identity="canonical" if ":canon:" in nid else "source")
        G.nodes[nid].setdefault("source_domains", []).append(d)
        if p["cluster"] is not None:
            G.nodes[nid].setdefault("wdc_cluster_ids", set()).add(p["cluster"])
        if p["wdc_brand"]:
            G.nodes[nid].setdefault("wdc_brands", set()).add(p["wdc_brand"])

    norm_index = defaultdict(set)
    for d, nid in node_of.items():
        norm_index[pages[d]["norm"]].add(nid)

    unresolved = Counter()
    for d, r in recs.items():
        url = (wdc.get(d) or {}).get("url") or r.get("url", "")
        pid = node_of.get(d)
        cname = company_name(r)
        cid = f"company:{cname.lower()}" if cname else None
        if cid:
            add_node(G, cid, "company", cname)
        if not pid:
            st["pages_without_product"] += 1
            continue
        # offers (page-context-derived)
        if cid:
            edge(cid, pid, "offers", d, url, derived="page_context")
        # manufactures (manufacturer evidence)
        if str(r.get("manufactures", "")).strip().lower() == "yes" and cid:
            edge(cid, pid, "manufactures", d, url)
        # has_brand
        b = brand_name(r)
        if b:
            bid = f"company:{b.lower()}"
            add_node(G, bid, "company", b)
            wb = pages[d]["wdc_brand"]
            edge(pid, bid, "has_brand", d, url,
                 wdc_brand_agrees=None if not wb else brands_compatible(brand_key(b), wb))
        # made_in
        loc, kind = resolve_location(r.get("made_in"))
        if loc:
            lid = f"location:{loc.lower()}"
            add_node(G, lid, "location", loc, kind=kind)
            edge(pid, lid, "made_in", d, url)
            st[f"made_in_{kind}"] += 1
        elif kind == "unresolved":
            raw = val(r.get("made_in_raw") or r.get("made_in"))
            G.nodes[pid].setdefault("made_in_unresolved", []).append(raw)
            unresolved[raw] += 1
            st["made_in_unresolved"] += 1
        # is_component_of
        parent = val(r.get("is_component_of"))
        if parent:
            pn = norm_name(parent)
            if is_generic(pn):
                st["parent_generic_skipped"] += 1
            elif pn == pages[d]["norm"]:
                st["parent_self_skipped"] += 1
            else:
                if is_specific_parent(pn):
                    hits = norm_index.get(pn, set()) - {pid}
                    if len(hits) == 1:
                        par = next(iter(hits))
                        st["parent_linked_to_product"] += 1
                    else:
                        par = f"product:parent:{pn}"
                        st["parent_shared_specific"] += 1
                        if len(hits) > 1:
                            st["parent_ambiguous"] += 1
                            rows.append({"decision": "candidate", "route": "parent_ambiguous",
                                         "key": pn, "reason": "several product nodes share this "
                                         "parent name; not linked", "n_pages": len(hits),
                                         "domains": d, "names": parent, "brands": ""})
                    add_node(G, par, "product", parent, identity="parent")
                else:
                    par = f"product:parentsrc:{d}:{pn}"
                    add_node(G, par, "product", parent, identity="parent_family_or_unspecific")
                    st["parent_page_scoped"] += 1
                edge(pid, par, "is_component_of", d, url)

    for (u, v, rel), a in agg.items():
        G.add_edge(u, v, key=rel, relation=rel, source_count=a["source_count"],
                   domains=sorted(a["domains"]), sample_urls=a["sample_urls"],
                   **{k: x for k, x in a.items() if k not in ("source_count", "domains", "sample_urls")})
    st["unresolved_top"] = "; ".join(f"{k} ({c})" for k, c in unresolved.most_common(10))
    return G, rows, st


# --------------------------------------------------------------------- CQs
def edges_of(G, rel):
    return [(u, v, d) for u, v, d in G.edges(data=True) if d["relation"] == rel]


def cq8_pairs(G):
    """Company pairs linked to the same product node (manufactures / offers /
    has_brand) where NO single page supports both links. Returns a list of
    ((company1, relation1, domains1), (company2, relation2, domains2), product)."""
    prod_comp = defaultdict(set)
    for u, v, d in G.edges(data=True):
        r = d["relation"]
        if r in ("manufactures", "offers"):
            prod_comp[v].add((u, r, frozenset(d["domains"])))
        elif r == "has_brand":
            prod_comp[u].add((v, r, frozenset(d["domains"])))
    seen, out = set(), []
    for p, cs in prod_comp.items():
        by_company = defaultdict(set)
        rels = defaultdict(set)
        for c, r, ds in cs:
            by_company[c] |= ds
            rels[c].add(r)
        comps = sorted(by_company)
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                c1, c2 = comps[i], comps[j]
                if by_company[c1].isdisjoint(by_company[c2]) and (c1, c2, p) not in seen:
                    seen.add((c1, c2, p))
                    out.append(((c1, "/".join(sorted(rels[c1])), sorted(by_company[c1])),
                                (c2, "/".join(sorted(rels[c2])), sorted(by_company[c2])), p))
    return out


def run_cqs(G):
    out, res = [], {}
    nm = lambda n: G.nodes[n].get("name", n)
    manu = edges_of(G, "manufactures")
    deg = Counter(u for u, _, _ in manu)
    out.append("CQ1  For a given company, which products does it manufacture?")
    for c, k in deg.most_common(3):
        prods = [nm(v) for u, v, _ in manu if u == c][:5]
        out.append(f"  {nm(c)} ({k}): {prods}")
    out.append("\nCQ2  For a given product, who manufactures it and where is it made?")
    made = defaultdict(set)
    for u, v, _ in edges_of(G, "made_in"):
        made[u].add(nm(v))
    shown = 0
    for u, v, _ in manu:
        if v in made and shown < 3:
            out.append(f"  {nm(v)}: manufacturer {nm(u)}; made_in {sorted(made[v])}")
            shown += 1
    rel = Counter(d["relation"] for _, _, d in G.edges(data=True))
    res["cq3"] = dict(rel)
    out.append(f"\nCQ3  Relation distribution: {dict(rel)}")
    out.append("\nCQ4  Companies with the most manufactures edges (plausibility check):")
    for c, k in deg.most_common(10):
        out.append(f"  {nm(c)}: {k}")
    multi = {p: locs for p, locs in made.items() if len(locs) > 1}
    res["cq5_multi_origin_products"] = len(multi)
    out.append(f"\nCQ5  Products with MULTIPLE ORIGIN CLAIMS (not necessarily contradictions): {len(multi)}")
    for p, locs in list(multi.items())[:5]:
        out.append(f"  {nm(p)}: {sorted(locs)} (sources: {len(G.nodes[p].get('source_domains', []))})")
    comp = edges_of(G, "is_component_of")
    manu_in = defaultdict(list)
    for u, v, _ in manu:
        manu_in[v].append(u)
    brand_out = defaultdict(list)
    for u, v, _ in edges_of(G, "has_brand"):
        brand_out[u].append(v)
    cq6 = [(c, p, m) for c, p, _ in comp for m in manu_in.get(p, [])]
    cq7 = [(c, p, b) for c, p, _ in comp for b in brand_out.get(p, [])]
    res["cq6_chains"], res["cq7_chains"] = len(cq6), len(cq7)
    out.append(f"\nCQ6  component -> is_component_of -> parent <- manufactures <- company: {len(cq6)} chains")
    for c, p, m in cq6[:5]:
        out.append(f"  {nm(c)} -> {nm(p)} <- {nm(m)}")
    out.append(f"\nCQ7  component -> is_component_of -> parent -> has_brand -> brand: {len(cq7)} chains")
    for c, p, b in cq7[:5]:
        out.append(f"  {nm(c)} -> {nm(p)} -> {nm(b)}")
    pairs = cq8_pairs(G)
    examples = [f"  via {nm(p)} [{G.nodes[p].get('identity')}]: {nm(c1)} ({r1}) <-> {nm(c2)} ({r2})"
                for (c1, r1, d1), (c2, r2, d2), p in pairs[:10]]
    res["cq8_candidate_pairs"] = len(pairs)
    out.append(f"\nCQ8  Company pairs linked only through a shared product node from different "
               f"pages: {len(pairs)} CANDIDATE pairs (call them 'hidden' only after the audit)")
    out += examples
    return "\n".join(out), res


# --------------------------------------------------------------------- main
def main():
    want = sys.argv[1:] or list(CONFIGS)
    os.makedirs(OUT, exist_ok=True)
    wdc = load_wdc()
    recs = {}
    for c in CONFIGS:
        if not os.path.exists(CONFIGS[c]):
            print(f"[WARN] {c}: {CONFIGS[c]} not found -- normalize it first")
            continue
        recs[c] = load_records(CONFIGS[c])
        print(f"[LOAD] {c}: {len(recs[c]):,} pages")
    common = set.intersection(*(set(r) for r in recs.values())) if len(recs) == 4 else None
    if common is None:
        print("[WARN] not all 4 configurations available: 'common' page set skipped")
    else:
        print(f"[INFO] common page set (records in all 4 configurations): {len(common):,}")

    summary = []
    for c in want:
        if c not in recs:
            continue
        for set_name, doms in (("full", set(recs[c])), ("common", common)):
            if doms is None:
                continue
            G, rows, st = build(recs[c], wdc, doms)
            cq_text, cq = run_cqs(G)
            tag = f"{c}_{set_name}"
            with open(f"{OUT}/kg_{tag}.gpickle", "wb") as f:
                pickle.dump(G, f)
            with open(f"{OUT}/entity_matches_{tag}.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["decision", "route", "key", "reason", "n_pages",
                                                  "domains", "names", "brands"])
                w.writeheader()
                w.writerows(rows)
            with open(f"{OUT}/cq_{tag}.txt", "w", encoding="utf-8") as f:
                f.write(cq_text + "\n")
            types = Counter(d["type"] for _, d in G.nodes(data=True))
            ident = Counter(d.get("identity") for _, d in G.nodes(data=True) if d["type"] == "product")
            row = {"config": c, "page_set": set_name, "pages": len(doms),
                   "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                   **{f"nodes_{k}": v for k, v in types.items()},
                   **{f"products_{k}": v for k, v in ident.items()},
                   **{f"edges_{k}": v for k, v in cq["cq3"].items()},
                   **{k: v for k, v in st.items()},
                   **{k: v for k, v in cq.items() if k != "cq3"}}
            summary.append(row)
            print(f"\n=== {tag}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
            print(f"  edges by relation: {cq['cq3']}")
            print(f"  products: {dict(ident)}")
            print(f"  merges: route1 {st['route1_clusters_accepted']} clusters / "
                  f"{st['route1_pages_merged']} pages (rejected {st['route1_clusters_rejected']}: "
                  f"brand conflict {st['route1_rejected_brand_conflict']}, unspecific names "
                  f"{st['route1_rejected_unspecific']}, member conflict "
                  f"{st['route1_rejected_member_conflict']}; members withheld as unspecific "
                  f"{st['route1_members_withheld_unspecific']}); "
                  f"route2 {st['route2_groups_accepted']} groups / {st['route2_pages_merged']} pages; "
                  f"route3 candidates {st['route3_candidates']}")
            print(f"  made_in: country {st['made_in_country']}, subdivision {st['made_in_subdivision']}, "
                  f"region {st['made_in_region']}, unresolved {st['made_in_unresolved']}")
            print(f"  parents: linked {st['parent_linked_to_product']}, shared specific "
                  f"{st['parent_shared_specific']}, page-scoped {st['parent_page_scoped']}, "
                  f"generic skipped {st['parent_generic_skipped']}")
            print(f"  CQ5 multi-origin products {cq['cq5_multi_origin_products']} | CQ6 {cq['cq6_chains']} "
                  f"| CQ7 {cq['cq7_chains']} | CQ8 candidate pairs {cq['cq8_candidate_pairs']}")

    keys = sorted({k for r in summary for k in r}, key=lambda k: (k not in ("config", "page_set"), k))
    with open(f"{OUT}/summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    print(f"\n[DONE] {OUT}/summary.csv, kg_*.gpickle, entity_matches_*.csv, cq_*.txt")


if __name__ == "__main__":
    main()
