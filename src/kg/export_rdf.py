"""
export_rdf.py -- export the knowledge graphs built by build_kg.py to RDF and answer
the competency questions CQ1-CQ8 with SPARQL. No LLM calls.

Run from the project root (after build_kg.py):
    python3 -m src.kg.export_rdf                     # all 4 configurations, full page set
    python3 -m src.kg.export_rdf qwen_single common  # one configuration / page set

RDF model (namespace sc: = https://w3id.org/sckg/ontology#, data under .../resource/)
  Classes     sc:Company, sc:Product, sc:Location
  Relations   sc:manufactures (Company->Product), sc:hasBrand (Product->Company),
              sc:madeIn (Product->Location), sc:isComponentOf (Product->Product),
              sc:offers (Company->Product; page-context-derived, sc:derivation "page_context")
  Labels      rdfs:label (surface name), sc:identity (source / canonical / parent ...)
  Provenance  every company-product link (manufactures, offers, hasBrand) is also
              stored as an sc:Assertion node per source page:
              sc:company, sc:product, sc:relation, sc:sourceDomain.
              CQ8 needs this: a pair counts only if NO single page supports both links.
              (Written only for products linked to >= 2 companies -- the only ones
              that can form a pair.)
              sc:linkedTo (Company->Product) summarises any of the three links.
  CQ8 = (pairs linked to the same product) - (pairs that share a supporting page).

Outputs (data/interim/kg/)
  kg_<config>_<set>.nt            N-Triples
  sparql/cq1.rq ... cq8.rq        the queries (for the thesis appendix)
  sparql_<config>_<set>.txt       SPARQL answers + check against the Python CQ counts
"""

import csv
import os
import pickle
import shutil
import sys
import time
from urllib.parse import quote

from pyoxigraph import RdfFormat, Store

D = "data/interim/kg"
SC = "https://w3id.org/sckg/ontology#"
RES = "https://w3id.org/sckg/resource/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
REL = {"manufactures": "manufactures", "has_brand": "hasBrand", "made_in": "madeIn",
       "is_component_of": "isComponentOf", "offers": "offers"}
CLS = {"company": "Company", "product": "Product", "location": "Location"}

PREFIX = f"PREFIX sc: <{SC}>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
QUERIES = {
    "cq1": PREFIX + """# CQ1: for the companies with the most manufactures links, which products?
SELECT ?company ?product WHERE {
  { SELECT ?c (COUNT(?p0) AS ?n) WHERE { ?c sc:manufactures ?p0 } GROUP BY ?c ORDER BY DESC(?n) LIMIT 3 }
  ?c sc:manufactures ?p . ?c rdfs:label ?company . ?p rdfs:label ?product .
} LIMIT 15""",
    "cq2": PREFIX + """# CQ2: for a product, who manufactures it and where is it made?
SELECT ?product ?manufacturer ?location WHERE {
  ?m sc:manufactures ?p . ?p sc:madeIn ?l .
  ?p rdfs:label ?product . ?m rdfs:label ?manufacturer . ?l rdfs:label ?location .
} LIMIT 5""",
    "cq3": PREFIX + """# CQ3: distribution of relation types
SELECT ?relation (COUNT(*) AS ?n) WHERE {
  ?s ?relation ?o .
  FILTER(?relation IN (sc:manufactures, sc:hasBrand, sc:madeIn, sc:isComponentOf, sc:offers))
} GROUP BY ?relation ORDER BY DESC(?n)""",
    "cq4": PREFIX + """# CQ4: companies with the most manufactures links (plausibility check)
SELECT ?company (COUNT(?p) AS ?n) WHERE { ?c sc:manufactures ?p . ?c rdfs:label ?company }
GROUP BY ?company ORDER BY DESC(?n) LIMIT 10""",
    "cq5": PREFIX + """# CQ5: products with multiple origin claims (not necessarily contradictions)
SELECT (COUNT(*) AS ?n) WHERE {
  { SELECT ?p WHERE { ?p sc:madeIn ?l } GROUP BY ?p HAVING (COUNT(DISTINCT ?l) > 1) }
}""",
    "cq6": PREFIX + """# CQ6: component -> isComponentOf -> parent <- manufactures <- company
SELECT (COUNT(*) AS ?n) WHERE { ?c sc:isComponentOf ?p . ?m sc:manufactures ?p }""",
    "cq7": PREFIX + """# CQ7: component -> isComponentOf -> parent -> hasBrand -> brand
SELECT (COUNT(*) AS ?n) WHERE { ?c sc:isComponentOf ?p . ?p sc:hasBrand ?b }""",
    # CQ8 = pairs linked to the same product (A) minus pairs that share a page (B).
    # Same definition as NOT EXISTS, but two plain joins (much faster).
    "cq8a": PREFIX + """# CQ8 part A: company pairs linked to the same product (any page)
SELECT (COUNT(*) AS ?n) WHERE {
  { SELECT DISTINCT ?c1 ?c2 ?p WHERE {
      ?c1 sc:linkedTo ?p . ?c2 sc:linkedTo ?p . FILTER(STR(?c1) < STR(?c2)) } }
}""",
    "cq8b": PREFIX + """# CQ8 part B: of those, pairs where one page supports both links
SELECT (COUNT(*) AS ?n) WHERE {
  { SELECT DISTINCT ?c1 ?c2 ?p WHERE {
      ?a1 sc:sourceDomain ?d ; sc:product ?p ; sc:company ?c1 .
      ?a2 sc:sourceDomain ?d ; sc:product ?p ; sc:company ?c2 .
      FILTER(STR(?c1) < STR(?c2)) } }
}""",
}


def iri(node_id):
    return f"<{RES}{quote(node_id, safe='')}>"


def lit(s):
    s = str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")
    return f'"{s}"'


def export(G, path):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        def t(s, p, o):
            nonlocal n
            f.write(f"{s} <{p}> {o} .\n")
            n += 1
        for nid, d in G.nodes(data=True):
            t(iri(nid), RDF_TYPE, f"<{SC}{CLS[d['type']]}>")
            if d.get("name"):
                t(iri(nid), LABEL, lit(d["name"]))
            if d.get("identity"):
                t(iri(nid), SC + "identity", lit(d["identity"]))
        k = 0
        linked = set()
        # provenance assertions are only needed for CQ8, i.e. for products linked to
        # >= 2 different companies; writing them for all products made the file huge
        comp_of = {}
        for u, v, d in G.edges(data=True):
            if d["relation"] in ("manufactures", "offers", "has_brand"):
                c, p = (v, u) if d["relation"] == "has_brand" else (u, v)
                comp_of.setdefault(p, set()).add(c)
        multi = {p for p, cs in comp_of.items() if len(cs) >= 2}
        for u, v, d in G.edges(data=True):
            rel = d["relation"]
            t(iri(u), SC + REL[rel], iri(v))
            if rel in ("manufactures", "offers", "has_brand"):
                company, product = (v, u) if rel == "has_brand" else (u, v)
                if (company, product) not in linked:
                    linked.add((company, product))
                    t(iri(company), SC + "linkedTo", iri(product))
                for dom in (d["domains"] if product in multi else []):
                    k += 1
                    a = f"<{RES}assertion/{k}>"
                    t(a, RDF_TYPE, f"<{SC}Assertion>")
                    t(a, SC + "company", iri(company))
                    t(a, SC + "product", iri(product))
                    t(a, SC + "relation", lit(rel))
                    t(a, SC + "sourceDomain", lit(dom))
    return n


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scalar(store, q):
    for row in store.query(q):
        return int(str(row[0].value))
    return 0


def main():
    args = sys.argv[1:]
    configs = [args[0]] if args else ["qwen_single", "qwen_iter", "llama_single", "llama_iter"]
    sets = [args[1]] if len(args) > 1 else ["full"]
    os.makedirs(f"{D}/sparql", exist_ok=True)
    for name, q in QUERIES.items():
        with open(f"{D}/sparql/{name}.rq", "w", encoding="utf-8") as f:
            f.write(q + "\n")
    py = {}
    if os.path.exists(f"{D}/summary.csv"):
        with open(f"{D}/summary.csv", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                py[(r["config"], r["page_set"])] = r

    for c in configs:
        for s in sets:
            tag = f"{c}_{s}"
            gp = f"{D}/kg_{tag}.gpickle"
            if not os.path.exists(gp):
                print(f"[WARN] {gp} missing -- run build_kg.py first")
                continue
            with open(gp, "rb") as f:
                G = pickle.load(f)
            nt = f"{D}/kg_{tag}.nt"
            log(f"{tag}: writing RDF")
            n = export(G, nt)
            log(f"{tag}: {n:,} triples, loading into the SPARQL store")
            db = f"{D}/oxigraph_{tag}"
            shutil.rmtree(db, ignore_errors=True)
            store = Store(db)  # on-disk store: low memory, fast bulk loading
            store.bulk_load(path=nt, format=RdfFormat.N_TRIPLES)
            log(f"{tag}: running CQ1-CQ4")
            out = [f"{tag}: {n:,} triples written to {nt}"]
            for name in ("cq1", "cq2", "cq3", "cq4"):
                out.append(f"\n{name.upper()}")
                for row in store.query(QUERIES[name]):
                    out.append("  " + " | ".join(str(x.value).rsplit("#", 1)[-1] for x in row))
            log(f"{tag}: running CQ5-CQ8")
            counts = {k: scalar(store, QUERIES[k]) for k in ("cq5", "cq6", "cq7")}
            a, b = scalar(store, QUERIES["cq8a"]), scalar(store, QUERIES["cq8b"])
            counts["cq8"] = a - b
            log(f"{tag}: CQ8 = {a:,} - {b:,}")
            ref = py.get((c, s), {})
            pymap = {"cq5": "cq5_multi_origin_products", "cq6": "cq6_chains",
                     "cq7": "cq7_chains", "cq8": "cq8_candidate_pairs"}
            out.append("\nCHECK against the Python CQ results (summary.csv):")
            ok_all = True
            for k, v in counts.items():
                r = ref.get(pymap[k])
                ok = r is not None and str(v) == str(r)
                ok_all &= ok
                out.append(f"  {k.upper()}: SPARQL {v:,} | Python {r} | {'OK' if ok else 'MISMATCH'}")
            text = "\n".join(out)
            with open(f"{D}/sparql_{tag}.txt", "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(text.split("\nCQ1")[0])
            print("\n".join(out[-5:]))
            print(f"[{'OK' if ok_all else 'CHECK'}] {tag}\n")
    print(f"[DONE] queries in {D}/sparql/, answers in {D}/sparql_<config>_<set>.txt")


if __name__ == "__main__":
    main()
