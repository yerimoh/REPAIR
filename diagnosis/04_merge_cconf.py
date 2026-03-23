# final_terms_pre_api.py
# - NO API calls
# - NO query generation
# - Just collect "terms" (words/concepts) from existing PathA/PathB outputs
#
# Inputs:
#   - Path A (Cintra): cintra.bottom_XXpct.jsonl  — one concept per confused query (highest-CCS from D-(q))
#   - Path B (Cinter): pathB_finch_train.components.bottom_XXpct.json  — one concept per FINCH cluster
#
# Outputs:
#   - final_terms.bottom_XXpct.json
#   - final_terms.bottom_XXpct.txt
#   - prints counts: A terms / B terms / overlap / union
#
# Run:
#   python 03.all_concept.py --pct 30

import os
import json
import argparse
from collections import Counter
from typing import Dict, Any, List, Set, Tuple

# =========================
# Paths (EDIT if needed)
# =========================
#DIA_OUT_DIR = "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/ver2_percentile"
DIA_OUT_DIR = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2_ab/02_train_percentile_v1_ab"
PATHA_DIR = os.path.join(DIA_OUT_DIR, "pathA_ccs_train")
PATHB_DIR = os.path.join(DIA_OUT_DIR, "pathB_finch_train")

OUT_DIR = os.path.join(DIA_OUT_DIR, "PathAB_terms")
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# Defaults
# =========================
DEFAULT_PCT = 30
DEFAULT_TOP_A = 83006400          # Path A: take top-N entities by CCS
DEFAULT_TOP_COMP = 83006400       # Path B: take top-N components (sorted by avg_margin asc in your saved json)
DEFAULT_TOP_KW_PER_COMP = 50000  # Path B: top-k keywords per component

# =========================
# Utils
# =========================
def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm(s: str) -> str:
    return (s or "").strip()

def norm_key(s: str) -> str:
    # case-insensitive dedup key
    return norm(s).lower()

# =========================
# Extractors
# =========================
def extract_terms_pathA(pathA_cintra: str, top_n: int) -> List[Dict[str, Any]]:
    """
    Returns Cintra: list of dicts {"term": str, "source":"A", ...meta}
    Reads cintra.bottom_XXpct.jsonl — one concept per confused query (highest CCS from D-(q)).
    """
    out = []
    seen: Set[str] = set()
    with open(pathA_cintra, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= top_n:
                break
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            term = norm(r.get("concept", ""))
            if not term:
                continue
            out.append({
                "term": term,
                "source": "A",
                "CCS": float(r.get("CCS", 0.0)),
                "query_i": r.get("query_i", -1),
            })
    return out

def extract_terms_pathB(pathB_json: str, top_comp: int, top_kw: int) -> List[Dict[str, Any]]:
    """
    Returns Cinter: list of dicts {"term": str, "source":"B", ...meta}
    Reads pathB_finch_train.components.bottom_XXpct.json — one concept per FINCH cluster
    (most frequent concept among cluster members' candidate concepts).
    top_kw is unused (one concept per cluster); kept for API compatibility.
    """
    data = load_json(pathB_json)
    comps = data.get("components_sorted_by_avg_margin_asc", [])
    out = []
    for comp in comps[:top_comp]:
        comp_id = comp.get("component_id", None)
        avg_m = float(comp.get("avg_margin", 0.0))
        size = int(comp.get("size", 0))
        term = norm(comp.get("concept", ""))
        if not term:
            continue
        out.append({
            "term": term,
            "source": "B",
            "component_id": comp_id,
            "component_size": size,
            "component_avg_margin": avg_m,
            "concept_count": int(comp.get("concept_count", 0)),
        })
    return out

# =========================
# Main
# =========================
def main(pct: int, top_a: int, top_comp: int, top_kw: int):
    # PathA: Cintra — one concept per confused query (per-query highest-CCS)
    pathA_cintra = os.path.join(PATHA_DIR, f"cintra.bottom_{pct:02d}pct.jsonl")
    # PathB: Cinter — one concept per FINCH cluster (most frequent)
    pathB_json = os.path.join(PATHB_DIR, f"pathB_finch_train.components.bottom_{pct:02d}pct.json")

    if not os.path.exists(pathA_cintra):
        raise FileNotFoundError(f"missing Cintra (PathA): {pathA_cintra}")
    if not os.path.exists(pathB_json):
        raise FileNotFoundError(f"missing Cinter (PathB): {pathB_json}")

    a_terms = extract_terms_pathA(pathA_cintra, top_a)
    b_terms = extract_terms_pathB(pathB_json, top_comp, top_kw)

    # dedup within A/B (case-insensitive)
    a_seen, b_seen = set(), set()
    a_unique, b_unique = [], []

    for r in a_terms:
        k = norm_key(r["term"])
        if k in a_seen:
            continue
        a_seen.add(k)
        a_unique.append(r)

    for r in b_terms:
        k = norm_key(r["term"])
        if k in b_seen:
            continue
        b_seen.add(k)
        b_unique.append(r)

    Aset = set(a_seen)
    Bset = set(b_seen)
    overlap = Aset & Bset
    union = Aset | Bset

    # Build final list with provenance
    final = []
    # map key -> best representative casing + meta
    # prefer PathA casing if exists, else PathB
    a_map = {norm_key(r["term"]): r for r in a_unique}
    b_map = {norm_key(r["term"]): r for r in b_unique}

    for k in sorted(union):
        if k in a_map and k in b_map:
            term = a_map[k]["term"]  # prefer A casing
            final.append({
                "term": term,
                "from": ["A", "B"],
                "A": a_map[k],
                "B": b_map[k],
            })
        elif k in a_map:
            term = a_map[k]["term"]
            final.append({
                "term": term,
                "from": ["A"],
                "A": a_map[k],
            })
        else:
            term = b_map[k]["term"]
            final.append({
                "term": term,
                "from": ["B"],
                "B": b_map[k],
            })

    # Save
    out_json = os.path.join(OUT_DIR, f"final_terms.bottom_{pct:02d}pct.json")
    out_txt  = os.path.join(OUT_DIR, f"final_terms.bottom_{pct:02d}pct.txt")

    meta = {
        "pct": pct,
        "inputs": {"pathA_cintra": pathA_cintra, "pathB_json": pathB_json},
        "params": {"top_a": top_a, "top_comp": top_comp, "top_kw_per_comp": top_kw},
        "counts": {
            "A_raw": len(a_terms),
            "B_raw": len(b_terms),
            "A_unique": len(Aset),
            "B_unique": len(Bset),
            "overlap": len(overlap),
            "union_total": len(union),
        },
        "notes": [
            "Cconf = Cintra ∪ Cinter  [REPAIR paper §3.3]",
            "Cintra (PathA): per-query highest-CCS concept from D-(q)",
            "Cinter (PathB): most frequent concept per FINCH cluster",
            "Terms are deduplicated case-insensitively.",
        ],
        "terms": final,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(out_txt, "w", encoding="utf-8") as f:
        # write just terms (representative casing)
        for r in final:
            f.write(r["term"] + "\n")

# ==========================================
    # FORMAT THE SUMMARY TEXT
    # ==========================================
    summary_text = f"""
================ Final Terms (PRE-API) ================
pct                 : bottom {pct}%
-------------------------------------------------------
Path A raw terms     : {len(a_terms)}
Path A unique terms  : {len(Aset)}
Path B raw terms     : {len(b_terms)}
Path B unique terms  : {len(Bset)}
-------------------------------------------------------
Overlap (A ∩ B)      : {len(overlap)}
Union   (A ∪ B)      : {len(union)}
-------------------------------------------------------
[Saved] {out_json}
[Saved] {out_txt}
======================================================="""

    # Print to console
    print(summary_text)

    # Append to concept_info.txt
    info_file_path = os.path.join(OUT_DIR, "concept_info.txt")
    with open(info_file_path, "a", encoding="utf-8") as f:
        f.write(summary_text + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=DEFAULT_PCT)
    ap.add_argument("--top_a", type=int, default=DEFAULT_TOP_A, help="Top-N PathA entities by CCS")
    ap.add_argument("--top_comp", type=int, default=DEFAULT_TOP_COMP, help="Top-N PathB components (sorted)")
    ap.add_argument("--top_kw", type=int, default=DEFAULT_TOP_KW_PER_COMP, help="Top-k keywords per component")
    args = ap.parse_args()

    main(args.pct, args.top_a, args.top_comp, args.top_kw)
