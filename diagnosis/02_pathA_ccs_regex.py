# path_a_ccs_from_train_percentile.py
# - QUEST Path A: Document-centric Entity Ambiguity (CCS)
# - Input: confused_train_queries.bottom_XXpct.cos.jsonl from diagnosis_train_percentile.py
# - Output: ranked distractor entities with CCS scores + debug stats
#
# Run:
#   sr 1 48 python path_a_ccs_from_train_percentile.py --pct  \
#     --dia_out_dir "/.../ver2_percentile_train" \
#     --train_pt "/.../Material_Bio_merged_dataset_final.pt" \
#     --out_dir "/.../ver2_percentile_train/pathA_ccs_train" \
#     --top_neg_n 30
# sr 1 48 python 02.pathA_doc_entity_ccs.py --pct 30 --top_neg_n 30 --INPUT_DEFAULT_OUT_DIR "" ####################
# Notes:
# - No re-encoding.
# - Uses topk_scores_cos from diagnosis output as attention proxy (fast).
# - Doc text is reconstructed from TRAIN_PT in the SAME filtering/order rule.
#   If you have keep_indices saved in your train-compatible cache meta, pass it via --keep_indices.

import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Set, Any, Optional

import torch
# path_a_ccs_from_train_percentile.py
# - QUEST Path A: Document-centric Entity Ambiguity (CCS)
# - Input: confused_train_queries.bottom_XXpct.cos.jsonl from diagnosis_train_percentile.py
# - Output: ranked distractor entities with CCS scores + debug stats

import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Set, Any, Optional

import torch

# =========================
# Config defaults
# =========================
DEFAULT_TOP_NEG_N = 10
EPS_POS = 0.5
MIN_ENTITY_LEN = 2
MAX_ENTITIES_PER_DOC = 200

# =========================
# Default Paths (SAFE)
# =========================
DEFAULT_TRAIN_PT = "/gallery_millet/yerim.oh/MatRetriever/01.train/01-02.Detector/Ver2_train_final.pt"

def rank_weight(rank0: int) -> float:
    # 1/log2(rank+2), rank starts at 0
    return 1.0 / math.log2(rank0 + 2.0)

# =========================
# Field helpers
# =========================
def pick_first(d: Dict, keys: List[str]) -> str:
    for k in keys:
        v = d.get(k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

def build_doc_text_from_item(x: Dict) -> str:
    # doc text = title + abstract (best) else one side
    title = pick_first(x, ["paper_Title", "title", "query", "question"])
    abst  = pick_first(x, ["paper_abstract", "abstract", "passage", "positive", "text"])
    if title and abst:
        return f"{title}\n{abst}"
    return title or abst

def load_train_docs_in_order(train_pt: str, keep_indices_path: Optional[str] = None) -> List[str]:
    """
    Reconstruct doc texts aligned to train doc index.
    - If keep_indices_path is given: use those row indices (strong alignment)
    - Else: keep rows where both query and doc exist (fallback alignment)
    """
    data = torch.load(train_pt, map_location="cpu")

    if keep_indices_path and os.path.exists(keep_indices_path):
        keep = torch.load(keep_indices_path, map_location="cpu")
        if isinstance(keep, torch.Tensor):
            keep = keep.tolist()
        print(f"[KeepIndices] loaded: {keep_indices_path} (n={len(keep)})")

        docs = []
        for idx in keep:
            x = data[idx]
            t = build_doc_text_from_item(x)
            docs.append(t if t else "")
        return docs

    # fallback filter
    docs = []
    kept = 0
    for x in data:
        q = pick_first(x, ["paper_Title", "query", "question"])
        d = pick_first(x, ["paper_abstract", "passage", "positive", "abstract", "text"])
        if q and d:
            t = build_doc_text_from_item(x)
            docs.append(t if t else "")
            kept += 1

    print(f"[Docs] fallback filter kept rows (q&d present): {kept}")
    return docs

# =========================
# Entity extraction (heuristic)
# =========================
ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y",
    "Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce",
    "Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir",
    "Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu","Am","Cm",
    "Bk","Cf","Es","Fm","Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc",
    "Lv","Ts","Og"
}

STOPWORDS = {
    "the","a","an","and","or","of","in","on","for","with","to","from","by","as","at","is","are",
    "was","were","be","been","this","that","these","those","we","our","their","they","it","its",
    "study","method","results","data","analysis","using","used","based","paper","model"
}

RE_FORMULA = re.compile(r"\b(?:[A-Z][a-z]?\d*(?:[\.\-]\d+)?){2,}\b")  # Al2O3, CH3NH3PbI3
RE_ALLCAPS = re.compile(r"\b[A-Z]{2,}\b")                             # XRD, TEM, PCE
RE_HYPHEN  = re.compile(r"\b[A-Za-z]{2,}(?:-[A-Za-z]{2,})+\b")        # lead-free
RE_PAREN   = re.compile(r"\b[A-Za-z]{2,}\([A-Za-z0-9\-\+]+\)\b")      # perovskite(ABX3)

def clean_entity(e: str) -> str:
    e = e.strip()
    e = e.strip(".,;:()[]{}<>\"'`")
    return e

def extract_entities(text: str) -> Set[str]:
    if not text:
        return set()

    ents: Set[str] = set()

    for m in RE_FORMULA.finditer(text):
        e = clean_entity(m.group(0))
        if len(e) >= MIN_ENTITY_LEN and e.lower() not in STOPWORDS:
            ents.add(e)

    for m in RE_HYPHEN.finditer(text):
        e = clean_entity(m.group(0))
        if len(e) >= MIN_ENTITY_LEN and e.lower() not in STOPWORDS:
            ents.add(e)

    for m in RE_PAREN.finditer(text):
        e = clean_entity(m.group(0))
        if len(e) >= MIN_ENTITY_LEN and e.lower() not in STOPWORDS:
            ents.add(e)

    for m in RE_ALLCAPS.finditer(text):
        e = clean_entity(m.group(0))
        if len(e) >= 2 and e.lower() not in STOPWORDS:
            ents.add(e)

    toks = re.findall(r"[A-Za-z]{1,3}|\d+|[A-Za-z]{4,}", text)
    for t in toks:
        if t in ELEMENTS:
            ents.add(t)

    if len(ents) > MAX_ENTITIES_PER_DOC:
        ents = set(list(ents)[:MAX_ENTITIES_PER_DOC])
    return ents

# =========================
# IO helpers
# =========================
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

# =========================
# Main Path A logic
# =========================
def main(
    pct: int,
    dia_out_dir: str,
    train_pt: str,
    out_dir: Optional[str],
    top_neg_n: int,
    keep_indices_path: Optional[str] = None,
):
    # ---- default out_dir if not given
    if not out_dir or out_dir.strip() == "":
        out_dir = os.path.join(dia_out_dir, "pathA_ccs_train")

    # -------------------------
    # Load confused set jsonl
    # -------------------------
    conf_path = os.path.join(dia_out_dir, f"confused_train_queries.bottom_{pct:02d}pct.cos.jsonl")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"missing: {conf_path}")

    conf = read_jsonl(conf_path)
    print(f"[Confused] pct={pct} -> {len(conf)} queries")
    print(f"[Input] {conf_path}")

    # -------------------------
    # Reconstruct TRAIN doc texts aligned to doc indices
    # -------------------------
    print("[LoadDocs] reconstructing TRAIN doc texts...")
    docs = load_train_docs_in_order(train_pt, keep_indices_path=keep_indices_path)
    Nd = len(docs)
    print(f"[Docs] Nd(train docs)={Nd}")

    # sanity: max doc index used
    max_used = -1
    for rec in conf:
        max_used = max(max_used, int(rec.get("pos_doc_index", -1)))
        for didx in rec.get("topk_doc_indices_cos", []):
            max_used = max(max_used, int(didx))
    if max_used >= Nd:
        print(f"[WARN] max doc index in confused set = {max_used}, but Nd={Nd}")
        print("       -> doc indices may not match reconstructed TRAIN docs.")
        print("       -> strongly recommend using keep_indices for exact alignment.")

    # -------------------------
    # Aggregate document frequencies
    # -------------------------
    freq_neg = Counter()
    freq_pos = Counter()
    examples = defaultdict(list)
    per_query_neg_ents: Dict[int, set] = defaultdict(set)  # qi -> entities in D-(q)

    for rec in conf:
        qi = int(rec.get("i", -1))
        pos_idx = int(rec.get("pos_doc_index", -1))

        topk_idx = rec.get("topk_doc_indices_cos", [])

        # POS: document frequency in D+
        pos_text = docs[pos_idx] if 0 <= pos_idx < Nd else ""
        for e in extract_entities(pos_text):
            freq_pos[e] += 1

        # NEG: pick top N excluding pos → D-(q)
        picked = []
        for rank0, didx in enumerate(topk_idx):
            didx = int(didx)
            if didx == pos_idx:
                continue
            picked.append((rank0, didx))
            if len(picked) >= top_neg_n:
                break

        for rank0, didx in picked:
            dtext = docs[didx] if 0 <= didx < Nd else ""
            ents = extract_entities(dtext)
            if not ents:
                continue

            for e in ents:
                freq_neg[e] += 1
                per_query_neg_ents[qi].add(e)
                if len(examples[e]) < 5:
                    examples[e].append({
                        "q_i": qi,
                        "doc_idx": didx,
                        "rank0": rank0,
                    })

    # -------------------------
    # Compute CCS (REPAIR paper Eq. 5)
    # CCS(e) = df(e; D-) / (df(e; D+) + ε)
    # -------------------------
    ccs_dict: Dict[str, float] = {}
    rows = []
    for e, fn in freq_neg.items():
        fp = freq_pos.get(e, 0)
        ccs = fn / (fp + EPS_POS)
        ccs_dict[e] = ccs
        rows.append({
            "entity": e,
            "CCS": float(ccs),
            "freq_neg": int(fn),
            "freq_pos": int(fp),
            "examples": examples.get(e, []),
        })

    rows.sort(key=lambda x: x["CCS"], reverse=True)

    # -------------------------
    # Cintra: per-query highest-CCS concept from D-(q)
    # (REPAIR §3.3: "we construct Cintra by selecting the highest-CCS concept per query")
    # -------------------------
    cintra = []
    for rec in conf:
        qi = int(rec.get("i", -1))
        neg_ents = per_query_neg_ents.get(qi, set())
        if not neg_ents:
            continue
        best = max(neg_ents, key=lambda e: ccs_dict.get(e, 0.0))
        cintra.append({
            "query_i": qi,
            "query": rec.get("query", ""),
            "concept": best,
            "CCS": float(ccs_dict.get(best, 0.0)),
        })

    # -------------------------
    # Save outputs
    # -------------------------
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, f"pathA.entities.rank.train.bottom_{pct:02d}pct.json")
    out_top  = os.path.join(out_dir, f"pathA.entities.top200.train.bottom_{pct:02d}pct.jsonl")
    out_cintra = os.path.join(out_dir, f"cintra.bottom_{pct:02d}pct.jsonl")
    out_per_query = os.path.join(out_dir, f"per_query_neg_concepts.bottom_{pct:02d}pct.json")

    payload = {
        "pct": pct,
        "confused_queries": len(conf),
        "top_neg_n": top_neg_n,
        "eps_pos": EPS_POS,
        "num_train_docs": Nd,
        "input_confused_jsonl": conf_path,
        "train_pt": train_pt,
        "keep_indices": keep_indices_path,
        "out_dir": out_dir,
        "note": "CCS(e) = df(e; D-) / (df(e; D+) + eps)  [REPAIR paper Eq. 5, pure document frequency ratio]",
        "entities": rows,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_top, "w", encoding="utf-8") as f:
        for r in rows[:200]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save Cintra (one concept per confused query)
    with open(out_cintra, "w", encoding="utf-8") as f:
        for r in cintra:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save per-query neg concepts (used by PathB FINCH for Cinter selection)
    per_query_serializable = {str(qi): list(ents) for qi, ents in per_query_neg_ents.items()}
    with open(out_per_query, "w", encoding="utf-8") as f:
        json.dump(per_query_serializable, f, ensure_ascii=False)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_top}")
    print(f"[Saved] {out_cintra}  (Cintra: {len(cintra)} concepts, one per confused query)")
    print(f"[Saved] {out_per_query}  (per-query neg concepts for PathB)")
    print("[Done] Path A CCS extraction complete.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=5, help="bottom percentile (e.g., 5,10,15,...)")
    ap.add_argument("--dia_out_dir", type=str, required=True, help="diagnosis_train_percentile output dir")
    ap.add_argument("--train_pt", type=str, default=DEFAULT_TRAIN_PT, help="TRAIN pt")
    ap.add_argument("--out_dir", type=str, default="", help="output dir (default: {dia_out_dir}/pathA_ccs_train)")
    ap.add_argument("--top_neg_n", type=int, default=DEFAULT_TOP_NEG_N, help="use top-N negatives per query")
    ap.add_argument("--keep_indices", type=str, default=None, help="(recommended) keep_indices.pt path for exact alignment")
    args = ap.parse_args()

    main(
        pct=args.pct,
        dia_out_dir=args.dia_out_dir,
        train_pt=args.train_pt,
        out_dir=args.out_dir,
        top_neg_n=args.top_neg_n,
        keep_indices_path=args.keep_indices,
    )