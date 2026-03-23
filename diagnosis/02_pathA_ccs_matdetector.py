# path_a_ccs_from_train_percentile.py
# - QUEST Path A: Document-centric Entity Ambiguity (CCS)
# - Input: confused_train_queries.bottom_XXpct.cos.jsonl from diagnosis_train_percentile.py
# - Output: ranked distractor entities with CCS scores + debug stats
#
# Run:
#   sr 1 48 python 02.pathA_MatDetector.py 
# sr 1 48 python 02.pathA_doc_entity_ccs.py --pct 30 --top_neg_n 30 ####################
# Notes:
# - No re-encoding.
# - Uses topk_scores_cos from diagnosis output as attention proxy (fast).
# - Doc text is reconstructed from TRAIN_PT in the SAME filtering/order rule.
#   If you have keep_indices saved in your train-compatible cache meta, pass it via --keep_indices.


'''
아, **Path A 로직(Confused Concept Selection, CCS)**을 적용해서 진짜 '모델이 헷갈려 하는 컨셉'들을 추려내야 하는군요! 제가 앞서 드린 코드는 단순 빈도수 기반이었지만, 지금 주신 로직은 **모델의 진단 결과(Diagnosis)**를 바탕으로 **CCS(Concept Confusion Score)**를 계산하는 훨씬 정교한 방식입니다.

이 로직의 핵심은 **"정답(Positive)에는 안 나오는데, 오답(Negative) 상위권에 자꾸 나타나서 모델을 방해하는 컨셉"**에 높은 점수를 주는 것입니다.

주신 메인 로직을 실행하기 위해 필요한 extract_entities, rank_weight, 그리고 데이터 로드 함수들을 포함하여 전체 흐름이 완결된 코드를 구성해 드립니다.
'''
import os
import json
import torch
import argparse
import re
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# =========================
# Configuration & Constants
# =========================
EPS_POS = 1e-6  # Zero-division 방지
DEFAULT_DIA_OUT_DIR = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1"
DEFAULT_TRAIN_PT = "Ver2_train_final_matdetector.pt" # MatDetector 결과가 포함된 파일
DEFAULT_OUT_DIR = "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/PathAB_termsMatDetector"
DEFAULT_TOP_NEG_N = 5

# =========================
# Helper Functions
# =========================

def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def rank_weight(rank0: int) -> float:
    """상위 랭크일수록 더 높은 가중치 (1/log2(rank+2))"""
    import math
    return 1.0 / math.log2(rank0 + 2)

def extract_entities(doc_entry: Dict) -> List[str]:
    """
    MatDetector가 미리 뽑아놓은 mat_concepts_all 사용
    단순 텍스트가 아니라 dict 형태의 entry를 받음
    """
    if not doc_entry:
        return []
    # 미리 MatDetector로 뽑아둔 컨셉 리스트 반환
    return doc_entry.get("mat_concepts_all", [])

def load_train_docs_in_order(train_pt: str, keep_indices_path: Optional[str] = None) -> List[Dict]:
    """TRAIN pt 로드 및 인덱스 정렬"""
    full_data = torch.load(train_pt, map_location="cpu")
    
    # keep_indices가 있다면 필터링 (학습 시 사용된 데이터와 매칭)
    if keep_indices_path and os.path.exists(keep_indices_path):
        keep_idx = torch.load(keep_indices_path, map_location="cpu")
        # 리스트라면 인덱스로, bool mask라면 해당 위치로 추출
        if isinstance(keep_idx, list):
            return [full_data[i] for i in keep_idx]
        else:
            return [d for i, d in enumerate(full_data) if keep_idx[i]]
            
    return full_data

# =========================
# Main Path A logic
# =========================
def main(
    pct: int,
    dia_out_dir: str,
    train_pt: str,
    out_dir: str,
    top_neg_n: int,
    keep_indices_path: Optional[str] = None,
):
    # 1. Diagnosis(헷갈리는 쿼리 세트) 로드
    conf_path = os.path.join(dia_out_dir, f"train_percentile_v1/{pct}/confused_train_queries.bottom_{pct:02d}pct.cos.jsonl")
    if not os.path.exists(conf_path):
        # 경로가 다를 경우를 대비한 fallback
        conf_path = os.path.join(dia_out_dir, f"confused_train_queries.bottom_{pct:02d}pct.cos.jsonl")
        if not os.path.exists(conf_path):
            raise FileNotFoundError(f"missing: {conf_path}")

    conf = read_jsonl(conf_path)
    print(f"[Confused] pct={pct} -> {len(conf)} queries")

    # 2. MatDetector 결과가 포함된 문서 데이터 로드
    print("[LoadDocs] Reconstructing docs with MatDetector concepts...")
    docs = load_train_docs_in_order(train_pt, keep_indices_path=keep_indices_path)
    Nd = len(docs)
    print(f"[Docs] Total docs loaded: {Nd}")

    # 3. Aggregation (document frequency counts)
    freq_neg = Counter()
    freq_pos = Counter()
    examples = defaultdict(list)
    per_query_neg_ents = defaultdict(set)  # qi -> entities in D-(q)

    for rec in tqdm(conf, desc="Processing Confusion"):
        qi = int(rec["i"])
        pos_idx = int(rec["pos_doc_index"])
        topk_idx = rec.get("topk_doc_indices_cos", [])

        # ---- POS entities (D+ document frequency)
        pos_doc = docs[pos_idx] if 0 <= pos_idx < Nd else None
        for e in extract_entities(pos_doc):
            freq_pos[e] += 1

        # ---- NEG entities: top-N retrieved docs excluding pos → D-(q)
        picked = []
        for rank0, didx in enumerate(topk_idx):
            didx = int(didx)
            if didx == pos_idx: continue
            picked.append((rank0, didx))
            if len(picked) >= top_neg_n: break

        for rank0, didx in picked:
            neg_doc = docs[didx] if 0 <= didx < Nd else None
            ents = extract_entities(neg_doc)
            if not ents: continue

            for e in ents:
                freq_neg[e] += 1
                per_query_neg_ents[qi].add(e)
                if len(examples[e]) < 5:
                    examples[e].append({"q_i": qi, "doc_idx": didx, "rank0": rank0})

    # 4. Compute CCS (REPAIR paper Eq. 5): CCS(e) = df(e; D-) / (df(e; D+) + ε)
    ccs_dict = {}
    rows = []
    for e, fn in freq_neg.items():
        fp = freq_pos.get(e, 0)
        ccs = fn / (fp + EPS_POS)
        ccs_dict[e] = ccs
        rows.append({
            "term": e,
            "CCS": float(ccs),
            "freq_neg": int(fn),
            "freq_pos": int(fp),
            "from": "PathA_Diagnosis"
        })

    rows.sort(key=lambda x: x["CCS"], reverse=True)

    # Cintra: for each q in Qconf, select highest-CCS concept from its D-(q)
    # (REPAIR §3.3: "we construct Cintra by selecting the highest-CCS concept per query")
    cintra = []
    for rec in conf:
        qi = int(rec["i"])
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

    # 5. Save outputs
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"final_terms.bottom_{pct:02d}pct.json")
    out_cintra = os.path.join(out_dir, f"cintra.bottom_{pct:02d}pct.jsonl")
    out_per_query = os.path.join(out_dir, f"per_query_neg_concepts.bottom_{pct:02d}pct.json")

    output_payload = {
        "meta": {
            "pct": pct,
            "top_neg_n": top_neg_n,
            "note": "CCS(e) = df(e; D-) / (df(e; D+) + eps)  [REPAIR paper Eq. 5]",
        },
        "terms": rows[:1000]
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    # Save Cintra (one concept per confused query)
    with open(out_cintra, "w", encoding="utf-8") as f:
        for r in cintra:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save per-query neg concepts (for PathB FINCH Cinter selection)
    per_query_serializable = {str(qi): list(ents) for qi, ents in per_query_neg_ents.items()}
    with open(out_per_query, "w", encoding="utf-8") as f:
        json.dump(per_query_serializable, f, ensure_ascii=False)

    print(f"[Done] Saved {len(rows)} terms to {out_json}")
    print(f"[Done] Cintra: {len(cintra)} concepts -> {out_cintra}")
    print(f"[Done] Per-query concepts -> {out_per_query}")

if __name__ == "__main__":
    # 실행 시 필요한 인자들은 환경에 맞춰 조정하세요
    main(pct=30, dia_out_dir=DEFAULT_DIA_OUT_DIR, train_pt=DEFAULT_TRAIN_PT, out_dir=DEFAULT_OUT_DIR, top_neg_n=5)