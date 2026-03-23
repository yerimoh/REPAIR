#!/usr/bin/env bash
# run_repair_diagnosis.sh
# ============================================================
# REPAIR Full Diagnosis Pipeline (one iteration)
#   Stage I  — Diagnosis  : Steps 1-4
#   Stage II — Expansion  : Steps 5-6
#   Stage III— Differentiation: Steps 7-9
#
# Usage:
#   # 기본 실행 (PCT=30)
#   sr 1 48 --exclude=hockney --qos=q-high-yerim.oh bash run_repair_diagnosis.sh
#
#   # PCT 변경
#   PCT=20 sr 1 48 --exclude=hockney --qos=q-high-yerim.oh bash run_repair_diagnosis.sh
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# ★ 설정값 (필요 시 수정)
# ============================================================
PCT=${PCT:-30}                        # bottom percentile for confused query selection
TOP_NEG_N=30                          # PathA: top-N negatives per query for CCS
TOP_K=15                              # Differentiation: hard negatives per query
CONSISTENCY_K=5                       # Consistency filter threshold (κ=2 in paper; 5 for broader pool)
TARGET_K=20                           # Expansion: target papers per concept from API

export CUDA_VISIBLE_DEVICES=0

# Paths
DIA_OUT_DIR="/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1_ab"
TRAIN_PT="/gallery_millet/yerim.oh/MatRetriever/01.train/01-02.Detector/Ver2_train_final.pt"
SEMANTIC_KEY_FILE="/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/semantic_api.txt"

# Derived paths
CACHE_DIR="${DIA_OUT_DIR}/cache"
PATHA_OUT_DIR="${DIA_OUT_DIR}/pathA_ccs_train"
PATHB_OUT_DIR="${DIA_OUT_DIR}/pathB_finch_train"
PATHAB_TERMS_DIR="${DIA_OUT_DIR}/PathAB_terms"
EXPANSION_JSONL="${DIA_OUT_DIR}/06.semantic.bottom_${PCT}pct.jsonl"
EMBEDDING_CACHE_DIR="${DIA_OUT_DIR}/08.embeddings_cache"
HARD_NEG_JSONL="${DIA_OUT_DIR}/09.hard_negatives.bottom_${PCT}pct.jsonl"
FINAL_JSONL="${DIA_OUT_DIR}/10.final_train_data.bottom_${PCT}pct.jsonl"
CONCEPT_TXT="${PATHAB_TERMS_DIR}/final_terms.bottom_${PCT}pct.txt"

echo "========================================================"
echo "  REPAIR Diagnosis Pipeline"
echo "  PCT=${PCT}  TOP_NEG_N=${TOP_NEG_N}  TOP_K=${TOP_K}"
echo "  SCRIPT_DIR=${SCRIPT_DIR}"
echo "  DIA_OUT_DIR=${DIA_OUT_DIR}"
echo "========================================================"

# ============================================================
# Stage I / Step 1: Margin computation → confused query set
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage I / Step 1] Diagnosis: cosine-margin + bottom-${PCT}% confused queries"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# NOTE: 01.diagnosis_train_percentile.py has hardcoded paths.
#       BASE_MODEL, ADAPTER_DIR, TRAIN_PT, OUT_DIR are set inside the script.
python "${SCRIPT_DIR}/01.diagnosis_train_percentile.py"

# ============================================================
# Stage I / Step 2: PathA — CCS (Eq. 5) + Cintra
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage I / Step 2] PathA: CCS = df(e;D-) / (df(e;D+) + ε)  →  Cintra"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${SCRIPT_DIR}/02.pathA_doc_entity_ccs.py" \
    --pct          "${PCT}" \
    --dia_out_dir  "${DIA_OUT_DIR}" \
    --train_pt     "${TRAIN_PT}" \
    --out_dir      "${PATHA_OUT_DIR}" \
    --top_neg_n    "${TOP_NEG_N}"

# ============================================================
# Stage I / Step 3: PathB — FINCH clustering + Cinter
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage I / Step 3] PathB: FINCH 1-NN clustering  →  Cinter"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Step 1에서 생성된 normalized query cache를 자동으로 찾음
TRAIN_QUERY_CACHE=$(ls "${CACHE_DIR}"/cache.train_query_reps.normfp16.*.pt 2>/dev/null | head -1 || true)
if [ -z "${TRAIN_QUERY_CACHE}" ]; then
    echo "[ERROR] Normalized query cache not found in ${CACHE_DIR}"
    echo "        Step 1이 정상 완료되었는지 확인하세요."
    exit 1
fi
echo "  Query cache: ${TRAIN_QUERY_CACHE}"

python "${SCRIPT_DIR}/02.pathb.py" \
    --pct               "${PCT}" \
    --dia_out_dir       "${DIA_OUT_DIR}" \
    --out_dir           "${PATHB_OUT_DIR}" \
    --train_query_cache "${TRAIN_QUERY_CACHE}" \
    --patha_out_dir     "${PATHA_OUT_DIR}" \
    --device            cuda

# ============================================================
# Stage I / Step 4: Cconf = Cintra ∪ Cinter
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage I / Step 4] Cconf = Cintra ∪ Cinter"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${SCRIPT_DIR}/03.all_concept.py" --pct "${PCT}"

# ============================================================
# Stage II / Step 5: Confusing query generation (LLM)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage II / Step 5] Confusing query generation from Cconf"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
# NOTE: 05.p_query_make.py has hardcoded FILE_PATH.
#       final_terms.bottom_30pct.txt 경로를 스크립트 내부에서 수정하거나
#       PCT=30 그대로 사용하는 경우 그냥 실행 가능.
if [ ! -f "${CONCEPT_TXT}" ]; then
    echo "[ERROR] Concept txt not found: ${CONCEPT_TXT}"
    echo "        Step 4가 정상 완료되었는지 확인하세요."
    exit 1
fi
python "${SCRIPT_DIR}/05.p_query_make.py"

# ============================================================
# Stage II / Step 6: Expansion — Semantic Scholar API
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage II / Step 6] Expansion: Semantic Scholar API (target_k=${TARGET_K})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${SCRIPT_DIR}/06.sementic.py" \
    --concept_txt   "${CONCEPT_TXT}" \
    --key_file      "${SEMANTIC_KEY_FILE}" \
    --out_jsonl     "${EXPANSION_JSONL}" \
    --target_k      "${TARGET_K}" \
    --workers       9999 \
    --resume

# ============================================================
# Stage III / Step 7: Embed expansion candidates
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage III / Step 7] Embed expansion candidates"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
mkdir -p "${EMBEDDING_CACHE_DIR}"
python "${SCRIPT_DIR}/08.sort_negative.py" \
    --input_jsonl   "${EXPANSION_JSONL}" \
    --output_dir    "${EMBEDDING_CACHE_DIR}" \
    --batch_size    512 \
    --num_gpus      1

# ============================================================
# Stage III / Step 8: Hard negative selection
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage III / Step 8] Hard negative selection (top_k=${TOP_K})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${SCRIPT_DIR}/09.make_N.py" \
    --input_jsonl   "${EXPANSION_JSONL}" \
    --embedding_dir "${EMBEDDING_CACHE_DIR}" \
    --output_jsonl  "${HARD_NEG_JSONL}" \
    --top_k         "${TOP_K}" \
    --num_gpus      1

# ============================================================
# Stage III / Step 9: Consistency filter (Differentiation)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[Stage III / Step 9] Consistency filter + single hard negative (κ=${CONSISTENCY_K})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python "${SCRIPT_DIR}/12.BGE_filter.py" \
    --input_jsonl   "${HARD_NEG_JSONL}" \
    --embedding_dir "${EMBEDDING_CACHE_DIR}" \
    --output_jsonl  "${FINAL_JSONL}" \
    --top_k         "${TOP_K}" \
    --consistency_k "${CONSISTENCY_K}" \
    --batch_size    128

# ============================================================
echo ""
echo "========================================================"
echo "  REPAIR Pipeline Complete!"
echo "  최종 학습 데이터: ${FINAL_JSONL}"
echo "========================================================"
