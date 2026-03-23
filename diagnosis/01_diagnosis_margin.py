# diagnosis_train_percentile.py
# ------------------------------------------------------------
# Goal
#   1) TRAIN_PT를 다시 읽어서 "TRAIN과 동일한 인코딩 규칙"으로
#      - train-compatible doc_cache (docs = train passages only)
#      - keep_indices (q+pos가 존재하는 row index)
#      - meta / offset_cache
#      를 새로 생성/저장
#
#   2) 위 캐시를 입력으로 받아 TRAIN query도 동일 규칙으로 인코딩(캐시)
#   3) cosine-margin = s(q, d_pos) - max_{d != d_pos} s(q, d) 계산
#   4) bottom percentile confusion set 저장
#
# Speed
#   - TopK search: FAISS(GPU FlatIP) 우선 사용
#   - FAISS 없으면: pinned-memory + GPU matmul (chunked) fallback
#
# Run:
#   sr 1 48 python 01.diagnosis_train_percentile.py
#
# ------------------------------------------------------------

import os
import json
import hashlib
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
from peft import PeftModel


# =========================
# Paths (USER)
# =========================
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
ADAPTER_DIR = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/checkpoints/BEG_BMR2_FULL/checkpoint-step-100"
TRAIN_PT = "/gallery_millet/yerim.oh/MatRetriever/01.train/01-02.Detector/Ver2_train_final.pt"
# Output root
OUT_DIR = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1_ab"
os.makedirs(OUT_DIR, exist_ok=True)

# Cache outputs (train-compatible)
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Diagnosis outputs
ALL_JSONL = os.path.join(OUT_DIR, "01.train_all_query_margins.cos_ab.jsonl")
META_JSON = os.path.join(OUT_DIR, "01.train_diagnosis_meta.cos_percentile_ab.json")
BOTTOM_PCTS = [5, 10, 15, 20, 25, 30, 3, 40, 45, 50,55, 60, 65, 70, 75, 80, 85, 90, 95,100]


# =========================
# TRAIN-style Templates
# =========================
TEMPLATES = {
    "passage": "Represent this passage.\nPassage: {text} {eos}",
    "query_mat_paper": "Given a query, retrieve passages that are relevant to the query.\nQuery: {text} {eos}",
    "query_default": "Given a query, retrieve passages that are relevant to the query.\nQuery: {text} {eos}",
}


# =========================
# Config
# =========================
@dataclass
class Cfg:
    device: str = "cuda"
    gpu_id: int = 0

    # encoding
    max_len: int = 512
    batch_size_doc: int = 1024
    batch_size_query: int = 1024
    use_bfloat16_backbone: bool = True   # encoder backbone dtype (forward)
    save_cache_dtype: str = "float16"    # cache save dtype: float16 recommended

    # search
    topk: int = 50
    doc_chunk: int = 20000
    query_batch: int = 2048
    normalize_chunk: int = 200000
    use_fp16_matmul: bool = True
    pin_memory: bool = True

    # faiss
    prefer_faiss: bool = True
    faiss_use_gpu: bool = True

    # recompute controls
    force_rebuild_doc_cache: bool = False
    force_rebuild_query_cache: bool = False
    overwrite_outputs: bool = False
    write_all_jsonl: bool = True

    # correctness guard
    strict_alignment: bool = True  # True면 keep_indices 기반으로 pos=i 정렬을 "보장"해야만 진행

CFG = Cfg()


# =========================
# Utilities: text processing
# =========================
def _process_text(text: str, template_key: str, eos_token: str) -> str:
    template = TEMPLATES.get(template_key, TEMPLATES["query_default"])
    return template.format(text=text, eos=eos_token)

def _force_eos_at_end(tokens: Dict[str, torch.Tensor], eos_id: int) -> Dict[str, torch.Tensor]:
    """
    TRAIN에서 쓰던 로직과 동일(길이 꽉차면 마지막 토큰을 eos로 교체,
    아니면 실제 끝(length)에 eos 삽입)
    """
    input_ids = tokens["input_ids"]
    attn = tokens["attention_mask"]
    B, L = input_ids.shape
    lengths = attn.sum(dim=1)  # (B,)

    full = lengths >= L
    if full.any():
        input_ids[full, L - 1] = eos_id
        attn[full, L - 1] = 1

    not_full = ~full
    if not_full.any():
        idx = lengths[not_full]
        input_ids[not_full, idx] = eos_id
        attn[not_full, idx] = 1

    tokens["input_ids"] = input_ids
    tokens["attention_mask"] = attn
    return tokens


# =========================
# Encoder (TRAIN-compatible)
#   - AutoModel + PeftModel
#   - EOS last-token pooling
# =========================
class RetrieverEncoder(nn.Module):
    def __init__(self, base_model: str, adapter_dir: str, device: str, use_bfloat16: bool = True):
        super().__init__()
        dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        backbone = AutoModel.from_pretrained(
            base_model,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
        backbone.config.use_cache = False
        backbone = PeftModel.from_pretrained(backbone, adapter_dir)
        self.backbone = backbone
        self.device = device

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state                          # [B,T,H]
        seq_len = attention_mask.sum(dim=1) - 1                      # [B]  eos 위치
        B = last_hidden.size(0)
        reps = last_hidden[torch.arange(B, device=last_hidden.device), seq_len]  # [B,H]
        return reps


@torch.no_grad()
def encode_texts_train_style(
    model: RetrieverEncoder,
    tokenizer: AutoTokenizer,
    texts: List[str],
    template_key: str,
    batch_size: int,
    max_length: int,
    device: str,
    desc: str,
) -> torch.Tensor:
    """
    TRAIN과 동일:
      - template 적용
      - padding_side='right'
      - EOS 강제 삽입
      - EOS last-token pooling
    반환: CPU float32 (안정)
    """
    model.eval()
    tokenizer.padding_side = "right"

    all_reps = []
    total = len(texts)
    num_batches = (total + batch_size - 1) // batch_size

    eos = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    for b in tqdm(range(num_batches), desc=desc):
        s = b * batch_size
        e = min(total, s + batch_size)
        batch = texts[s:e]

        formatted = [_process_text(t, template_key, eos) for t in batch]
        toks = tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        toks = _force_eos_at_end(toks, eos_id)

        reps = model(
            toks["input_ids"].to(device, non_blocking=True),
            toks["attention_mask"].to(device, non_blocking=True),
        )
        all_reps.append(reps.float().cpu())

        del toks, reps

    return torch.cat(all_reps, dim=0)


# =========================
# Data loading (train)
# =========================
def load_train_rows(pt_path: str) -> Tuple[List[str], List[str], List[int]]:
    """
    TRAIN_PT에서 (query, passage) pair를 로딩하되
    q+pos 모두 있는 row만 사용.
    returns:
      queries, passages, keep_indices (원본 pt index)
    """
    data = torch.load(pt_path, map_location="cpu")
    queries, passages, keep = [], [], []
    for i, x in enumerate(data):
        q = x.get("paper_Title") or x.get("query") or x.get("question")
        d = x.get("paper_abstract") or x.get("passage") or x.get("positive")
        if not q or not d:
            continue
        queries.append(q)
        passages.append(d)
        keep.append(i)
    return queries, passages, keep


# =========================
# Cache helpers
# =========================
def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def train_cache_key() -> str:
    payload = {
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR,
        "train_pt": TRAIN_PT,
        "templates": TEMPLATES,
        "max_len": CFG.max_len,
        "pooling": "eos_last_token",
        "padding_side": "right",
        "force_eos": True,
        "use_bfloat16_backbone": CFG.use_bfloat16_backbone,
    }
    return _md5(json.dumps(payload, ensure_ascii=False, sort_keys=True))

def cache_paths(key: str) -> Dict[str, str]:
    # doc cache (train-only)
    doc_cache = os.path.join(CACHE_DIR, f"cache.train_doc_reps.{key}.pt")
    # keep indices
    keep_cache = os.path.join(CACHE_DIR, f"cache.train_keep_indices.{key}.pt")
    # offset cache (train-only => offset = Ntrain)
    offset_cache = os.path.join(CACHE_DIR, f"cache.train_valid_pos_offset.{key}.pt")
    # meta
    meta_cache = os.path.join(CACHE_DIR, f"cache.train_doc_meta.{key}.json")

    # query cache (train)
    q_cache = os.path.join(CACHE_DIR, f"cache.train_query_reps.{key}.pt")
    q_meta  = os.path.join(CACHE_DIR, f"cache.train_query_meta.{key}.json")

    # normalized caches (train-only)
    d_norm = os.path.join(CACHE_DIR, f"cache.train_doc_reps.normfp16.{key}.pt")
    q_norm = os.path.join(CACHE_DIR, f"cache.train_query_reps.normfp16.{key}.pt")
    n_meta = os.path.join(CACHE_DIR, f"cache.train_norm_meta.{key}.json")

    return {
        "doc_cache": doc_cache,
        "keep_cache": keep_cache,
        "offset_cache": offset_cache,
        "meta_cache": meta_cache,
        "q_cache": q_cache,
        "q_meta": q_meta,
        "d_norm": d_norm,
        "q_norm": q_norm,
        "n_meta": n_meta,
    }

def save_tensor(path: str, x: torch.Tensor, dtype: str = "float16"):
    if dtype == "float16":
        torch.save(x.half().cpu(), path)
    elif dtype == "bfloat16":
        torch.save(x.to(torch.bfloat16).cpu(), path)
    else:
        torch.save(x.float().cpu(), path)

def load_fp16_cpu(path: str) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    if x.dtype != torch.float16:
        x = x.half()
    return x

def normalize_inplace_fp16_cpu(x_fp16_cpu: torch.Tensor, chunk: int, desc: str):
    assert x_fp16_cpu.device.type == "cpu"
    N = x_fp16_cpu.size(0)
    for s in tqdm(range(0, N, chunk), desc=desc):
        e = min(N, s + chunk)
        block = x_fp16_cpu[s:e].float()
        block = F.normalize(block, p=2, dim=1)
        x_fp16_cpu[s:e] = block.half()


# =========================
# TopK search (FAISS preferred)
# =========================
def try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None

@torch.no_grad()
def topk_search_faiss_ip(
    q_norm_fp16_cpu: torch.Tensor,
    d_norm_fp16_cpu: torch.Tensor,
    topk: int,
    use_gpu: bool,
    gpu_id: int,
):
    """
    q,d는 이미 L2 normalized라 cosine == inner product.
    FAISS는 float32 권장. (48GB GPU면 보통 OK)
    returns torch tensors on CPU:
      scores: [Nq, topk] float32
      idx:    [Nq, topk] int64
    """
    faiss = try_import_faiss()
    if faiss is None:
        raise RuntimeError("faiss not available")

    q = q_norm_fp16_cpu.float().numpy()
    d = d_norm_fp16_cpu.float().numpy()
    dim = d.shape[1]

    index = faiss.IndexFlatIP(dim)

    if use_gpu:
        # GPU resources
        res = faiss.StandardGpuResources()
        # (옵션) temp memory 조정 가능
        # res.setTempMemory(1024 * 1024 * 1024)  # 1GB
        index = faiss.index_cpu_to_gpu(res, gpu_id, index)

    index.add(d)
    scores, idx = index.search(q, topk)
    scores_t = torch.from_numpy(scores).float()
    idx_t = torch.from_numpy(idx).long()
    return scores_t, idx_t

@torch.no_grad()
def topk_search_torch_fast(
    q_normed_fp16_cpu: torch.Tensor,
    d_normed_fp16_cpu: torch.Tensor,
    topk: int,
    doc_chunk: int,
    query_batch: int,
    device: str,
    use_fp16_matmul: bool = True,
    pin_memory: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback: pinned CPU -> GPU non_blocking + chunk matmul + topk merge
    (chunk마다 synchronize() 없음)
    """
    assert q_normed_fp16_cpu.device.type == "cpu"
    assert d_normed_fp16_cpu.device.type == "cpu"

    if pin_memory:
        if not q_normed_fp16_cpu.is_pinned():
            q_normed_fp16_cpu = q_normed_fp16_cpu.pin_memory()
        if not d_normed_fp16_cpu.is_pinned():
            d_normed_fp16_cpu = d_normed_fp16_cpu.pin_memory()

    Nq, H = q_normed_fp16_cpu.shape
    Nd, _ = d_normed_fp16_cpu.shape

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")

    # TF32 (속도 ↑, 보통 retriever에서 큰 문제 없음)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    topk_scores_all = torch.empty((Nq, topk), dtype=torch.float32)
    topk_indices_all = torch.empty((Nq, topk), dtype=torch.long)

    num_q_batches = (Nq + query_batch - 1) // query_batch
    num_d_chunks = (Nd + doc_chunk - 1) // doc_chunk

    for qb in tqdm(range(num_q_batches), desc=f"TopK TORCH (q_batch={query_batch}, doc_chunk={doc_chunk})"):
        qs = qb * query_batch
        qe = min(Nq, qs + query_batch)

        q_batch = q_normed_fp16_cpu[qs:qe].to(dev, non_blocking=True) if use_cuda else q_normed_fp16_cpu[qs:qe]
        bsz = q_batch.size(0)

        topk_scores = torch.full((bsz, topk), -1e9, dtype=torch.float32, device=dev)
        topk_indices = torch.full((bsz, topk), -1, dtype=torch.long, device=dev)

        for dc in range(num_d_chunks):
            ds = dc * doc_chunk
            de = min(Nd, ds + doc_chunk)

            d_chunk = d_normed_fp16_cpu[ds:de].to(dev, non_blocking=True) if use_cuda else d_normed_fp16_cpu[ds:de]

            if use_cuda and use_fp16_matmul:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    scores = q_batch @ d_chunk.T
                scores = scores.float()
            else:
                scores = q_batch.float() @ d_chunk.float().T

            merged_scores = torch.cat([topk_scores, scores], dim=1)
            new_idx = torch.arange(ds, de, device=dev).unsqueeze(0).expand(bsz, -1)
            merged_indices = torch.cat([topk_indices, new_idx], dim=1)

            vals, pos = torch.topk(merged_scores, k=topk, dim=1)
            idx = torch.gather(merged_indices, 1, pos)
            topk_scores, topk_indices = vals, idx

            del d_chunk, scores, merged_scores, merged_indices, vals, pos, idx

        topk_scores_all[qs:qe] = topk_scores.detach().cpu()
        topk_indices_all[qs:qe] = topk_indices.detach().cpu()

        del q_batch, topk_scores, topk_indices
        if use_cuda:
            torch.cuda.empty_cache()

    return topk_scores_all, topk_indices_all


def percentile_thresholds(margins: torch.Tensor, pcts: List[int]) -> Dict[int, float]:
    m = margins.float()
    out = {}
    for p in pcts:
        q = p / 100.0
        try:
            thr = torch.quantile(m, q).item()
        except Exception:
            sorted_m = torch.sort(m).values
            idx = int(round(q * (sorted_m.numel() - 1)))
            idx = max(0, min(sorted_m.numel() - 1, idx))
            thr = sorted_m[idx].item()
        out[p] = float(thr)
    return out


# =========================
# Main pipeline
# =========================
def build_or_load_train_doc_cache(tokenizer, model, key: str, paths: Dict[str, str]):
    # If exists and not forcing rebuild
    if (not CFG.force_rebuild_doc_cache) and os.path.exists(paths["doc_cache"]) and os.path.exists(paths["keep_cache"]) and os.path.exists(paths["meta_cache"]):
        meta = json.load(open(paths["meta_cache"], "r", encoding="utf-8"))
        if meta.get("cache_key") == key:
            d_reps = load_fp16_cpu(paths["doc_cache"])
            keep = torch.load(paths["keep_cache"], map_location="cpu")
            offset = int(torch.load(paths["offset_cache"], map_location="cpu")) if os.path.exists(paths["offset_cache"]) else int(d_reps.size(0))
            return d_reps, keep, offset, meta

    # Rebuild
    queries, passages, keep = load_train_rows(TRAIN_PT)
    N = len(passages)
    if N == 0:
        raise RuntimeError("No usable (query,pos) rows found in TRAIN_PT.")

    # Encode docs (train passages only)
    device = CFG.device if torch.cuda.is_available() and CFG.device == "cuda" else "cpu"
    d_reps_f32 = encode_texts_train_style(
        model=model,
        tokenizer=tokenizer,
        texts=passages,
        template_key="passage",
        batch_size=CFG.batch_size_doc,
        max_length=CFG.max_len,
        device=device,
        desc="Encode TRAIN docs (passages) [TRAIN-style]",
    )

    # Save cache
    save_tensor(paths["doc_cache"], d_reps_f32, dtype=CFG.save_cache_dtype)
    torch.save(torch.tensor(keep, dtype=torch.long), paths["keep_cache"])
    torch.save(int(N), paths["offset_cache"])  # train-only => offset = N

    meta = {
        "cache_key": key,
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR,
        "train_pt": TRAIN_PT,
        "templates": TEMPLATES,
        "max_len": CFG.max_len,
        "pooling": "eos_last_token",
        "padding_side": "right",
        "force_eos": True,
        "use_bfloat16_backbone": CFG.use_bfloat16_backbone,
        "cache_dtype": CFG.save_cache_dtype,
        "doc_cache": paths["doc_cache"],
        "keep_indices": paths["keep_cache"],
        "offset_cache": paths["offset_cache"],
        "num_train_pairs_used": int(N),
        "note": "train-compatible doc_cache: docs=train passages only, aligned so pos_doc_index=i.",
    }
    json.dump(meta, open(paths["meta_cache"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    d_reps_fp16 = load_fp16_cpu(paths["doc_cache"])
    keep_t = torch.load(paths["keep_cache"], map_location="cpu")
    offset = int(torch.load(paths["offset_cache"], map_location="cpu"))
    return d_reps_fp16, keep_t, offset, meta


def build_or_load_train_query_cache(tokenizer, model, key: str, paths: Dict[str, str], train_queries: List[str]):
    if (not CFG.force_rebuild_query_cache) and os.path.exists(paths["q_cache"]) and os.path.exists(paths["q_meta"]):
        meta = json.load(open(paths["q_meta"], "r", encoding="utf-8"))
        if meta.get("cache_key") == key and meta.get("num_queries") == len(train_queries):
            q_reps = load_fp16_cpu(paths["q_cache"])
            return q_reps, meta

    device = CFG.device if torch.cuda.is_available() and CFG.device == "cuda" else "cpu"
    q_reps_f32 = encode_texts_train_style(
        model=model,
        tokenizer=tokenizer,
        texts=train_queries,
        template_key="query_mat_paper",
        batch_size=CFG.batch_size_query,
        max_length=CFG.max_len,
        device=device,
        desc="Encode TRAIN queries [TRAIN-style]",
    )

    save_tensor(paths["q_cache"], q_reps_f32, dtype=CFG.save_cache_dtype)

    meta = {
        "cache_key": key,
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR,
        "train_pt": TRAIN_PT,
        "max_len": CFG.max_len,
        "template": "query_mat_paper",
        "pooling": "eos_last_token",
        "padding_side": "right",
        "force_eos": True,
        "cache_dtype": CFG.save_cache_dtype,
        "q_cache": paths["q_cache"],
        "num_queries": int(len(train_queries)),
        "note": "train-compatible query_cache.",
    }
    json.dump(meta, open(paths["q_meta"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    q_reps_fp16 = load_fp16_cpu(paths["q_cache"])
    return q_reps_fp16, meta


def build_or_load_norm_cache(key: str, paths: Dict[str, str], d_reps_fp16: torch.Tensor, q_reps_fp16: torch.Tensor):
    if (not CFG.force_rebuild_doc_cache) and os.path.exists(paths["d_norm"]) and os.path.exists(paths["q_norm"]) and os.path.exists(paths["n_meta"]):
        meta = json.load(open(paths["n_meta"], "r", encoding="utf-8"))
        if meta.get("cache_key") == key and meta.get("num_docs") == int(d_reps_fp16.size(0)) and meta.get("num_queries") == int(q_reps_fp16.size(0)):
            d_norm = load_fp16_cpu(paths["d_norm"])
            q_norm = load_fp16_cpu(paths["q_norm"])
            return d_norm, q_norm, meta

    d_norm = d_reps_fp16.clone()
    q_norm = q_reps_fp16.clone()
    normalize_inplace_fp16_cpu(d_norm, CFG.normalize_chunk, desc="Normalize TRAIN docs (CPU fp16)")
    normalize_inplace_fp16_cpu(q_norm, CFG.normalize_chunk, desc="Normalize TRAIN queries (CPU fp16)")

    torch.save(d_norm, paths["d_norm"])
    torch.save(q_norm, paths["q_norm"])

    meta = {
        "cache_key": key,
        "dtype": "float16",
        "num_docs": int(d_norm.size(0)),
        "num_queries": int(q_norm.size(0)),
        "d_norm": paths["d_norm"],
        "q_norm": paths["q_norm"],
    }
    json.dump(meta, open(paths["n_meta"], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return d_norm, q_norm, meta


def main():
    device = CFG.device if (CFG.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[Device] {device} (gpu_id={CFG.gpu_id})")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"

    # Model
    model = RetrieverEncoder(
        base_model=BASE_MODEL,
        adapter_dir=ADAPTER_DIR,
        device=device,
        use_bfloat16=CFG.use_bfloat16_backbone,
    ).to(device)

    key = train_cache_key()
    paths = cache_paths(key)
    print(f"[CacheKey] {key}")

    # 1) Build/load train-compatible doc_cache
    d_reps_fp16, keep_indices, valid_pos_offset, doc_meta = build_or_load_train_doc_cache(tokenizer, model, key, paths)
    print(f"[DocCache] docs={tuple(d_reps_fp16.shape)} dtype={d_reps_fp16.dtype}")
    print(f"[Keep] keep_indices={keep_indices.numel()} valid_pos_offset={valid_pos_offset}")

    # 2) Load TRAIN queries aligned to docs (same filtering/order as doc_cache build)
    #    We rebuild queries list from TRAIN_PT and apply keep_indices to be safe.
    #    But note: our doc_cache build already used the filtered order; simplest is to re-load and re-filter in same way.
    train_queries, train_passages, keep2 = load_train_rows(TRAIN_PT)
    if CFG.strict_alignment:
        # Since doc_cache is built from load_train_rows output order, keep_indices should match keep2 exactly.
        # If mismatch => you changed filtering logic. Stop for correctness.
        if len(keep2) != keep_indices.numel():
            raise RuntimeError(f"[ALIGN] keep count mismatch: doc_cache_keep={keep_indices.numel()} vs reload_keep={len(keep2)}")
        # also spot-check first/last
        if keep2[0] != int(keep_indices[0].item()) or keep2[-1] != int(keep_indices[-1].item()):
            raise RuntimeError("[ALIGN] keep_indices content mismatch. Filtering/order changed. Rebuild caches or unify loader logic.")
        if len(train_passages) != d_reps_fp16.size(0):
            raise RuntimeError(f"[ALIGN] Ndocs mismatch: passages={len(train_passages)} vs doc_cache={d_reps_fp16.size(0)}")

    Ntrain = len(train_queries)
    print(f"[Train] Ntrain_pairs={Ntrain}")

    # 3) Build/load train-compatible query_cache
    q_reps_fp16, q_meta = build_or_load_train_query_cache(tokenizer, model, key, paths, train_queries)
    print(f"[QueryCache] queries={tuple(q_reps_fp16.shape)} dtype={q_reps_fp16.dtype}")
    if q_reps_fp16.size(0) != d_reps_fp16.size(0):
        raise RuntimeError(f"[ALIGN] q_reps({q_reps_fp16.size(0)}) != d_reps({d_reps_fp16.size(0)})")

    # 4) Normalize caches (CPU fp16)
    d_norm, q_norm, n_meta = build_or_load_norm_cache(key, paths, d_reps_fp16, q_reps_fp16)

    # 5) TopK search (FAISS preferred)
    print("[Search] TopK cosine (via IP on normalized vectors)")
    topk_scores = None
    topk_indices = None
    used = None

    if CFG.prefer_faiss:
        faiss = try_import_faiss()
        if faiss is not None:
            try:
                t0 = time.time()
                topk_scores, topk_indices = topk_search_faiss_ip(
                    q_norm_fp16_cpu=q_norm,
                    d_norm_fp16_cpu=d_norm,
                    topk=CFG.topk,
                    use_gpu=(CFG.faiss_use_gpu and device == "cuda"),
                    gpu_id=CFG.gpu_id,
                )
                used = "faiss_gpu" if (CFG.faiss_use_gpu and device == "cuda") else "faiss_cpu"
                print(f"[Search] FAISS done ({used}) in {time.time()-t0:.2f}s")
            except Exception as e:
                print(f"[Search] FAISS failed -> fallback torch ({e})")

    if topk_scores is None:
        t0 = time.time()
        topk_scores, topk_indices = topk_search_torch_fast(
            q_normed_fp16_cpu=q_norm,
            d_normed_fp16_cpu=d_norm,
            topk=CFG.topk,
            doc_chunk=CFG.doc_chunk,
            query_batch=CFG.query_batch,
            device=device,
            use_fp16_matmul=CFG.use_fp16_matmul,
            pin_memory=CFG.pin_memory,
        )
        used = "torch_gpu_matmul" if device == "cuda" else "torch_cpu"
        print(f"[Search] Torch done ({used}) in {time.time()-t0:.2f}s")

    # 6) Margin computation: pos index = i
    Nq = q_norm.size(0)
    pos_idx = torch.arange(Nq, dtype=torch.long)
    pos_scores = (q_norm.float() * d_norm[pos_idx].float()).sum(dim=1)

    top1_docidx = topk_indices[:, 0].clone()
    top1_score = topk_scores[:, 0].clone()

    max_neg_scores = torch.empty(Nq, dtype=torch.float32)
    max_neg_docidx = torch.empty(Nq, dtype=torch.long)

    for i in tqdm(range(Nq), desc="Compute max-neg (exclude pos)"):
        p = int(pos_idx[i].item())
        best_s = -1e9
        best_j = -1
        # topk list에서 pos만 제외하고 최고 점수
        for j, s in zip(topk_indices[i].tolist(), topk_scores[i].tolist()):
            if j == p:
                continue
            if s > best_s:
                best_s = s
                best_j = j
        max_neg_scores[i] = best_s
        max_neg_docidx[i] = best_j

    margins = pos_scores - max_neg_scores
    thr_by_pct = percentile_thresholds(margins, BOTTOM_PCTS)

    # 7) Write ALL margins
    if CFG.write_all_jsonl:
        if (not CFG.overwrite_outputs) and os.path.exists(ALL_JSONL):
            print(f"[Skip] ALL_JSONL exists: {ALL_JSONL}")
        else:
            with open(ALL_JSONL, "w", encoding="utf-8") as f:
                for i in tqdm(range(Nq), desc="Write TRAIN ALL margins (cos)"):
                    rec = {
                        "i": i,
                        "query": train_queries[i],
                        "pos_doc_index": int(pos_idx[i].item()),
                        "pos_score_cos": float(pos_scores[i].item()),
                        "top1_doc_index_cos": int(top1_docidx[i].item()),
                        "top1_score_cos": float(top1_score[i].item()),
                        "max_neg_doc_index_cos": int(max_neg_docidx[i].item()),
                        "max_neg_score_cos": float(max_neg_scores[i].item()),
                        "margin_cos": float(margins[i].item()),
                        "topk_doc_indices_cos": topk_indices[i].tolist(),
                        "topk_scores_cos": topk_scores[i].tolist(),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[Saved] {ALL_JSONL}")

    # 8) Write percentile confused sets
    pct_stats = {}
    for p in BOTTOM_PCTS:
        thr = thr_by_pct[p]
        mask = margins <= thr
        num = int(mask.sum().item())
        ratio = float(num) / max(1, Nq)
        out_conf = os.path.join(OUT_DIR, f"confused_train_queries.bottom_{p:02d}pct.cos.jsonl")

        pct_stats[str(p)] = {
            "threshold": thr,
            "confused": num,
            "ratio": ratio,
            "output_jsonl": out_conf,
        }

        if (not CFG.overwrite_outputs) and os.path.exists(out_conf):
            print(f"[Skip] {out_conf}")
            continue

        idxs = torch.nonzero(mask).view(-1).tolist()
        with open(out_conf, "w", encoding="utf-8") as f:
            for i in tqdm(idxs, desc=f"Write TRAIN bottom {p}% (cos)"):
                rec = {
                    "i": i,
                    "query": train_queries[i],
                    "margin_cos": float(margins[i].item()),
                    "pos_doc_index": int(pos_idx[i].item()),
                    "pos_score_cos": float(pos_scores[i].item()),
                    "max_neg_doc_index_cos": int(max_neg_docidx[i].item()),
                    "max_neg_score_cos": float(max_neg_scores[i].item()),
                    "top1_doc_index_cos": int(top1_docidx[i].item()),
                    "top1_score_cos": float(top1_score[i].item()),
                    "topk_doc_indices_cos": topk_indices[i].tolist(),
                    "topk_scores_cos": topk_scores[i].tolist(),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[Saved] {out_conf}")

    # 9) Meta
    m = margins
    meta = {
        "task": "train_percentile_diagnosis",
        "base_model": BASE_MODEL,
        "adapter_dir": ADAPTER_DIR,
        "train_pt": TRAIN_PT,
        "cache_key": key,
        "cache_dir": CACHE_DIR,
        "doc_cache_meta": paths["meta_cache"],
        "doc_cache": paths["doc_cache"],
        "keep_indices": paths["keep_cache"],
        "offset_cache": paths["offset_cache"],
        "query_cache": paths["q_cache"],
        "query_meta": paths["q_meta"],
        "norm_cache": {
            "d_norm": paths["d_norm"],
            "q_norm": paths["q_norm"],
            "n_meta": paths["n_meta"],
        },
        "encoding_rules": {
            "templates": TEMPLATES,
            "padding_side": "right",
            "force_eos": True,
            "pooling": "eos_last_token",
            "max_len": CFG.max_len,
        },
        "counts": {
            "num_train_pairs_used": int(Nq),
            "valid_pos_offset": int(valid_pos_offset),
        },
        "search": {
            "method": used,
            "topk": CFG.topk,
            "doc_chunk": CFG.doc_chunk,
            "query_batch": CFG.query_batch,
            "use_fp16_matmul": CFG.use_fp16_matmul,
            "pin_memory": CFG.pin_memory,
            "prefer_faiss": CFG.prefer_faiss,
            "faiss_use_gpu": CFG.faiss_use_gpu,
            "gpu_id": CFG.gpu_id,
        },
        "outputs": {
            "all_margins_jsonl": ALL_JSONL if CFG.write_all_jsonl else None,
            "bottom_percentiles": BOTTOM_PCTS,
            "percentile_thresholds": {str(k): v for k, v in thr_by_pct.items()},
            "percentile_stats": pct_stats,
        },
        "margin_cos_stats": {
            "min": float(m.min().item()),
            "max": float(m.max().item()),
            "mean": float(m.mean().item()),
            "p01": float(torch.quantile(m, 0.01).item()),
            "p05": float(torch.quantile(m, 0.05).item()),
            "p10": float(torch.quantile(m, 0.10).item()),
            "p50": float(torch.quantile(m, 0.50).item()),
            "p90": float(torch.quantile(m, 0.90).item()),
            "p95": float(torch.quantile(m, 0.95).item()),
            "p99": float(torch.quantile(m, 0.99).item()),
        },
        "notes": [
            "doc_cache is rebuilt from TRAIN_PT with TRAIN-style passage template + EOS forcing + EOS last-token pooling.",
            "queries are encoded with TRAIN-style query_mat_paper template + EOS forcing + EOS last-token pooling.",
            "alignment is guaranteed because docs=train passages only and pos_doc_index=i for i-th query.",
            "topk search uses FAISS FlatIP if available; otherwise uses torch pinned-memory chunked matmul.",
        ],
    }

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Saved] META: {META_JSON}")
    print("[Done] train percentile diagnosis complete.")


if __name__ == "__main__":
    main()
