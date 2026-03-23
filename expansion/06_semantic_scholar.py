# python 06.sementic.py
# https://chatgpt.com/c/69730a6e-476c-8320-a800-661c72924839

# 06.semantic_batch_from_txt_json_tqdm.py
'''
python 06.sementic.py \
  --concept_txt /gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/PathAB_terms/final_terms.bottom_100pct.txt \
  --out_jsonl /gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/06.semantic.bottom_100pct.jsonl \
  --target_k 20 --workers 9999 --resume


'''
# 06.semantic_parallel_by_keys.py
# ------------------------------------------------------------
# Concept-level parallelism using Semantic Scholar API keys.
#  - 1 worker thread = 1 fixed API key
#  - Each worker pulls concepts from a shared queue
#  - Each concept fetches offsets sequentially (0,100,200,...) until target_k
#  - Output JSONL (one line per concept result) for robustness + resume
#
# Run example:
#   sr 1 48 python 06.semantic_parallel_by_keys.py \
#     --concept_txt /.../final_terms.bottom_100pct.txt \
#     --out_jsonl /.../06.semantic.bottom_100pct.jsonl \
#     --target_k 20 --workers 9999 --resume
# ------------------------------------------------------------

import os
import re
import json
import time
import argparse
import threading
from typing import List, Dict, Any, Optional
from queue import Queue, Empty

import requests
from tqdm import tqdm

DEFAULT_KEY_FILE = "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/semantic_api.txt"
DEFAULT_CONCEPT_TXT = "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/PathAB_terms/final_terms.bottom_100pct.txt"
DEFAULT_OUT_JSONL = "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/06.semantic.bottom_100pct.jsonl"


# -------------------------
# Utils
# -------------------------
def norm(s: str) -> str:
    return (s or "").strip()

def title_key(title: str) -> str:
    return re.sub(r"\s+", " ", norm(title).lower())

def ensure_parent(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)

def load_keys(path: str) -> List[str]:
    keys = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            k = line.strip()
            if k and not k.startswith("#"):
                keys.append(k)
    if not keys:
        raise ValueError(f"No API keys found in {path}")
    return keys

def load_concepts_txt(path: str) -> List[str]:
    seen = set()
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            k = s.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(s)
    return out

def dedup_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in papers:
        doi = norm(p.get("doi", "")).lower()
        key = f"doi:{doi}" if doi else f"ttl:{title_key(p.get('title',''))}"
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def load_done_from_jsonl(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                c = norm(obj.get("concept", "")).lower()
                if c:
                    done.add(c)
            except Exception:
                continue
    return done


# -------------------------
# Semantic Scholar API
# -------------------------
def s2_fetch_page(query: str, limit: int, offset: int, api_key: str, timeout: int = 60) -> List[Dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "paperId,title,abstract,year,venue,externalIds,url"

    headers = {"x-api-key": api_key} if api_key else {}
    params = {"query": query, "limit": min(limit, 100), "offset": offset, "fields": fields}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []

    out = []
    for it in data.get("data", []) or []:
        ext = it.get("externalIds") or {}
        doi = ext.get("DOI") or ext.get("Doi") or ""
        out.append({
            "pid": it.get("paperId", "") or "",
            "doi": doi or "",
            "title": it.get("title", "") or "",
            "abstract": it.get("abstract", "") or "",
            "year": it.get("year", None),
            "venue": it.get("venue", "") or "",
            "url": it.get("url", "") or "",
            "source": "semantic_scholar"
        })
    return out


def get_k_from_s2_single_key(concept: str,
                             api_key: str,
                             target_k: int = 20,
                             page_limit: int = 100,
                             max_offset: int = 900,
                             timeout: int = 60,
                             per_page_sleep: float = 0.0) -> Dict[str, Any]:
    """
    One concept, using ONE API key (fixed per worker).
    Offsets fetched sequentially until target_k.
    """
    t0 = time.time()
    concept = norm(concept)
    if not concept:
        return {"concept": concept, "got_k": 0, "papers": [], "elapsed_sec": 0.0}

    papers: List[Dict[str, Any]] = []
    last_offset = None

    for offset in range(0, max_offset + 1, page_limit):
        last_offset = offset
        got = s2_fetch_page(concept, page_limit, offset, api_key, timeout=timeout)
        if got:
            papers.extend(got)
            papers = dedup_papers(papers)
        if len(papers) >= target_k:
            break
        if per_page_sleep > 0:
            time.sleep(per_page_sleep)

    papers.sort(key=lambda p: 0 if norm(p.get("abstract", "")) else 1)
    papers = papers[:target_k]

    return {
        "concept": concept,
        "got_k": len(papers),
        "papers": papers,
        "elapsed_sec": round(time.time() - t0, 3),
        "last_offset": last_offset,
    }


# -------------------------
# Worker pool
# -------------------------
def worker_loop(worker_id: int,
                api_key: str,
                q: Queue,
                out_path: str,
                lock: threading.Lock,
                pbar: tqdm,
                args):
    while True:
        try:
            concept = q.get_nowait()
        except Empty:
            return

        try:
            res = get_k_from_s2_single_key(
                concept=concept,
                api_key=api_key,
                target_k=args.target_k,
                page_limit=args.page_limit,
                max_offset=args.max_offset,
                timeout=args.timeout,
                per_page_sleep=args.per_page_sleep,
            )
        except Exception as e:
            res = {
                "concept": concept,
                "got_k": 0,
                "papers": [],
                "error": str(e),
            }

        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            pbar.update(1)
            pbar.set_postfix_str(f"last_got={res.get('got_k',0)} worker={worker_id}")

        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_txt", default=DEFAULT_CONCEPT_TXT)
    ap.add_argument("--key_file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--out_jsonl", default=DEFAULT_OUT_JSONL)

    ap.add_argument("--target_k", type=int, default=20)
    ap.add_argument("--workers", type=int, default=999999)  # will cap by key count
    ap.add_argument("--page_limit", type=int, default=100)
    ap.add_argument("--max_offset", type=int, default=900)
    ap.add_argument("--timeout", type=int, default=60)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--per_page_sleep", type=float, default=0.0)
    args = ap.parse_args()

    keys = load_keys(args.key_file)
    concepts = load_concepts_txt(args.concept_txt)

    ensure_parent(args.out_jsonl)

    done = load_done_from_jsonl(args.out_jsonl) if args.resume else set()
    todo = [c for c in concepts if c.lower() not in done]

    eff_workers = min(args.workers, len(keys), len(todo))
    keys = keys[:eff_workers]  # one key per worker

    print(f"[INFO] total concepts={len(concepts)} done={len(done)} todo={len(todo)}")
    print(f"[INFO] keys={len(load_keys(args.key_file))} effective_workers={eff_workers}")
    print(f"[INFO] out_jsonl={args.out_jsonl}")

    q = Queue()
    for c in todo:
        q.put(c)

    lock = threading.Lock()
    pbar = tqdm(total=len(todo), desc="SemanticScholar concepts (parallel-by-keys)", dynamic_ncols=True)

    threads = []
    for wid, key in enumerate(keys):
        t = threading.Thread(target=worker_loop, args=(wid, key, q, args.out_jsonl, lock, pbar, args), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    pbar.close()
    print(f"[OK] appended JSONL: {args.out_jsonl}")


if __name__ == "__main__":
    main()
