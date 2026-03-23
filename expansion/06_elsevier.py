# python 06.Elsever.py
# 06.elsevier_scopus_batch_from_txt_json_tqdm.py

'''
python 06.Elsever.py \
  --concept_txt /gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1/PathAB_terms/final_terms.bottom_30pct.txt \
  --out_jsonl /gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1/06.elsevier_iter1.jsonl \
  --target_k 15 --workers 9999 --page_count 25 --max_start 200 --view COMPLETE \
  --resume

'''
# 06.elsevier_concept_parallel_by_keys.py
# ------------------------------------------------------------
# Key idea:
#   - Parallelize by CONCEPTS using API keys as worker identity
#   - 1 worker = 1 API key (fixed)
#   - Each worker processes concepts sequentially from a shared queue
#
# Output:
#   - JSONL (one line per concept result) for robustness
#   - Optional: merge JSONL -> one JSON at end (separate script)
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

DEFAULT_KEY_FILE = "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/Elsevier.txt"
DEFAULT_CONCEPT_TXT = "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/PathAB_terms/final_terms.bottom_100pct.txt"
DEFAULT_OUT_JSONL = "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/06.elsevier_scopus.bottom_100pct.jsonl"

SCOPUS_SEARCH_ENDPOINT = "https://api.elsevier.com/content/search/scopus"

def should_accept(papers_all):
    if not papers_all:
        return False
    non_empty_title = sum(1 for p in papers_all if norm(p.get("title", "")))
    return non_empty_title > 0  # title이 하나라도 있어야 accept
# -------------------------
# Utils
# -------------------------
def norm(s: str) -> str:
    return (s or "").strip()

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

def title_key(title: str) -> str:
    t = norm(title).lower()
    t = re.sub(r"\s+", " ", t)
    return t

def dedup_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for p in papers:
        doi = norm(p.get("doi", "")).lower()
        ttl = title_key(p.get("title", ""))
        key = ("doi:" + doi) if doi else ("ttl:" + ttl)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out

def ensure_parent(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)

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
# Scopus Search API (single page)
# -------------------------
def extract_scopus_entry(it: Dict[str, Any]) -> Dict[str, Any]:
    title = it.get("dc:title", "") or ""
    doi = it.get("prism:doi", "") or ""
    eid = it.get("eid", "") or ""
    cover_date = it.get("prism:coverDate", "") or ""
    venue = it.get("prism:publicationName", "") or ""
    abstract = it.get("dc:description", "") or ""  # may be empty depending on entitlement
    year = int(cover_date[:4]) if (cover_date and cover_date[:4].isdigit()) else None

    url = ""
    links = it.get("link")
    if isinstance(links, list):
        for lk in links:
            if isinstance(lk, dict) and lk.get("@ref") in ("scopus", "self"):
                url = lk.get("@href", "") or url

    return {
        "eid": eid,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "year": year,
        "venue": venue,
        "coverDate": cover_date,
        "url": url,
        "source": "elsevier_scopus",
    }

def scopus_fetch_page(query: str,
                      start: int,
                      count: int,
                      api_key: str,
                      insttoken: Optional[str] = None,
                      view: str = "STANDARD",
                      timeout: int = 60) -> Dict[str, Any]:
    headers = {"Accept": "application/json", "X-ELS-APIKey": api_key}
    if insttoken:
        headers["X-ELS-Insttoken"] = insttoken

    params = {"query": query, "start": start, "count": count, "view": view}

    r = requests.get(SCOPUS_SEARCH_ENDPOINT, headers=headers, params=params, timeout=timeout)
    status = r.status_code
    if status != 200:
        return {"status": status, "entries": []}

    data = r.json()
    sr = data.get("search-results", {}) or {}
    entries = sr.get("entry", []) or []
    if not isinstance(entries, list):
        entries = []
    return {"status": status, "entries": entries}


def query_variants(concept: str) -> List[str]:
    c = norm(concept)
    if not c:
        return []
    vars_ = [c, c.replace("-", " "), c.replace("-", "")]
    parts = re.split(r"[-\s]+", c)
    for p in parts:
        if len(p) >= 8:
            vars_.append(p)
    # dedup preserve order
    seen = set()
    out = []
    for q in vars_:
        k = q.lower()
        if k not in seen and q.strip():
            seen.add(k)
            out.append(q)
    return out


def get_k_scopus_for_concept(concept: str,
                             api_key: str,
                             target_k: int,
                             page_count: int,
                             max_start: int,
                             view: str,
                             insttoken: Optional[str],
                             timeout: int,
                             per_concept_sleep: float = 0.0) -> Dict[str, Any]:
    """
    IMPORTANT:
      - No inner parallelism. One worker handles one concept using one key.
      - Pages are fetched sequentially until target_k is met.
      - Fallback query variants if needed.
    """
    t0 = time.time()
    dbg = {"last_status": None, "used_query": "", "tries": []}


    tried = []
    for q in query_variants(concept):
        papers_all = []

        for start in range(0, max_start + 1, page_count):
            resp = scopus_fetch_page(q, start, page_count, api_key, insttoken, view, timeout)
            entries = resp.get("entries", []) or []
            if entries:
                for it in entries:
                    if isinstance(it, dict):
                        papers_all.append(extract_scopus_entry(it))
                papers_all = dedup_papers(papers_all)

            if len(papers_all) >= target_k:
                break

        tried.append({
            "q": q,
            "hits": len(papers_all),
            "non_empty_title": sum(1 for p in papers_all if norm(p.get("title","")))
        })

        # ✅ 핵심: 0개면(혹은 title 다 비면) 다음 variant로
        if not should_accept(papers_all):
            continue

        papers_all.sort(key=lambda p: 0 if norm(p.get("abstract","")) else 1)
        return {
            "concept": concept,
            "used_query": q,
            "got_k": min(len(papers_all), target_k),
            "papers": papers_all[:target_k],
            "tried_queries": tried,
        }

    # 모든 variant 실패
    return {
        "concept": concept,
        "used_query": "",
        "got_k": 0,
        "papers": [],
        "tried_queries": tried,
    }


# -------------------------
# Worker pool by keys
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
            res = get_k_scopus_for_concept(
                concept=concept,
                api_key=api_key,
                target_k=args.target_k,
                page_count=args.page_count,
                max_start=args.max_start,
                view=args.view,
                insttoken=args.insttoken,
                timeout=args.timeout,
                per_concept_sleep=args.per_page_sleep,
            )
        except Exception as e:
            res = {
                "concept": concept,
                "got_k": 0,
                "papers": [],
                "error": str(e),
                "elapsed_sec": None,
                "status": None,
            }

        # append JSONL safely
        with lock:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            pbar.update(1)
            pbar.set_postfix_str(f"last_got={res.get('got_k',0)} key_worker={worker_id}")

        q.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_txt", default=DEFAULT_CONCEPT_TXT)
    ap.add_argument("--key_file", default=DEFAULT_KEY_FILE)
    ap.add_argument("--out_jsonl", default=DEFAULT_OUT_JSONL)

    ap.add_argument("--target_k", type=int, default=15)
    ap.add_argument("--workers", type=int, default=999999)  # will cap by num keys

    ap.add_argument("--page_count", type=int, default=25)
    ap.add_argument("--max_start", type=int, default=200)
    ap.add_argument("--view", type=str, default="STANDARD")

    ap.add_argument("--insttoken", type=str, default="")
    ap.add_argument("--timeout", type=int, default=60)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--per_page_sleep", type=float, default=0.0, help="tiny sleep between page calls (per worker)")

    args = ap.parse_args()

    keys = load_keys(args.key_file)
    concepts = load_concepts_txt(args.concept_txt)

    ensure_parent(args.out_jsonl)

    done = load_done_from_jsonl(args.out_jsonl) if args.resume else set()
    todo = [c for c in concepts if c.lower() not in done]

    insttoken = norm(args.insttoken)
    args.insttoken = insttoken if insttoken else None

    # ✅ workers = min(user_workers, num_keys, num_concepts)
    eff_workers = min(args.workers, len(keys), len(todo))
    keys = keys[:eff_workers]  # one key per worker

    print(f"[INFO] total concepts={len(concepts)} done={len(done)} todo={len(todo)}")
    print(f"[INFO] keys={len(load_keys(args.key_file))} effective_workers={eff_workers}")
    print(f"[INFO] out={args.out_jsonl}")

    q = Queue()
    for c in todo:
        q.put(c)

    lock = threading.Lock()
    pbar = tqdm(total=len(todo), desc="Elsevier concepts (parallel-by-keys)", dynamic_ncols=True)

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
