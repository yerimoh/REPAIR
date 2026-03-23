# pathb_finch_train.py
# REPAIR Path B: Inter-Query Distractor Mining - TRAIN version
# - Cluster confused queries using FINCH (Sarfraz et al., 2019)
#   FINCH: each point connects to its first nearest neighbor (1-NN)
#   Connected components of the resulting undirected graph form clusters.
#   Parameter-free: no k to tune, no predefined cluster count.
# - From each cluster, aggregate extracted candidate concepts and select the most frequent → Cinter
#
# Run:
#   sr 1 48 python 02.pathb.py --pct 30
#
# Outputs:
#   OUT_DIR/pathB_finch_train.assignments.bottom_XXpct.jsonl
#   OUT_DIR/pathB_finch_train.components.bottom_XXpct.json  (Cinter)

import os
import json
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

# =========================
# Defaults (edit if needed)
# =========================
DEFAULT_DIA_OUT_DIR = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1"
DEFAULT_OUT_DIR = os.path.join(DEFAULT_DIA_OUT_DIR, "pathB_finch_train")
DEFAULT_TRAIN_QUERY_CACHE = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/cache/cache.train_query_reps.normfp16.26cd5f049cb0698c4c518b6887a98304.pt"
DEFAULT_PATHA_OUT_DIR = os.path.join(DEFAULT_DIA_OUT_DIR, "pathA_ccs_train")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def load_reps_fp16(path: str) -> torch.Tensor:
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict) and "reps" in x:
        x = x["reps"]
    if x.dtype != torch.float16:
        x = x.half()
    return x

def load_per_query_concepts(patha_out_dir: str, pct: int) -> Dict[int, List[str]]:
    """Load per-query neg concepts saved by PathA (for Cinter selection)."""
    path = os.path.join(patha_out_dir, f"per_query_neg_concepts.bottom_{pct:02d}pct.json")
    if not os.path.exists(path):
        print(f"[WARN] PathA per-query concepts not found: {path}")
        print("       Run 02.pathA_doc_entity_ccs.py first. Falling back to empty concepts.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# =========================
# Union-Find for connected components
# =========================
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# =========================
# FINCH: First Integer Neighbor Clustering Hierarchy (Sarfraz et al., 2019)
# =========================
@torch.no_grad()
def finch_1nn_components(
    X_norm: torch.Tensor,
    device: str = "cuda",
    chunk: int = 4096,
) -> Dict[int, List[int]]:
    """
    FINCH partition 1 (Sarfraz et al., 2019).

    Algorithm:
      1. For each point i, find its first nearest neighbor: nn[i] = argmax_{j≠i} sim(i, j)
      2. Build undirected graph: connect i <-> nn[i]
      3. Connected components of this graph = FINCH clusters

    This is parameter-free (no k to tune, no predefined cluster count).
    """
    N = X_norm.shape[0]
    if N <= 1:
        return {0: list(range(N))}

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = torch.device("cuda" if use_cuda else "cpu")
    X = X_norm.float().to(dev)

    # Compute 1-NN for each point via chunked inner product
    nn = torch.empty(N, dtype=torch.long, device="cpu")
    for s in tqdm(range(0, N, chunk), desc="FINCH: 1-NN search"):
        e = min(N, s + chunk)
        sim = X[s:e] @ X.T           # [chunk, N]
        # Mask self-similarity
        for local_i in range(e - s):
            sim[local_i, s + local_i] = -2.0
        nn[s:e] = sim.argmax(dim=1).cpu()
        del sim

    if use_cuda:
        torch.cuda.empty_cache()

    # Build undirected graph: i <-> nn[i], then find connected components
    uf = UnionFind(N)
    for i in range(N):
        uf.union(i, int(nn[i]))

    comps: Dict[int, List[int]] = defaultdict(list)
    for i in range(N):
        comps[uf.find(i)].append(i)
    return dict(comps)

# =========================
# Main
# =========================
def main(
    pct: int,
    dia_out_dir: str,
    out_dir: str,
    train_query_cache: str,
    patha_out_dir: str,
    device: str,
):
    os.makedirs(out_dir, exist_ok=True)

    conf_path = os.path.join(dia_out_dir, f"confused_train_queries.bottom_{pct:02d}pct.cos.jsonl")
    if not os.path.exists(conf_path):
        raise FileNotFoundError(f"missing: {conf_path}")

    conf = read_jsonl(conf_path)
    N = len(conf)
    if N == 0:
        raise RuntimeError("empty confused set")

    use_cuda = (device == "cuda" and torch.cuda.is_available())
    dev = "cuda" if use_cuda else "cpu"

    print(f"[PathB-FINCH-TRAIN] pct={pct}  confused={N}  device={dev}")
    print(f"[Input] {conf_path}")
    print(f"[TrainQueryCache] {train_query_cache}")

    # Load normalized train query embeddings, slice to confused indices
    q_reps = load_reps_fp16(train_query_cache).float()  # [Ntrain, D]
    q_reps = F.normalize(q_reps, p=2, dim=1)

    idxs = [int(r["i"]) for r in conf]
    X = q_reps[idxs].contiguous()   # [N, D] normalized float32 CPU

    # FINCH: parameter-free 1-NN clustering (Sarfraz et al., 2019)
    print("[FINCH] Building 1-NN graph and finding connected components...")
    comps = finch_1nn_components(X, device=dev)
    print(f"[FINCH] {len(comps)} clusters from {N} confused queries")

    # Load per-query neg concepts from PathA (for Cinter selection)
    per_query_concepts = load_per_query_concepts(patha_out_dir, pct)

    # Build reverse map: local index → component root
    local_to_root: Dict[int, int] = {}
    for root, members in comps.items():
        for j in members:
            local_to_root[j] = int(root)

    # Cinter: for each FINCH cluster, select the most frequent concept
    # (REPAIR §3.3: "we aggregate the previously extracted candidate concepts
    #  and select the most frequent one")
    summaries = []
    for root, members_local in comps.items():
        ms = [float(conf[j].get("margin_cos", 0.0)) for j in members_local]
        avg_m = float(sum(ms) / len(ms)) if ms else 0.0

        # Aggregate candidate concepts from all member queries
        concept_counter: Counter = Counter()
        for j in members_local:
            qi = int(conf[j]["i"])
            for c in per_query_concepts.get(qi, []):
                concept_counter[c] += 1

        most_frequent = concept_counter.most_common(1)[0][0] if concept_counter else ""
        most_frequent_count = int(concept_counter[most_frequent]) if most_frequent else 0

        summaries.append({
            "component_root": int(root),
            "size": int(len(members_local)),
            "avg_margin": avg_m,
            "min_margin": float(min(ms)) if ms else 0.0,
            "concept": most_frequent,           # Cinter concept for this cluster
            "concept_count": most_frequent_count,
            "sample_queries": [conf[j].get("query", "") for j in members_local[:5]],
        })

    summaries.sort(key=lambda x: (x["avg_margin"], -x["size"]))

    # Compact component ids (sorted by avg_margin asc)
    ordered_roots = [s["component_root"] for s in summaries]
    root_to_compact = {rid: ci for ci, rid in enumerate(ordered_roots)}

    # Write assignments
    assign_path = os.path.join(out_dir, f"pathB_finch_train.assignments.bottom_{pct:02d}pct.jsonl")
    with open(assign_path, "w", encoding="utf-8") as f:
        for local_j, rec in enumerate(conf):
            root = local_to_root[local_j]
            entry = {
                "i": int(rec["i"]),
                "query": rec.get("query", ""),
                "margin_cos": float(rec.get("margin_cos", 0.0)),
                "component_root": int(root),
                "component_id": int(root_to_compact[root]),
                "concept": summaries[root_to_compact[root]]["concept"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write Cinter (components + one concept per cluster)
    summary_path = os.path.join(out_dir, f"pathB_finch_train.components.bottom_{pct:02d}pct.json")
    cinter_concepts = [s["concept"] for s in summaries if s["concept"]]
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "pct": pct,
            "confused_queries": int(N),
            "clustering": "FINCH partition-1 (Sarfraz et al., 2019)",
            "device": dev,
            "train_query_cache": train_query_cache,
            "patha_out_dir": patha_out_dir,
            "num_clusters": int(len(summaries)),
            "num_cinter_concepts": int(len(cinter_concepts)),
            "note": (
                "FINCH: each point connects to its 1-NN (undirected). "
                "Connected components = clusters. "
                "Cinter = most frequent concept per cluster from PathA per-query concepts."
            ),
            "components_sorted_by_avg_margin_asc": [
                {"component_id": int(root_to_compact[s["component_root"]]), **s}
                for s in summaries
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {assign_path}")
    print(f"[Saved] {summary_path}  (Cinter: {len(cinter_concepts)} concepts from {len(summaries)} clusters)")
    print("[Done] Path B (FINCH) TRAIN complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pct", type=int, default=30, help="bottom percentile (e.g., 5,10,15,20,25,30)")
    ap.add_argument("--dia_out_dir", type=str, default=DEFAULT_DIA_OUT_DIR)
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--train_query_cache", type=str, default=DEFAULT_TRAIN_QUERY_CACHE)
    ap.add_argument("--patha_out_dir", type=str, default=DEFAULT_PATHA_OUT_DIR,
                    help="PathA output dir containing per_query_neg_concepts.bottom_XXpct.json")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    main(
        pct=args.pct,
        dia_out_dir=args.dia_out_dir,
        out_dir=args.out_dir,
        train_query_cache=args.train_query_cache,
        patha_out_dir=args.patha_out_dir,
        device=dev,
    )
