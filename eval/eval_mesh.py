import os
import csv
import json
import logging
from typing import List
from collections import defaultdict

import torch
from torch import Tensor
from tqdm import trange
from transformers import AutoTokenizer, AutoModel
from transformers.file_utils import PaddingStrategy

import pytrec_eval

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── instruction format from paper (Appendix B, Table 10) ──────────────────────
QUERY_INSTRUCTION = "Given a concept, retrieve passages for its definition."
PASSAGE_PREFIX    = "Represent this passage"


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden.shape[0]
    return last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]


class BMRetrieverEncoder:
    def __init__(self, model_path: str):
        import os as _os, json as _json
        logger.info(f"Loading model: {model_path}")

        adapter_cfg_path = _os.path.join(model_path, "adapter_config.json")
        if _os.path.exists(adapter_cfg_path):
            from peft import PeftModel
            with open(adapter_cfg_path) as f:
                adapter_cfg = _json.load(f)
            base_model_name = adapter_cfg["base_model_name_or_path"]
            logger.info(f"LoRA adapter detected. Base model: {base_model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            base_model = AutoModel.from_pretrained(base_model_name, torch_dtype=torch.float32)
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

        # NOTE: DataParallel breaks with DynamicCache on multi-GPU — use 1 GPU only
        self.model.cuda().eval()
        self.max_length = 512

    @torch.no_grad()
    def encode(self, texts: List[str], batch_size: int = 128) -> Tensor:
        all_embeddings = []
        for i in trange(0, len(texts), batch_size, desc="Encoding"):
            batch = texts[i: i + batch_size]
            batch_dict = self.tokenizer(
                batch,
                max_length=self.max_length - 1,
                return_attention_mask=False,
                return_token_type_ids=False,
                padding=PaddingStrategy.DO_NOT_PAD,
                truncation=True,
            )
            batch_dict['input_ids'] = [
                ids + [self.tokenizer.eos_token_id]
                for ids in batch_dict['input_ids']
            ]
            batch_dict = self.tokenizer.pad(
                batch_dict,
                padding=True,
                return_attention_mask=True,
                return_tensors='pt',
            ).to("cuda")

            with torch.amp.autocast("cuda"):
                outputs = self.model(**batch_dict)
            emb = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_queries(self, queries: List[str], batch_size: int = 64) -> Tensor:
        formatted = [f"{QUERY_INSTRUCTION}\nQuery: {q}" for q in queries]
        if formatted:
            logger.info(f"Query example: {formatted[0][:120]}")
        return self.encode(formatted, batch_size=batch_size)

    def encode_corpus(self, texts: List[str], batch_size: int = 128) -> Tensor:
        formatted = [f"{PASSAGE_PREFIX}\npassage: {t}" for t in texts]
        if formatted:
            logger.info(f"Passage example: {formatted[0][:120]}")
        return self.encode(formatted, batch_size=batch_size)


def load_mesh(data_path: str):
    corpus = {}
    with open(os.path.join(data_path, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            corpus[obj["_id"]] = {
                "title": obj.get("title", ""),
                "text":  obj.get("text", ""),
            }

    queries = {}
    with open(os.path.join(data_path, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"]

    qrels = defaultdict(dict)
    tsv_path = os.path.join(data_path, "qrels", "test.tsv")
    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            if len(row) >= 3:
                qrels[row[0]][row[1]] = int(row[2])

    logger.info(f"Corpus: {len(corpus):,}  Queries: {len(queries):,}  Qrels: {len(qrels):,}")
    return corpus, queries, dict(qrels)


def compute_mrr_at_k(results, qrels, k=5):
    """Compute MRR@k manually."""
    rr_sum = 0.0
    n = 0
    for qid, scored_docs in results.items():
        if qid not in qrels:
            continue
        relevant = set(qrels[qid].keys())
        # sort by score descending, take top-k
        ranked = sorted(scored_docs.items(), key=lambda x: x[1], reverse=True)[:k]
        for rank, (did, _) in enumerate(ranked, start=1):
            if did in relevant:
                rr_sum += 1.0 / rank
                break
        n += 1
    return rr_sum / n if n > 0 else 0.0


def compute_metrics(qrels, results, k_values):
    metrics_str = set()
    for k in k_values:
        metrics_str.add(f"recall.{k}")

    qrels_int   = {qid: {did: int(s)   for did, s in rels.items()} for qid, rels in qrels.items()}
    results_flt = {qid: {did: float(s) for did, s in docs.items()} for qid, docs in results.items()}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_int, metrics_str)
    per_q = evaluator.evaluate(results_flt)

    agg = {}
    for k in k_values:
        vals = [per_q[qid].get(f"recall_{k}", 0.0) for qid in per_q]
        agg[f"recall_{k}"] = sum(vals) / len(vals) if vals else 0.0

    return agg


def main(
    model_path:   str = "BMRetriever/BMRetriever-410M",
    data_path:    str = "/gallery_millet/yerim.oh/MatRetriever/02.test/test_data/MeSH/mesh_ir",
    corpus_batch: int = 128,
    query_batch:  int = 64,
    top_k:        int = 10,
):
    corpus, queries, qrels = load_mesh(data_path)
    encoder = BMRetrieverEncoder(model_path)

    # ── encode corpus (definition text only, not prepending title) ─────────────
    corpus_ids   = list(corpus.keys())
    corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
    logger.info(f"Encoding {len(corpus_texts):,} corpus passages...")
    corpus_emb = encoder.encode_corpus(corpus_texts, batch_size=corpus_batch)
    logger.info(f"Corpus embeddings: {corpus_emb.shape}")

    # ── encode queries (concept names) ─────────────────────────────────────────
    query_ids   = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    logger.info(f"Encoding {len(query_texts):,} queries...")
    query_emb = encoder.encode_queries(query_texts, batch_size=query_batch)
    logger.info(f"Query embeddings: {query_emb.shape}")

    # ── dot product retrieval ──────────────────────────────────────────────────
    logger.info("Computing dot product scores...")
    corpus_emb = corpus_emb.cuda()
    query_emb  = query_emb.cuda()

    results = {}
    batch_size = 512
    for qi in trange(0, len(query_ids), batch_size, desc="Retrieving"):
        q_batch = query_emb[qi: qi + batch_size]           # [B, D]
        scores  = q_batch @ corpus_emb.T                   # [B, N]
        k = min(top_k, len(corpus_ids))
        topk_scores, topk_idx = torch.topk(scores, k, largest=True)
        for bi, qid in enumerate(query_ids[qi: qi + batch_size]):
            results[qid] = {
                corpus_ids[int(idx)]: float(score)
                for idx, score in zip(topk_idx[bi].cpu().tolist(), topk_scores[bi].cpu().tolist())
            }

    del corpus_emb, query_emb
    torch.cuda.empty_cache()

    # ── evaluate ───────────────────────────────────────────────────────────────
    agg     = compute_metrics(qrels, results, k_values=[1, 5])
    mrr_at5 = compute_mrr_at_k(results, qrels, k=5)

    logger.info("\n========== MeSH Results ==========")
    logger.info(f"Paper target (410M): R@1=31.5  R@5=53.8  MRR@5=39.8")
    logger.info(f"Corpus size: {len(corpus):,} (paper: 29,600)")
    logger.info(f"R@1    : {agg['recall_1']*100:.1f}")
    logger.info(f"R@5    : {agg['recall_5']*100:.1f}")
    logger.info(f"MRR@5  : {mrr_at5*100:.1f}")
    logger.info("===================================")

    out_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"mesh_{model_path.replace('/', '-')}_metrics.txt"
    )
    with open(out_file, "w") as fw:
        fw.write("==== MeSH Entity Linking Metrics ====\n")
        fw.write(f"Model: {model_path}\n")
        fw.write(f"Corpus size: {len(corpus):,} (paper: 29,600)\n")
        fw.write(f"Paper target (410M): R@1=31.5  R@5=53.8  MRR@5=39.8\n\n")
        fw.write("Key metrics (x100):\n")
        fw.write(f"  R@1    : {agg['recall_1']*100:.2f}\n")
        fw.write(f"  R@5    : {agg['recall_5']*100:.2f}\n")
        fw.write(f"  MRR@5  : {mrr_at5*100:.2f}\n")
    logger.info(f"Saved to {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--data_path",    default="test_data/MeSH/mesh_ir")
    parser.add_argument("--corpus_batch", type=int, default=128)
    parser.add_argument("--query_batch",  type=int, default=64)
    parser.add_argument("--top_k",        type=int, default=10)
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.corpus_batch, args.query_batch, args.top_k)
