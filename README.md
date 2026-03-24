# REPAIR: Resolving Long-Tail Confusion in Scientific Retrievers via Fact-Verified Iterative Refinement

<p align="center">
  <a href="https://arxiv.org/abs/2513.REPAIR"><img src="https://img.shields.io/badge/arXiv-REPAIR-b31b1b.svg" alt="arXiv"></a>
  <a href="https://github.com/yerimoh/REPAIR"><img src="https://img.shields.io/github/stars/yerimoh/REPAIR?style=social" alt="GitHub Stars"></a>
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p align="center">
  <b>ACL 2025 Submission</b>
</p>

---

## Overview

**REPAIR** (**RE**triever via E**p**istemic **A**PI-Guided Ite**r**ative Refinement) is a self-evolving data augmentation framework for scientific dense retrievers. It iteratively synthesizes training data to address *long-tail concept confusion* and *high fact-sensitivity* — two structural properties that fundamentally distinguish scientific literature from general-domain text.

### Key Ideas

| Challenge | REPAIR's Solution |
|-----------|-------------------|
| **Long-tailed concept distribution** | Diagnose retrieval failures via margin analysis; identify distractor concepts with CCS scoring and FINCH clustering |
| **Hallucination-prone augmentation** | Ground every generated query-document pair through external API verification (Semantic Scholar, PubChem, Materials Project) |
| **Fine-grained factual distinctions** | Resolve confusions with fact-contrastive hard negatives selected by the current retriever |

> **REPAIR iterates three stages:** (1) **Diagnosis** identifies long-tail distractor concepts that the retriever confuses; (2) **Expansion** grounds concepts in externally verified evidence from scientific APIs; (3) **Differentiation** resolves fine-grained factual distinctions via hard negative mining.

---

## Results

### Standard Scientific IR (nDCG@10)

| Model | Scale | #Pairs | NFCorpus | SciFact | SciDocs | TREC-COVID | AVG | BIOSSES |
|-------|-------|--------|----------|---------|---------|------------|-----|---------|
| BMRETRIEVER-410M | 410M | 11.4M | 0.321 | 0.711 | 0.167 | 0.831 | 0.508 | 0.840 |
| **REPAIR-500M** | **500M** | **4M** | **0.376** | 0.680 | **0.196** | 0.812 | **0.516** | **0.853** |
| BMRETRIEVER-2B | 2B | 10M | 0.351 | 0.760 | 0.199 | 0.863 | 0.543 | 0.828 |
| **REPAIR-1.5B** | **1.5B** | **4M** | 0.376 | 0.757 | 0.201 | 0.853 | **0.546** | **0.849** |
| BMRETRIEVER-7B | 7B | 11.4M | 0.364 | 0.778 | 0.201 | 0.861 | 0.551 | 0.847 |
| **REPAIR-7B** | **7B** | **4M** | **0.413** | **0.789** | **0.227** | 0.842 | **0.568** | 0.846 |

REPAIR outperforms same-scale baselines using **3× less training data**.

### Retrieval-Oriented Applications (Materials & Biomedical, nDCG@20)

| Model | iCliniq R@5 | iCliniq R@20 | ChemLit-mat | ChemLit-biomed | MeSH MRR@5 | RELISH nDCG |
|-------|-------------|--------------|-------------|----------------|-----------|-------------|
| BMRETRIEVER-410M | 60.6 | 72.8 | 64.0 | 64.0 | 39.8 | 91.2 |
| **REPAIR-500M** | **61.3** | **74.1** | **64.4** | **65.2** | **42.8** | **91.5** |
| BMRETRIEVER-2B | 70.0 | 81.2 | 61.4 | 72.6 | 59.5 | 91.5 |
| **REPAIR-1.5B** | 65.0 | **80.8** | **68.3** | **72.9** | 51.7 | 90.9 |
| BMRETRIEVER-7B | 68.4 | 79.7 | 67.2 | 71.1 | 61.1 | 92.2 |
| **REPAIR-7B** | **70.1** | **81.3** | **71.5** | **73.5** | **62.0** | **92.8** |

---

## Installation

```bash
git clone https://github.com/yerimoh/REPAIR.git
cd REPAIR
pip install -r requirements.txt
```

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0 with CUDA ≥ 11.8
  *(Single GPU required — DataParallel is not supported due to DynamicCache)*
- Hugging Face `transformers`, `peft`
- `faiss-gpu` for efficient nearest-neighbor search

```bash
pip install torch>=2.0 transformers peft faiss-gpu tqdm requests
```

---

## Repository Structure

```
REPAIR/
├── diagnosis/                          # Stage I — Diagnosis
│   ├── 01_diagnosis_margin.py          #   Cosine-margin computation → Qconf
│   ├── 02_pathA_ccs_regex.py           #   Intra-query CCS (regex entity extraction) → Cintra
│   ├── 02_pathA_ccs_matdetector.py     #   Intra-query CCS (MatDetector extraction) → Cintra
│   ├── 03_pathB_finch.py               #   Inter-query FINCH clustering → Cinter
│   ├── 04_merge_cconf.py               #   Cconf = Cintra ∪ Cinter
│   └── utils_matdetector.py            #   MatDetector entity extraction utilities
│
├── expansion/                          # Stage II — Expansion
│   ├── 05_query_generation.py          #   LLM-based confusing query generation
│   ├── 06_semantic_scholar.py          #   Semantic Scholar API retrieval & verification
│   └── 06_elsevier.py                  #   Elsevier API retrieval & verification
│
├── differentiation/                    # Stage III — Differentiation
│   ├── 07_embed_candidates.py          #   Embed expansion candidate documents
│   ├── 08_hard_negative_selection.py   #   Model-aware hard negative selection (Eq. 7)
│   └── 09_consistency_filter.py        #   Consistency filter (Eq. 6) + single d⁻
│
├── eval/                               # Evaluation scripts
│   ├── eval_mesh.py
│   └── test_data/
│
├── run_repair.sh                       # End-to-end pipeline runner
└── requirements.txt
```

---

## Methodology

### Retriever Formulation

REPAIR uses a shared decoder-only LM backbone $M_\theta$ (Qwen2.5 at 500M / 1.5B / 7B) with EOS pooling:

$$e_{q/d} = \text{Pool}_{\text{EOS}}\!\left(M_\theta\!\left(q/d \oplus [\text{EOS}]\right)\right)$$

Relevance is scored by dot product $s_\theta(q, d) = e_q^\top e_d$, and the retriever returns $\text{TopK}_\theta(q) \subset \mathcal{D}$ under $s_\theta$.

---

### Stage I — Diagnosis

**Goal:** Identify the long-tail scientific concepts responsible for retrieval confusion.

#### Step 1 — Low-Margin Query Selection

For each training query $q$ with ground-truth positive $d^+$, compute the confusion margin:

$$\Delta_\theta(q) = s_\theta(q, d^+) - \max_{d \neq d^+} s_\theta(q, d)$$

Queries in the bottom percentile of $\Delta_\theta$ form the confusion set $\mathcal{Q}_\text{conf}$.

```bash
python diagnosis/01_diagnosis_margin.py
# → confused_train_queries.bottom_30pct.cos.jsonl
```

#### Step 2 — Intra-Query Distractor Mining (PathA → Cintra)

Aggregate over the confusion set:
$$D^+ = \{d^+(q)\}_{q \in \mathcal{Q}_\text{conf}}, \qquad D^- = \bigcup_{q \in \mathcal{Q}_\text{conf}} D^-(q)$$

Score each extracted concept $e \in \mathcal{E}$ with the **Confusing Concept Score (CCS)** [Eq. 5]:

$$\text{CCS}(e) = \frac{df(e;\, D^-)}{df(e;\, D^+) + \varepsilon}$$

where $df(e; \cdot)$ is the *document frequency* of $e$ in the set, and $\varepsilon > 0$ prevents division by zero. A high CCS identifies concepts that frequently distract the retriever (appear in $D^-$) but are rare in ground-truth positives ($D^+$).

**Cintra** is constructed by selecting the **highest-CCS concept per query** from its $D^-(q)$.

```bash
python diagnosis/02_pathA_ccs_regex.py \
    --pct 30 \
    --dia_out_dir <diagnosis_output_dir> \
    --train_pt    <train_data.pt> \
    --out_dir     <patha_output_dir> \
    --top_neg_n   30
# → cintra.bottom_30pct.jsonl
# → per_query_neg_concepts.bottom_30pct.json  (used by PathB)
```

#### Step 3 — Inter-Query Distractor Mining (PathB → Cinter)

Cluster $\mathcal{Q}_\text{conf}$ using the **FINCH algorithm** (Sarfraz et al., 2019):

1. For each query $i$, find its **first nearest neighbor** $\text{nn}(i) = \arg\max_{j \neq i} \cos(e_i, e_j)$
2. Build an undirected graph with edges $i \leftrightarrow \text{nn}(i)$
3. Connected components of the graph = FINCH clusters

This approach is **parameter-free** — no cluster count $k$ is needed. From each resulting cluster, aggregate the candidate concepts extracted from member queries and select the **most frequent one** → **Cinter**.

```bash
python diagnosis/03_pathB_finch.py \
    --pct               30 \
    --dia_out_dir       <diagnosis_output_dir> \
    --out_dir           <pathb_output_dir> \
    --train_query_cache <query_cache.normfp16.pt> \
    --patha_out_dir     <patha_output_dir> \
    --device            cuda
# → pathB_finch_train.components.bottom_30pct.json  (Cinter)
```

#### Step 4 — Distractor Concept Set

$$\mathcal{C}_\text{conf} = \mathcal{C}_\text{intra} \cup \mathcal{C}_\text{inter}$$

```bash
python diagnosis/04_merge_cconf.py --pct 30
# → final_terms.bottom_30pct.txt   (Cconf)
# → final_terms.bottom_30pct.json
```

---

### Stage II — Expansion

**Goal:** Ground each distractor concept $c \in \mathcal{C}_\text{conf}$ in externally fact-verified evidence, yielding validated tuples $(q_\text{new}, d^+, D^-_\text{cand})$.

1. **Concept Grounding**: Retrieve physical/chemical attributes (synonyms, molecular weight, crystal structure, etc.) from **PubChem** and **Materials Project** for each $c$.
2. **Confusing Query Generation**: Sample 1–3 grounded concepts and prompt $M_\theta$ to generate a candidate query $q_\text{model}$ that it finds inherently ambiguous.
3. **Verification via External APIs**: Query Semantic Scholar / Elsevier with $q_\text{model}$. Discard if no results are returned. For valid searches:
   - Top-matching document title → **verified query** $q_\text{new}$
   - Top-matching document content → **positive** $d^+$
   - Remaining highly similar documents → **negative candidates** $D^-_\text{cand}$

Generation continues until the number of valid tuples reaches $2 \times |\mathcal{T}_0|$.

```bash
# LLM-based query generation from Cconf
python expansion/05_query_generation.py

# Semantic Scholar API retrieval & verification
python expansion/06_semantic_scholar.py \
    --concept_txt <final_terms.bottom_30pct.txt> \
    --key_file    <semantic_api_keys.txt> \
    --out_jsonl   <expansion.jsonl> \
    --target_k    20 \
    --resume
```

---

### Stage III — Differentiation

**Goal:** Construct training triplets $(q_\text{new}, d^+, d^-)$ using a **single model-aware hard negative**.

#### Consistency Filtering [Eq. 6]

Define a local candidate pool $\mathcal{D}_\text{pool} = \{d^+\} \cup D^-_\text{cand}$. Retain a tuple only if the current retriever ranks $d^+$ in the top $\kappa$ positions of $\mathcal{D}_\text{pool}$:

$$\mathbf{1}_\text{keep}(q_\text{new}) = \mathbf{1}\!\left[\text{rank}\!\left(d^+ \mid q_\text{new};\, \mathcal{D}_\text{pool}\right) \leq \kappa\right]$$

This ensures queries are answerable before hard negative mining proceeds.

#### Single Hard Negative Selection [Eq. 7]

Select the hardest negative from the API-verified candidates:

$$d^- = \arg\max_{d \in D^-_\text{cand}} s_\theta(q_\text{new}, d)$$

Using a single hard negative yields a more semantically meaningful decision boundary than multiple easy negatives.

```bash
# Embed expansion candidates
python differentiation/07_embed_candidates.py \
    --input_jsonl <expansion.jsonl> \
    --output_dir  <embed_cache/> \
    --num_gpus    1

# Model-aware hard negative selection
python differentiation/08_hard_negative_selection.py \
    --input_jsonl   <expansion.jsonl> \
    --embedding_dir <embed_cache/> \
    --output_jsonl  <hard_negatives.jsonl> \
    --top_k         15

# Consistency filter + single d⁻ selection
python differentiation/09_consistency_filter.py \
    --input_jsonl   <hard_negatives.jsonl> \
    --embedding_dir <embed_cache/> \
    --output_jsonl  <final_train_data.jsonl> \
    --top_k         15 \
    --consistency_k 5
```

---

### Iterative Contrastive Optimization

The retriever is updated via InfoNCE contrastive loss:

$$\mathcal{L}(q_\text{new}) = -\log \frac{\exp\!\left(s_\theta(q_\text{new}, d^+)/\tau\right)}{\displaystyle\sum_{d \in \{d^+\} \cup \mathcal{N}} \exp\!\left(s_\theta(q_\text{new}, d)/\tau\right)}$$

After each update, the margin landscape $\{\Delta_\theta(q)\}$ is recomputed and Stages I–III are repeated. REPAIR uses **2 iterations** by default; performance continues to improve beyond that, but with diminishing returns.

---

## End-to-End Pipeline

Run the full Diagnosis → Expansion → Differentiation pipeline:

```bash
# Default (PCT=30, single GPU)
sr 1 48 --exclude=hockney --qos=q-high-yerim.oh bash run_repair.sh

# Adjust bottom-percentile threshold
PCT=20 bash run_repair.sh
```

### Pipeline Steps

| Step | Script | Stage | Output |
|------|--------|-------|--------|
| 1 | `01_diagnosis_margin.py` | I — Diagnosis | `confused_train_queries.bottom_30pct.cos.jsonl` |
| 2 | `02_pathA_ccs_*.py` | I — Diagnosis | `cintra.bottom_30pct.jsonl`, `per_query_neg_concepts.json` |
| 3 | `03_pathB_finch.py` | I — Diagnosis | `pathB_finch_train.components.bottom_30pct.json` |
| 4 | `04_merge_cconf.py` | I — Diagnosis | `final_terms.bottom_30pct.txt` (Cconf) |
| 5 | `05_query_generation.py` | II — Expansion | LLM-generated candidate queries |
| 6 | `06_semantic_scholar.py` | II — Expansion | `expansion.bottom_30pct.jsonl` |
| 7 | `07_embed_candidates.py` | III — Differentiation | `embed_cache/` |
| 8 | `08_hard_negative_selection.py` | III — Differentiation | `hard_negatives.bottom_30pct.jsonl` |
| 9 | `09_consistency_filter.py` | III — Differentiation | `final_train_data.bottom_30pct.jsonl` |

### Key Hyperparameters

| Variable | Default | Description |
|----------|---------|-------------|
| `PCT` | `30` | Bottom-percentile for $\mathcal{Q}_\text{conf}$ selection |
| `TOP_NEG_N` | `30` | Top-$N$ retrieved negatives per query for CCS |
| `TARGET_K` | `20` | Target papers per concept from APIs |
| `TOP_K` | `15` | Hard negative candidate pool size |
| `CONSISTENCY_K` | `5` | Consistency filter threshold $\kappa$ |

---

## Evaluated Benchmarks

| Benchmark | Task | Domain | Metric |
|-----------|------|--------|--------|
| NFCorpus | Information Retrieval | Biomedical | nDCG@10 |
| SciFact | Information Retrieval | Biomedical | nDCG@10 |
| SciDocs | Information Retrieval | Scientific | nDCG@10 |
| TREC-COVID | Information Retrieval | Biomedical | nDCG@10 |
| BIOSSES | Sentence Similarity | Biomedical | Spearman's ρ |
| iCliniq | Question Answering | Clinical | R@5, R@20, nDCG@20 |
| ChemLit-QA (mat) | Question Answering | Materials Science | R@5, R@20, nDCG@20 |
| ChemLit-QA (biomed) | Question Answering | Biomedical | R@5, R@20, nDCG@20 |
| MeSH | Entity Linking | Biomedical | R@1, R@5, MRR@5 |
| RELISH | Paper Recommendation | Biomedical | MAP, nDCG |

---

## Citation

If you find REPAIR useful in your research, please cite:

```bibtex
TBD
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [FINCH Clustering](https://github.com/ssarfraz/FINCH-Clustering) (Sarfraz et al., 2019) — parameter-free clustering used in inter-query distractor mining
- [Semantic Scholar Open Research Corpus](https://www.semanticscholar.org/product/api) — external evidence retrieval for fact verification
- [PubChem](https://pubchem.ncbi.nlm.nih.gov/) & [Materials Project](https://next-gen.materialsproject.org/) — concept grounding metadata
- [Qwen2.5](https://huggingface.co/Qwen) — LLM backbone for the REPAIR retriever
- [BMRetriever](https://github.com/ritaranx/BMRetriever) (Xu et al., 2024) — baseline model and evaluation framework
