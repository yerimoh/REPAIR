'''
sr 1 48 --qos=q-high-yerim.oh python 12.BGE_filter.py \
    --input_jsonl "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/07.train_hard_negatives.jsonl" \
    --embedding_dir "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/08.embeddings_cache" \
    --output_jsonl "./mined_output.jsonl" \
    --top_k 15 \
    --consistency_k 5 \
    --batch_size 128
'''
import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import gc # 가비지 컬렉터 추가

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True, help="Original training data")
    parser.add_argument("--embedding_dir", required=True, help="Directory with .pt embedding files")
    parser.add_argument("--output_jsonl", required=True, help="Filtered & Mined output")
    parser.add_argument("--top_k", type=int, default=15, help="Number of hard negatives to retrieve")
    parser.add_argument("--consistency_k", type=int, default=5, help="Filter threshold (Keep if rank <= k)")
    parser.add_argument("--batch_size", type=int, default=128, help="Query batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")

    # 1. Load Embeddings
    abs_emb_path = os.path.join(args.embedding_dir, "all_abstract_embeddings.pt")
    title_emb_path = os.path.join(args.embedding_dir, "all_title_embeddings.pt")
    
    print(f"[Load] Loading Abstract Pool from {abs_emb_path} ...")
    abs_emb_map = torch.load(abs_emb_path, map_location="cpu")
    all_abs_texts = list(abs_emb_map.keys())
    abs_text_to_idx = {text: i for i, text in enumerate(all_abs_texts)}
    
    print("[Load] Stacking Abstract Embeddings to GPU...")
    all_abs_tensors = torch.stack(tuple(abs_emb_map.values())).to(device)
    
    # 딕셔너리 명시적 삭제 및 메모리 확보
    del abs_emb_map
    gc.collect() 

    print(f"[Load] Loading Title Embeddings from {title_emb_path} ...")
    title_emb_map = torch.load(title_emb_path, map_location="cpu")

    # 2. Process Data
    fout = open(args.output_jsonl, 'w', encoding='utf-8')
    
    batch_queries = []
    batch_metadata = []
    
    processed_count = 0
    total_kept_count = 0 
    
    # 파일 라인 수를 먼저 세기 (tqdm 진행률 표시용)
    print("[Start] Counting lines for tqdm...")
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"[Start] Processing {total_lines} lines (k={args.consistency_k})...")
    
    with open(args.input_jsonl, 'r', encoding='utf-8') as fin:
        for line in tqdm(fin, total=total_lines, desc="Filtering & Mining"):
            try:
                data = json.loads(line)
            except: 
                continue
            
            # --- JSON 구조에 맞게 수정된 파싱 부분 ---
            task = data.get("task", "mat_paper")
            concept = data.get("concept_group", "")
            title = data.get("query", "").strip()            # query 가 title
            abstract = data.get("pos_document", "").strip()  # pos_document 가 abstract
            doi = data.get("pos_doi", "")
            # ----------------------------------------
            
            if title not in title_emb_map or abstract not in abs_text_to_idx:
                continue
                
            q_emb = title_emb_map[title]
            gt_idx = abs_text_to_idx[abstract]
            
            batch_queries.append(q_emb)
            batch_metadata.append({
                "task": task,
                "concept": concept,
                "title": title,
                "doi": doi,
                "gt_idx": gt_idx,
                "gt_abstract": abstract
            })
            
            if len(batch_queries) >= args.batch_size:
                kept = process_batch(batch_queries, batch_metadata, all_abs_tensors, all_abs_texts, 
                                     fout, args.top_k, args.consistency_k, device)
                processed_count += len(batch_queries)
                total_kept_count += kept
                batch_queries = []
                batch_metadata = []

    # 남은 배치 처리
    if len(batch_queries) > 0:
        kept = process_batch(batch_queries, batch_metadata, all_abs_tensors, all_abs_texts, 
                             fout, args.top_k, args.consistency_k, device)
        processed_count += len(batch_queries)
        total_kept_count += kept

    fout.close()
    
    # 3. Summary
    survival_rate = (total_kept_count / processed_count) * 100 if processed_count > 0 else 0
    print(f"\n[Summary]")
    print(f"Total Processed: {processed_count}")
    print(f"Total Kept:      {total_kept_count}")
    print(f"Total Dropped:   {processed_count - total_kept_count}")
    print(f"Survival Rate:   {survival_rate:.2f}%")
    print(f"[Done] Output saved to {args.output_jsonl}")


def process_batch(queries, metadata, index_tensor, index_texts, fout, top_k, consistency_k, device):
    q_stack = torch.stack(queries).to(device)
    batch_kept = 0
    
    with torch.no_grad():
        scores = torch.mm(q_stack, index_tensor.transpose(0, 1))
        
    search_k = top_k + 1
    top_scores, top_indices = torch.topk(scores, k=search_k, dim=1)
    
    top_scores = top_scores.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    for i in range(len(queries)):
        meta = metadata[i]
        gt_idx = meta['gt_idx']
        
        # Consistency Filter
        rank = -1
        if gt_idx in top_indices[i]:
            rank = np.where(top_indices[i] == gt_idx)[0][0] + 1
        
        if rank == -1 or rank > consistency_k:
            continue
            
        # Hard Negative Mining
        hard_negatives = []
        hn_scores = []
        
        for j, idx in enumerate(top_indices[i]):
            if idx == gt_idx:
                continue 
            if len(hard_negatives) >= top_k:
                break
            hard_negatives.append(index_texts[idx])
            hn_scores.append(float(top_scores[i][j]))
        
        # --- 출력 포맷도 기존 형태를 유지하도록 매핑 ---
        out_record = {
            "task": meta['task'],
            "concept_group": meta['concept'],
            "query": meta['title'],
            "pos_document": meta['gt_abstract'],
            "pos_doi": meta['doi'],
            "hard_negatives": hard_negatives,
            "hard_negative_scores": hn_scores,
        }
        fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        batch_kept += 1
        
    return batch_kept

if __name__ == "__main__":
    main()