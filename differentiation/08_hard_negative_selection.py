'''
https://chatgpt.com/c/6974687f-41f8-8322-8a26-2376c2905db0


export CUDA_VISIBLE_DEVICES=0

export CUDA_VISIBLE_DEVICES=0,1

sr 2 48 python 09.make_N.py \
  --input_jsonl "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1/06.elsevier_iter1.jsonl"  \
  --embedding_dir "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/08.embeddings_cache" \
  --output_jsonl "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/07.train_hard_negatives.jsonl" \
  --top_k 15 \
  --num_gpus 2
  
sr 1 48 python /gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/09.make_N.py \
  --input_jsonl "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/06.elsevier_sfilted_trec-covid_title_words_cde.jsonl" \
  --embedding_dir "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/trec-covid_title_cde_embeddings_cache" \
  --output_jsonl "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/07.trec-covid_title_cde_train_hard_negatives.jsonl" \
  --top_k 15 \
  --num_gpus 1

'''
import os
import json
import argparse
import torch
import numpy as np
import torch.multiprocessing as mp
from tqdm import tqdm
import math

# 각 프로세스가 실행할 워커 함수
def worker(rank, gpu_id, lines, title_emb_map, abs_emb_map, output_file, top_k):
    # 각 프로세스별 GPU 설정
    device = torch.device(f"cuda:{gpu_id}")
    
    # CPU 과부하 방지를 위해 쓰레드 제한 (GPU 연산 위주이므로)
    torch.set_num_threads(4)
    
    print(f"[Worker-{rank}] Started on GPU {gpu_id}. Processing {len(lines)} lines.")
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        # 진행률 표시 (첫 번째 워커만 표시하거나, position으로 줄바꿈 처리)
        iterator = tqdm(lines, desc=f"Worker-{rank}", position=rank)
        
        for line in iterator:
            try:
                data = json.loads(line)
            except: continue
            
            concept = data.get("concept", "")
            papers = data.get("papers", [])
            if len(papers) < 3: continue
            
            # 2. Collect Valid Papers
            group_titles = []
            group_abstracts = []
            group_dois = []
            
            for p in papers:
                t = p.get("title", "").strip()
                a = p.get("abstract", "").strip()
                # 임베딩 맵에 존재하는지 확인 (CPU 메모리 조회)
                if t in title_emb_map and a in abs_emb_map:
                    group_titles.append(t)
                    group_abstracts.append(a)
                    group_dois.append(p.get("doi", ""))
            
            if len(group_titles) < 3: continue
            
            # 3. Stack Tensors (여기서 해당 GPU로 올림)
            try:
                # 리스트 컴프리헨션으로 텐서 추출 후 스택
                q_stack = torch.stack([title_emb_map[t] for t in group_titles]).to(device)
                d_stack = torch.stack([abs_emb_map[a] for a in group_abstracts]).to(device)
                
                # 4. Matrix Multiplication
                sim_matrix = torch.mm(q_stack, d_stack.transpose(0, 1))
                
                # 5. Process Results (CPU로 내려서 정렬)
                sim_matrix_np = sim_matrix.float().cpu().numpy()
                
                N = len(group_titles)
                for i in range(N):
                    row_scores = sim_matrix_np[i]
                    row_scores[i] = -1.0 # Mask self
                    
                    if len(row_scores) > top_k:
                        top_k_indices = np.argpartition(row_scores, -top_k)[-top_k:]
                        top_k_indices = top_k_indices[np.argsort(row_scores[top_k_indices])[::-1]]
                    else:
                        top_k_indices = np.argsort(row_scores)[::-1]
                    
                    hn_texts = []
                    hn_scores = []
                    hn_dois = []
                    
                    for idx in top_k_indices:
                        score = float(row_scores[idx])
                        if score == -1.0: continue
                        
                        hn_texts.append(group_abstracts[idx])
                        hn_scores.append(score)
                        hn_dois.append(group_dois[idx])
                    
                    if hn_texts:
                        out_record = {
                            "task": "mat_paper",
                            "concept_group": concept,
                            "query": group_titles[i],
                            "pos_document": group_abstracts[i],
                            "pos_doi": group_dois[i],
                            "hard_negatives": hn_texts,
                            "hard_negative_scores": hn_scores,
                            "hard_negative_dois": hn_dois
                        }
                        fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[Worker-{rank}] Error processing group: {e}")
                continue

    print(f"[Worker-{rank}] Finished.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--num_gpus", type=int, default=2, help="Number of GPUs to use")
    args = parser.parse_args()
    
    # 1. Load Embeddings (메인 프로세스에서 CPU로 로드)
    # 중요: CUDA 초기화 전에 로드해야 함. map_location='cpu' 명시
    print("[Main] Loading pre-computed embeddings into RAM (CPU)...")
    title_emb_path = os.path.join(args.embedding_dir, "all_title_embeddings.pt")
    abs_emb_path = os.path.join(args.embedding_dir, "all_abstract_embeddings.pt")
    
    title_emb_map = torch.load(title_emb_path, map_location="cpu")
    abs_emb_map = torch.load(abs_emb_path, map_location="cpu")
    print(f"[Main] Loaded {len(title_emb_map)} titles and {len(abs_emb_map)} abstracts.")
    
    # 2. Prepare Data Splitting
    with open(args.input_jsonl, 'r', encoding='utf-8') as fin:
        all_lines = fin.readlines()
    
    total_lines = len(all_lines)
    chunk_size = math.ceil(total_lines / args.num_gpus)
    print(f"[Main] Total lines: {total_lines}. Split into {args.num_gpus} chunks of size ~{chunk_size}.")
    
    processes = []
    temp_files = []
    
    # 3. Launch Processes
    # Linux의 기본 start_method인 'fork'를 사용해야 
    # 부모 프로세스의 메모리(임베딩 맵)를 자식 프로세스가 복사 없이(COW) 공유할 수 있음.
    # 단, 부모 프로세스에서 CUDA를 절대 미리 초기화하면 안 됨.
    
    for i in range(args.num_gpus):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_lines)
        lines_chunk = all_lines[start_idx:end_idx]
        
        # 각 워커가 쓸 임시 파일명
        temp_output = f"{args.output_jsonl}.part{i}"
        temp_files.append(temp_output)
        
        # GPU ID 할당 (0, 1 순차 할당)
        gpu_id = i
        
        p = mp.Process(
            target=worker, 
            args=(i, gpu_id, lines_chunk, title_emb_map, abs_emb_map, temp_output, args.top_k)
        )
        p.start()
        processes.append(p)
    
    # 4. Wait for completion
    for p in processes:
        p.join()
        
    print("[Main] All workers finished. Merging files...")
    
    # 5. Merge Output Files
    with open(args.output_jsonl, 'w', encoding='utf-8') as fout:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f_temp:
                    for line in f_temp:
                        fout.write(line)
                os.remove(temp_file) # 병합 후 임시파일 삭제
                
    print(f"[Main] Done. Saved to {args.output_jsonl}")

if __name__ == "__main__":
    # CUDA Multiprocessing 이슈 방지를 위해 Main에서는 fork 사용을 권장 (Linux default)
    # 단, 메인 프로세스에서 CUDA 함수를 호출하지 않도록 주의.
    try:
        mp.set_start_method('fork')
    except RuntimeError:
        pass # 이미 설정되어 있으면 패스
    main()