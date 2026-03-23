'''
{
  "concept_group": "Perovskite solar cell stability",
  "query": "Thermal stability analysis of MAPbI3 devices",
  "pos_document": "In this work, we demonstrate that MAPbI3 degrades at 85C...",
  "pos_doi": "10.1038/nature12345",
  "hard_negatives": [
    "Abstract B: We study the effect of humidity on MAPbI3...",
    "Abstract C: Interface engineering improves thermal stability...",
    "Abstract D: Comparative study of MAPbI3 and FAPbI3..."
  ],
  "hard_negative_scores": [
    0.8812,
    0.8543,
    0.8105
  ],
  "hard_negative_dois": [
    "10.1021/acs.energylett...",
    "10.1002/adma...",
    "10.1039/..."
  ]
}
export CUDA_VISIBLE_DEVICES=0,1

sr 2 48 --qos=q-high-yerim.oh python 08.sort_negative.py \
  --input_jsonl "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1/06.elsevier_iter1.jsonl" \
  --output_dir "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/08.embeddings_cache" \
  --batch_size 516 \
  --num_gpus 2
  BASE_MODEL_ID, BASE_MODEL_ID 바꿔라
  
""


sr 1 48 python 08.sort_negative.py \
  --input_jsonl "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/train_percentile_v1/06.elsevier_sfilted_trec-covid_title_words_cde.jsonl" \
  --output_dir "/gallery_millet/yerim.oh/MatRetriever/01.train/ver3.SPA/01.Diaonisis/trec-covid_title_cde_embeddings_cache" \
  --batch_size 516 \
  --num_gpus 1
  
  
  
https://gemini.google.com/app/ea17a5f32ebbe5a1?is_sa=1&is_sa=1&android-min-version=301356232&ios-min-version=322.0&campaign_id=bkws&utm_source=sem&utm_medium=paid-media&utm_campaign=bkws&pt=9008&mt=8&ct=p-growth-sem-bkws&gclsrc=aw.ds&gad_source=1&gad_campaignid=21109724830&gbraid=0AAAAApk5BhnHgYWw1VOe3-hd-k8WsIsQ9&gclid=Cj0KCQiA1czLBhDhARIsAIEc7uhj_mGzdJUSHofkOJg2q-XxEQhKw_1hvI_eYuhz7H7KZSjTlqSws94aAtIeEALw_wcB

'''
import os
import json
import argparse
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# ==========================================
# Config
# ==========================================
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B"
ADAPTER_PATH = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/checkpoints/BEG_BMR2_FULL/checkpoint-step-100"

TEMPLATES = {
    "passage": "Represent this passage.\nPassage: {text} {eos}",
    "query_mat_paper": "Given a query, retrieve passages that are relevant to the query.\nQuery: {text} {eos}"
}

def load_model(rank):
    # 로딩 메시지는 tqdm을 깨뜨릴 수 있으므로 제거하거나 최소화
    device = f"cuda:{rank}"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModel.from_pretrained(BASE_MODEL_ID, dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(device)
    model.eval()
    return model, tokenizer

def last_token_pooling(last_hidden_states, attention_mask):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    reps = last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    return reps

def compute_embeddings(rank, texts, template_key, output_file, batch_size=4096):
    """GPU 하나가 할당된 텍스트 리스트를 처리하고 저장"""
    device = f"cuda:{rank}"
    
    # 모델 로딩
    model, tokenizer = load_model(rank)
    
    template = TEMPLATES[template_key]
    formatted_texts = [template.format(text=t, eos=tokenizer.eos_token) for t in texts]
    
    embeddings_map = {} 
    total = len(texts)
    
    # [수정 포인트] desc에 현재 작업 내용 표시 + position으로 줄 나누기
    # template_key가 'passage'면 "Abstracts", 아니면 "Titles"라고 표시
    task_name = "Abstracts" if template_key == "passage" else "Titles   "
    desc = f"[GPU {rank}] {task_name}"
    
    # position=rank: 0번 GPU는 0번째 줄, 1번 GPU는 1번째 줄...
    # leave=True: 완료되어도 사라지지 않음
    pbar = tqdm(total=total, desc=desc, position=rank, leave=True, dynamic_ncols=True)

    for i in range(0, total, batch_size):
        batch_text_raw = texts[i : i+batch_size]
        batch_text_fmt = formatted_texts[i : i+batch_size]
        
        inputs = tokenizer(batch_text_fmt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            reps = last_token_pooling(outputs.last_hidden_state, inputs.attention_mask)
            reps = F.normalize(reps, p=2, dim=1) 
            
        reps_cpu = reps.float().cpu()
        for j, text in enumerate(batch_text_raw):
            embeddings_map[text] = reps_cpu[j]
            
        pbar.update(len(batch_text_raw))
            
    pbar.close()
    torch.save(embeddings_map, output_file)

def worker_main(rank, world_size, title_list, abstract_list, args):
    # 1. Titles 처리
    my_titles = title_list[rank::world_size]
    compute_embeddings(rank, my_titles, "query_mat_paper", f"{args.output_dir}/titles_part{rank}.pt", args.batch_size)
    
    # 2. Abstracts 처리 (Titles가 끝나면 같은 줄에서 이어서 시작됨)
    my_abstracts = abstract_list[rank::world_size]
    compute_embeddings(rank, my_abstracts, "passage", f"{args.output_dir}/abstracts_part{rank}.pt", args.batch_size)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_gpus", type=int, default=2)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Scan (CPU)
    print("[Main] Scanning for unique Titles and Abstracts...")
    unique_titles = set()
    unique_abstracts = set()
    
    with open(args.input_jsonl, 'r') as f:
        for line in tqdm(f, desc="Scanning JSONL"):
            try:
                data = json.loads(line)
                for p in data.get("papers", []):
                    t = p.get("title", "").strip()
                    a = p.get("abstract", "").strip()
                    if t: unique_titles.add(t)
                    if a and len(a) > 20: unique_abstracts.add(a)
            except: continue
            
    title_list = sorted(list(unique_titles))
    abstract_list = sorted(list(unique_abstracts))
    
    print(f"\n[Main] Found {len(title_list)} unique Titles")
    print(f"[Main] Found {len(abstract_list)} unique Abstracts")
    
    # [중요] GPU들이 줄을 차지할 공간을 미리 확보해줌 (화면 깨짐 방지)
    print("\n" * (args.num_gpus + 1))
    
    # 멀티프로세싱 시작
    mp.spawn(worker_main, args=(args.num_gpus, title_list, abstract_list, args), nprocs=args.num_gpus, join=True)
    
    # Merge
    # GPU 작업들이 끝난 후 줄바꿈을 해서 Merge 로그가 겹치지 않게 함
    print("\n" * (args.num_gpus + 2)) 
    print("[Main] Merging embedding files...")
    
    final_titles = {}
    final_abstracts = {}
    
    for r in tqdm(range(args.num_gpus), desc="Merging"):
        t_part = torch.load(f"{args.output_dir}/titles_part{r}.pt")
        a_part = torch.load(f"{args.output_dir}/abstracts_part{r}.pt")
        final_titles.update(t_part)
        final_abstracts.update(a_part)
        
        os.remove(f"{args.output_dir}/titles_part{r}.pt")
        os.remove(f"{args.output_dir}/abstracts_part{r}.pt")
        
    torch.save(final_titles, f"{args.output_dir}/all_title_embeddings.pt")
    torch.save(final_abstracts, f"{args.output_dir}/all_abstract_embeddings.pt")
    print("[Main] Done. All embeddings saved.")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()