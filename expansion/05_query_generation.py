#sr 1 48 python 05.p_query_make.py


import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
ADAPTER_DIR = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/checkpoints/BEG_BMR2_FULL/checkpoint-step-100"
FILE_PATH = "/gallery_millet/yerim.oh/MatRetriever/01.train/02.train/0.5/iter2/01.Diagnosis/02_train_percentile_v1/PathAB_terms/final_terms.bottom_30pct.txt"

def run_improved_generation():
    # 1. 용어 추출
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    selected_term = random.choice(terms)

    # 2. 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    # 3. 프롬프트 엔지니어링 (Few-shot 추가로 출력 형식 강제)
    # 모델에게 예시를 보여줘서 딴소리 못하게 막습니다.
    prompt = f"""You are an expert in materials science. Generate only one specific research query for the given term.
Example 1:
Term: Graphene
Query: Investigating the electron mobility and structural integrity of monolayer graphene under high-pressure conditions.

Example 2:
Term: Ti6Al4V
Query: Mechanical fatigue analysis of Ti6Al4V alloys fabricated via selective laser melting for biomedical implants.

Example 3:
Term: {selected_term}
Query:"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=40, # 짧고 강하게 생성 유도
            do_sample=True, 
            temperature=0.6,   # 너무 높으면 헛소리하므로 약간 낮춤
            top_p=0.9,
            repetition_penalty=1.2, # 동일 단어 반복 방지
            pad_token_id=tokenizer.eos_token_id
        )

    # 4. 결과 정제
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # Query: 이후의 내용만 가져오기
    if "Query:" in generated_text:
        final_query = generated_text.split("Query:")[-1].strip()
    else:
        final_query = generated_text.replace(prompt, "").strip()

    print("\n" + "="*50)
    print(f"Target Term: {selected_term}")
    print(f"Generated Research Query: {final_query}")
    print("="*50)

if __name__ == "__main__":
    run_improved_generation()