import os
import re
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification

# =========================
# Paths
# =========================
INPUT_PT = "/gallery_millet/yerim.oh/MatRetriever/01.train/01-02.Detector/Ver2_train_final.pt"
OUTPUT_PT = "Ver2_train_final_matdetector.pt"

MODEL_PATH = "yerim0210/MatDetector"
TOKENIZER_PATH = "/gallery_millet/yerim.oh/MatRetriever/01.train/01-02.Detector/matbert-base-cased"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- speed knobs ---
MAX_WORD_LEN = 32          # 128은 단어 단위엔 과함. 보통 16~32로 충분 (속도↑)
BATCH_SIZE = 1024           # GPU 메모리 보고 128~1024 사이로 조절
USE_FP16 = True            # GPU면 True 추천

# ✅ 모델카드 기준 라벨 순서(클래스 0~4)
ID2TAG = {
    0: "O",
    1: "B-matname",
    2: "I-matname",
    3: "B-mf",
    4: "I-mf",
}

SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}
DOI_URL_RE = re.compile(r"https?\s*:\s*/\s*/\s*\S+")

def normalize(x):
    x = "" if x is None else str(x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def iter_entries(pt_obj):
    """
    pt 구조가:
      - list[dict] 이거나
      - dict 안에 data/items/records/examples 키로 list가 들어있는 경우
    전부를 'entries' iterable로 반환.
    """
    if isinstance(pt_obj, list):
        return pt_obj
    if isinstance(pt_obj, dict):
        for k in ["data", "items", "records", "examples"]:
            if k in pt_obj and isinstance(pt_obj[k], list):
                return pt_obj[k]
    raise ValueError(f"Unknown pt structure: {type(pt_obj)}")

def extract_words(text: str):
    text = DOI_URL_RE.sub(" ", text)
    # split만 하면 충분히 빠름 (추가 토크나이징 규칙 원하면 여기 수정)
    return [w for w in text.split() if w.strip()]

@torch.no_grad()
def classify_words_batch(words, tokenizer, model):
    """
    words: list[str]
    returns: list[final_tag], list[token_tags(list[str])]
    - final_tag은 word 내 subtoken 태그들의 majority vote
    """
    # padding + truncation batch tokenization
    enc = tokenizer(
        words,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_WORD_LEN,
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # AMP
    if USE_FP16 and DEVICE.type == "cuda":
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

    # argmax per token
    pred_ids = logits.argmax(dim=-1)  # (B, L)

    # tokens -> tags per word
    batch_final = []
    batch_token_tags = []

    # convert ids -> tokens (batch)
    tokens_batch = [tokenizer.convert_ids_to_tokens(row.tolist()) for row in input_ids]

    for toks, labs in zip(tokens_batch, pred_ids):
        tags = []
        for tok, lab in zip(toks, labs.tolist()):
            if tok in SPECIAL_TOKENS:
                continue
            tags.append(ID2TAG.get(int(lab), "O"))

        if not tags:
            batch_final.append("O")
            batch_token_tags.append([])
            continue

        # majority vote
        counts = {}
        for t in tags:
            counts[t] = counts.get(t, 0) + 1
        final = max(counts, key=counts.get)

        batch_final.append(final)
        batch_token_tags.append(tags)

    return batch_final, batch_token_tags

def main():
    # -------------------------
    # Load
    # -------------------------
    pt_obj = torch.load(INPUT_PT, map_location="cpu")
    entries = iter_entries(pt_obj)

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        use_fast=True,          # 가능하면 fast가 훨씬 빠름
        do_lower_case=False
    )
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    print("num_labels =", model.config.num_labels)
    print("model.config.id2label =", model.config.id2label)

    # -------------------------
    # Cache: word -> (final_tag, token_tags)
    # -------------------------
    cache = {}

    out_entries = []
    global_label_hist = {"O": 0, "B-matname": 0, "I-matname": 0, "B-mf": 0, "I-mf": 0}

    # -------------------------
    # Process all entries
    # -------------------------
    for entry in tqdm(entries, desc="Entries", unit="entry"):
        title = normalize(entry.get("paper_Title"))
        abstract = normalize(entry.get("paper_abstract"))
        text = (title + "\n" + abstract).strip()

        words = extract_words(text)
        if not words:
            # empty safe
            entry_out = dict(entry)
            entry_out["mat_concepts"] = {"matname": [], "mf": []}
            entry_out["mat_concepts_all"] = []
            entry_out["mat_concepts_debug"] = {"label_hist": {"O": 0, "B-matname": 0, "I-matname": 0, "B-mf": 0, "I-mf": 0}}
            out_entries.append(entry_out)
            continue

        # --- local label hist ---
        label_hist = {"O": 0, "B-matname": 0, "I-matname": 0, "B-mf": 0, "I-mf": 0}
        matname, mf = [], []

        # 1) 캐시에 없는 단어만 모아서 배치 분류
        missing = [w for w in words if w not in cache]
        if missing:
            # 배치 단위로 쪼개기
            n_batches = math.ceil(len(missing) / BATCH_SIZE)
            for b in range(n_batches):
                chunk = missing[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
                finals, token_tags_list = classify_words_batch(chunk, tokenizer, model)
                for w, ftag, ttags in zip(chunk, finals, token_tags_list):
                    cache[w] = (ftag, ttags)

        # 2) 단어 순회하며 결과 누적 (캐시에서 즉시 조회)
        for w in words:
            final_tag, _token_tags = cache[w]

            if final_tag in label_hist:
                label_hist[final_tag] += 1
                global_label_hist[final_tag] += 1

            if final_tag in ("B-matname", "I-matname"):
                matname.append(w)
            elif final_tag in ("B-mf", "I-mf"):
                mf.append(w)

        # de-dup (순서 유지)
        matname = list(dict.fromkeys(matname))
        mf = list(dict.fromkeys(mf))

        entry_out = dict(entry)
        entry_out["mat_concepts"] = {"matname": matname, "mf": mf}
        entry_out["mat_concepts_all"] = matname + mf
        entry_out["mat_concepts_debug"] = {"label_hist": label_hist}

        out_entries.append(entry_out)

    # -------------------------
    # Save
    # -------------------------
    torch.save(out_entries, OUTPUT_PT)

    print("\n======= GLOBAL LABEL HISTOGRAM (word-level) =======")
    print(global_label_hist)
    print("\nSaved to:", OUTPUT_PT)
    print("Cache size (unique words):", len(cache))

if __name__ == "__main__":
    main()
