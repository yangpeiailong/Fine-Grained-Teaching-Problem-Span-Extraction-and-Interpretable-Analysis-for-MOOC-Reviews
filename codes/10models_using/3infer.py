import torch
import numpy as np
from transformers import BertTokenizer
import os

# ===================== Config (Must match training) =====================
MODEL_CHOICE = "bert_avg_guided"
USE_HARD_LABELS = False
FREEZE_BERT = False

# Fixed parameters
NUM_LABELS = 13
MAX_LEN = 128
BERT_PATH = "bert-base-chinese"  # Online HuggingFace model

# Label names
LABEL_NAMES = [
    "内容难度不适", "课程节奏过快", "讲解方式不佳", "作业与测试问题",
    "课件资料错误", "学习进度异常", "平台播放问题", "平台功能缺陷",
    "线上教学效果差", "课程内容问题", "考核评分不公", "学习体验糟糕", "希望改进教学"
]

# ===================== Relative Paths =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")
SAVE_DIR = os.path.join(ROOT_DIR, "trained_models")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# ===================== Model Builder =====================
from models import *

def build_model():
    if MODEL_CHOICE == "bert_cls_span":
        return BertCLS_Span(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    elif MODEL_CHOICE == "bert_avg_span":
        return BertAvg_Span(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    elif MODEL_CHOICE == "bert_att_span":
        return BertAtt_Span(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    elif MODEL_CHOICE == "bert_lstm_span":
        return BertLSTM_Span(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    elif MODEL_CHOICE == "bert_cls_guided":
        return BertCLS_Guided(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    elif MODEL_CHOICE == "bert_avg_guided":
        return BertAvg_Guided(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    elif MODEL_CHOICE == "bert_att_guided":
        return BertAtt_Guided(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    elif MODEL_CHOICE == "bert_lstm_guided":
        return BertLSTM_Guided(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    elif MODEL_CHOICE == "our_joint_guided":
        return OurJointGuided(NUM_LABELS, BERT_PATH, FREEZE_BERT)
    else:
        raise ValueError("Invalid model name!")

# ===================== Load Trained Model =====================
model = build_model().to(device)
model_path = os.path.join(SAVE_DIR, f"{MODEL_CHOICE}_best.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===================== Inference Function =====================
def predict_full(raw_text):
    enc = tokenizer(
        raw_text, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        cls_logits, start_logits, end_logits = model(input_ids, mask)

    # Move to CPU for numpy operations
    cls_scores = torch.sigmoid(cls_logits).squeeze().cpu().numpy()
    start_probs = start_logits.squeeze().cpu().numpy()
    end_probs = end_logits.squeeze().cpu().numpy()

    result = []
    for i, name in enumerate(LABEL_NAMES):
        score = float(cls_scores[i])
        if score < 0.2:
            continue

        s_idx = int(np.argmax(start_probs[i]))
        e_idx = int(np.argmax(end_probs[i]))

        # Clip to valid text range
        s_idx = max(0, min(s_idx, len(raw_text)-1))
        e_idx = max(s_idx, min(e_idx, len(raw_text)-1))
        span_text = raw_text[s_idx: e_idx+1]

        result.append({
            "问题类型": name,
            "置信度": round(score, 3),
            "位置": f"{s_idx} → {e_idx}",
            "抽取片段": span_text
        })
    return result

# ===================== Demo =====================
if __name__ == "__main__":
    print("="*60)
    print("✅ Model loaded successfully")
    print("="*60)

    bad_reviews = [
        "老师讲课速度太快，跟不上，课件还有多处错误",
        "视频经常卡顿，作业难度大且没有讲解",
        "课程内容模糊，听不懂，学习体验很差"
    ]

    good_reviews = [
        "课程讲解清晰，内容安排合理，非常好",
        "老师讲得细致，资料完整，收获很大",
        "平台流畅，教学质量高，非常推荐"
    ]

    print("\n========== Negative Reviews ==========")
    for review in bad_reviews:
        print(f"\n📝 Comment: {review}")
        res = predict_full(review)
        for item in res:
            print(f"   🔴 {item['问题类型']} ({item['置信度']})")
            print(f"        Position: {item['位置']}")
            print(f"        Span: {item['抽取片段']}")

    print("\n========== Positive Reviews ==========")
    for review in good_reviews:
        print(f"\n📝 Comment: {review}")
        res = predict_full(review)
        if not res:
            print("   ✅ No teaching problems detected")