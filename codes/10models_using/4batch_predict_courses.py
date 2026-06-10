from transformers import BertTokenizer
import sys
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# ===================== Auto Path Setting =====================
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import all models
from models import *

# ===================== Config (Same as Training) =====================
MODEL_CHOICE = "bert_avg_guided"
USE_HARD_LABELS = False
FREEZE_BERT = False
NUM_LABELS = 13
MAX_LEN = 128
BERT_PATH = "bert-base-chinese"  # Online Hugging Face

device = "cuda" if torch.cuda.is_available() else "cpu"

# 13 Labels
LABELS = [
    "内容难度不适", "课程节奏过快", "讲解方式不佳", "作业与测试问题",
    "课件资料错误", "学习进度异常", "平台播放问题", "平台功能缺陷",
    "线上教学效果差", "课程内容问题", "考核评分不公", "学习体验糟糕", "希望改进教学"
]
SPAN_LABELS = [f"{label}_片段" for label in LABELS]

# ===================== Relative Paths =====================
SAVE_DIR = os.path.join(project_root, "trained_models")
INPUT_DIR = os.path.join(project_root, "raw_datas_for_test")
OUTPUT_DIR = os.path.join(project_root, "predicted_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== Load Trained Model =====================
def load_trained_model():
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

    model = build_model().to(device)
    model_path = os.path.join(SAVE_DIR, f"{MODEL_CHOICE}_best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ===================== Global Model & Tokenizer =====================
model = load_trained_model()
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# ===================== Prediction Function =====================
def predict_full(text):
    raw_text = text
    enc = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=MAX_LEN, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        cls_logits, start_logits, end_logits = model(input_ids, mask)

    cls_scores = torch.sigmoid(cls_logits).squeeze().cpu().numpy()
    start_probs = start_logits.squeeze().cpu().numpy()
    end_probs = end_logits.squeeze().cpu().numpy()

    scores = []
    spans = []

    for i in range(NUM_LABELS):
        score = float(cls_scores[i])
        score = max(0.0, min(1.0, score))
        scores.append(round(score, 3))

        s_idx = int(np.argmax(start_probs[i]))
        e_idx = int(np.argmax(end_probs[i]))
        s_idx = max(0, min(s_idx, len(raw_text)-1))
        e_idx = max(s_idx, min(e_idx, len(raw_text)-1))
        span_text = raw_text[s_idx:e_idx+1] if score >= 0.2 else ""
        spans.append(span_text)

    score_dict = dict(zip(LABELS, scores))
    span_dict = dict(zip(SPAN_LABELS, spans))
    return {**score_dict, **span_dict}

# ===================== Batch Prediction =====================
def batch_predict_with_bar(text_list):
    results = []
    for text in tqdm(text_list, desc="评论预测进度", ncols=90, colour="green"):
        results.append(predict_full(str(text)))
    return results

# ===================== Main Batch Processing =====================
if __name__ == "__main__":
    excel_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".xlsx")]

    for fname in tqdm(excel_files, desc="总课程进度", ncols=90, colour="blue"):
        print(f"\n📂 Processing: {fname}")
        df = pd.read_excel(os.path.join(INPUT_DIR, fname))
        pred_list = batch_predict_with_bar(df["review_content"].dropna().tolist())
        df_pred = pd.DataFrame(pred_list)
        df_final = pd.concat([df.reset_index(drop=True), df_pred], axis=1)

        out_name = fname.replace(".xlsx", "_predictions.xlsx")
        out_path = os.path.join(OUTPUT_DIR, out_name)
        df_final.to_excel(out_path, index=False)
        print(f"✅ Saved: {out_path}")

    print("\n🎉🎉🎉 All course predictions completed!")