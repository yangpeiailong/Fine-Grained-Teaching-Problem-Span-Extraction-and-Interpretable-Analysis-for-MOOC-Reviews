import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

# ===================== Model & Training Config =====================
MODEL_CHOICE = "bert_avg_guided"
USE_SYNTHETIC = True
FREEZE_BERT = False
USE_HARD_LABELS = False
FOLD_FOR_FULL = 2

# Fixed Hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
LR = 5e-5
EPOCHS = 40
NUM_LABELS = 13

# Use online HuggingFace model (no local path)
BERT_PATH = "bert-base-chinese"

LABEL_NAMES = [
    "内容难度不适", "课程节奏过快", "讲解方式不佳", "作业与测试问题",
    "课件资料错误", "学习进度异常", "平台播放问题", "平台功能缺陷",
    "线上教学效果差", "课程内容问题", "考核评分不公", "学习体验糟糕", "希望改进教学"
]

# ===================== Relative Path Configuration =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")

DATA_PATH = os.path.join(ROOT_DIR, "cv_datasets")
SAVE_DIR = os.path.join(ROOT_DIR, "trained_models")
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# ===================== Dataset Definition =====================
class MOOCDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row.review_content)
        label = row[LABEL_NAMES].values.astype(np.float32)
        start = np.zeros((NUM_LABELS, MAX_LEN), np.float32)
        end = np.zeros((NUM_LABELS, MAX_LEN), np.float32)
        try:
            spans = json.loads(row["span_annotations(char_level)"])["spans"]
            name2idx = {n: i for i, n in enumerate(LABEL_NAMES)}
            for sp in spans:
                c = name2idx[sp["problem_type"]]
                s, e = sp["start_char_idx"], sp["end_char_idx"]
                if s < MAX_LEN:
                    start[c, s] = 1
                if e < MAX_LEN:
                    end[c, e] = 1
        except:
            pass

        enc = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LEN)
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "mask": torch.tensor(enc["attention_mask"]),
            "label": torch.tensor(label),
            "start": torch.tensor(start),
            "end": torch.tensor(end)
        }

# ===================== Joint Loss Function =====================
class JointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5
        self.beta = 0.5
        self.span_loss = nn.BCELoss()

    def forward(self, cls_pred, cls_true, start_pred, start_true, end_pred, end_true):
        if USE_HARD_LABELS:
            cls_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.], device=device))(cls_pred, cls_true)
        else:
            cls_loss = nn.MSELoss()(cls_pred, cls_true)

        loss = self.alpha * cls_loss
        loss += self.beta * self.span_loss(start_pred, start_true)
        loss += self.beta * self.span_loss(end_pred, end_true)
        return loss

# ===================== Import Models from models.py =====================
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

# ===================== Load Full Dataset =====================
def load_full_data_from_fold():
    suffix = "_with_synth" if USE_SYNTHETIC else ""
    train_file = f"train_fold{FOLD_FOR_FULL}{suffix}.xlsx"
    test_file = f"test_fold{FOLD_FOR_FULL}{suffix}.xlsx"

    train_df = pd.read_excel(os.path.join(DATA_PATH, train_file))
    test_df = pd.read_excel(os.path.join(DATA_PATH, test_file))
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"✅ Full dataset loaded: {len(full_df)} samples")
    return full_df

# ===================== Training Pipeline =====================
if __name__ == "__main__":
    full_df = load_full_data_from_fold()
    loader = DataLoader(MOOCDataset(full_df), batch_size=BATCH_SIZE, shuffle=True)

    model = build_model().to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    cri = JointLoss()
    save_path = os.path.join(SAVE_DIR, f"{MODEL_CHOICE}_best.pth")
    best_loss = 999

    for ep in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {ep + 1}")

        for b in pbar:
            ids = b["input_ids"].to(device)
            mask = b["mask"].to(device)
            y = b["label"].to(device)
            st = b["start"].to(device)
            et = b["end"].to(device)

            if USE_HARD_LABELS:
                y = (y >= 0.2).float()

            cls_pred, s_pred, e_pred = model(ids, mask)
            if not USE_HARD_LABELS:
                cls_pred = torch.sigmoid(cls_pred)

            loss = cri(cls_pred, y, s_pred, st, e_pred, et)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"EP {ep + 1} Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved: {save_path}")