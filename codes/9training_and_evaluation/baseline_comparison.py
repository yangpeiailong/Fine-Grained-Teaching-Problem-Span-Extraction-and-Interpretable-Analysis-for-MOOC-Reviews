import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import f1_score
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

# ===================== Global Configuration =====================
USE_SYNTHETIC = False
FREEZE_BERT = True
USE_HARD_LABELS = True
FOLD_NUM = 5
MAX_LEN = 128
BATCH_SIZE = 32
LR = 5e-5
EPOCHS = 40
NUM_LABELS = 13

LABEL_NAMES = [
    "内容难度不适", "课程节奏过快", "讲解方式不佳", "作业与测试问题",
    "课件资料错误", "学习进度异常", "平台播放问题", "平台功能缺陷",
    "线上教学效果差", "课程内容问题", "考核评分不公", "学习体验糟糕", "希望改进教学"
]

# ===================== Path Configuration (Relative) =====================
# Current script directory: codes/9training_and_evaluation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")

DATA_PATH = os.path.join(ROOT_DIR, "cv_datasets")
SAVE_ROOT = os.path.join(ROOT_DIR, "results_baseline")
os.makedirs(SAVE_ROOT, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use online HuggingFace model (no local path required)
BERT_PATH = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

# ===================== Save File Suffix =====================
def get_save_suffix():
    suffix = ""
    suffix += "_synth" if USE_SYNTHETIC else "_nosynth"
    suffix += "_freeze" if FREEZE_BERT else "_nofreeze"
    suffix += "_hard" if USE_HARD_LABELS else "_soft"
    return suffix

SAVE_SUFFIX = get_save_suffix()

# ===================== Loss Functions =====================
class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.pos_weight = pos_weight

    def forward(self, pred, true):
        loss = self.mse(pred, true)
        weight = torch.where(true > 0.1, self.pos_weight, 1.0)
        return (loss * weight).mean()

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=15.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.bce = None

    def forward(self, pred, true):
        weight = torch.tensor([self.pos_weight], device=true.device)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=weight)
        return self.bce(pred, true)

class JointLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5
        self.beta = 0.5
        self.span_loss = nn.BCELoss()

    def forward(self, cls_pred, cls_true, start_pred, start_true, end_pred, end_true):
        if USE_HARD_LABELS:
            weight = torch.tensor([15.0], device=cls_true.device)
            cls_loss = nn.BCEWithLogitsLoss(pos_weight=weight)(cls_pred, cls_true)
        else:
            cls_loss = WeightedMSELoss()(cls_pred, cls_true)

        loss = self.alpha * cls_loss
        loss += self.beta * self.span_loss(start_pred, start_true)
        loss += self.beta * self.span_loss(end_pred, end_true)
        return loss

# ===================== Evaluation Metrics =====================
def compute_metrics(preds, trues):
    if USE_HARD_LABELS:
        preds = torch.sigmoid(torch.tensor(preds)).numpy()

    if USE_HARD_LABELS:
        t_bin = (trues >= 0.5).astype(int).ravel()
        p_bin = (preds >= 0.5).astype(int).ravel()
    else:
        t_bin = (trues >= 0.2).astype(int).ravel()
        p_bin = (preds >= 0.2).astype(int).ravel()

    f1 = f1_score(t_bin, p_bin, zero_division=0)
    return {"F1_0.2": round(f1, 4)}

def compute_span_metrics(pred_spans, true_spans):
    tp, fp, fn = 0, 0, 0
    for p, t in zip(pred_spans, true_spans):
        tp += len(set(p) & set(t))
        fp += len(set(p) - set(t))
        fn += len(set(t) - set(p))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"Span_F1": round(f1, 4)}

# ===================== Dataset =====================
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
                if s < MAX_LEN: start[c, s] = 1
                if e < MAX_LEN: end[c, e] = 1
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

def load_data(data_path, train_file, test_file):
    train_df = pd.read_excel(os.path.join(data_path, train_file))
    test_df = pd.read_excel(os.path.join(data_path, test_file))
    return {"train": train_df, "val": test_df}

def get_dataloaders(data, batch_size):
    return {
        "train": DataLoader(MOOCDataset(data["train"]), batch_size=batch_size, shuffle=True),
        "val": DataLoader(MOOCDataset(data["val"]), batch_size=batch_size)
    }

# ===================== Model Definitions =====================
class BertCLS_Span(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls = nn.Linear(768, NUM_LABELS)
        self.start = nn.Linear(768, NUM_LABELS)
        self.end = nn.Linear(768, NUM_LABELS)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_out = self.cls(seq[:, 0, :])
        if not USE_HARD_LABELS:
            cls_out = torch.sigmoid(cls_out)
        start = torch.sigmoid(self.start(seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(seq)).permute(0, 2, 1)
        return cls_out, start, end

class BertAvg_Span(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls = nn.Linear(768, NUM_LABELS)
        self.start = nn.Linear(768, NUM_LABELS)
        self.end = nn.Linear(768, NUM_LABELS)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_out = self.cls(seq.mean(1))
        if not USE_HARD_LABELS:
            cls_out = torch.sigmoid(cls_out)
        start = torch.sigmoid(self.start(seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(seq)).permute(0, 2, 1)
        return cls_out, start, end

class BertAtt_Span(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.att = nn.MultiheadAttention(768, 1, batch_first=True)
        self.cls = nn.Linear(768, NUM_LABELS)
        self.start = nn.Linear(768, NUM_LABELS)
        self.end = nn.Linear(768, NUM_LABELS)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        att_out, _ = self.att(seq, seq, seq)
        cls_out = self.cls(att_out.mean(1))
        if not USE_HARD_LABELS:
            cls_out = torch.sigmoid(cls_out)
        start = torch.sigmoid(self.start(seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(seq)).permute(0, 2, 1)
        return cls_out, start, end

class BertLSTM_Span(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(256, NUM_LABELS)
        self.start = nn.Linear(768, NUM_LABELS)
        self.end = nn.Linear(768, NUM_LABELS)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        o, (h, _) = self.lstm(seq)
        cls_out = self.cls(torch.cat([h[-2], h[-1]], dim=1))
        if not USE_HARD_LABELS:
            cls_out = torch.sigmoid(cls_out)
        start = torch.sigmoid(self.start(seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(seq)).permute(0, 2, 1)
        return cls_out, start, end

class OurJointGuided(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.cls_fuse = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Linear(768, NUM_LABELS)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(NUM_LABELS, 768),
            nn.LayerNorm(768),
            nn.Sigmoid()
        )
        self.start_head = nn.Linear(768, NUM_LABELS)
        self.end_head = nn.Linear(768, NUM_LABELS)
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_token = seq[:, 0, :]
        avg_pool = seq.mean(1)
        cls_out = self.cls_fuse(torch.cat([cls_token, avg_pool], dim=-1))
        if not USE_HARD_LABELS:
            cls_out = torch.sigmoid(cls_out)

        cls_prob = torch.sigmoid(cls_out)
        gate_weight = self.gate_net(cls_prob).unsqueeze(1)
        guided_seq = self.norm(seq * gate_weight + seq)

        m = mask.unsqueeze(-1).bool()
        start = torch.sigmoid(self.start_head(guided_seq).masked_fill(~m, -1e4)).permute(0, 2, 1)
        end = torch.sigmoid(self.end_head(guided_seq).masked_fill(~m, -1e4)).permute(0, 2, 1)
        return cls_out, start, end

class BertCLS_Guided(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls = nn.Linear(768, NUM_LABELS)
        self.start = nn.Linear(768, NUM_LABELS)
        self.end = nn.Linear(768, NUM_LABELS)
        self.gate = nn.Sequential(nn.Linear(NUM_LABELS, 768), nn.Sigmoid())
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_out = self.cls(seq[:, 0, :])
        cls_prob = torch.sigmoid(cls_out)
        g = self.gate(cls_prob).unsqueeze(1)
        guided = self.norm(seq * g + seq)
        if not USE_HARD_LABELS:
            cls_out = cls_prob
        start = torch.sigmoid(self.start(guided)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(guided)).permute(0, 2, 1)
        return cls_out, start, end

class BertAvg_Guided(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls = nn.Linear(768, NUM_LABELS)
        self.start = nn.Linear(768, NUM_LABELS)
        self.end = nn.Linear(768, NUM_LABELS)
        self.gate = nn.Sequential(nn.Linear(NUM_LABELS, 768), nn.Sigmoid())
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_out = self.cls(seq.mean(1))
        cls_prob = torch.sigmoid(cls_out)
        g = self.gate(cls_prob).unsqueeze(1)
        guided = self.norm(seq * g + seq)
        if not USE_HARD_LABELS:
            cls_out = cls_prob
        start = torch.sigmoid(self.start(guided)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(guided)).permute(0, 2, 1)
        return cls_out, start, end

class BertAtt_Guided(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.att = nn.MultiheadAttention(768, 1, batch_first=True)
        self.cls = nn.Linear(768, NUM_LABELS)
        self.start = nn.Linear(768, NUM_LABELS)
        self.end = nn.Linear(768, NUM_LABELS)
        self.gate = nn.Sequential(nn.Linear(NUM_LABELS, 768), nn.Sigmoid())
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        att_out, _ = self.att(seq, seq, seq)
        cls_out = self.cls(att_out.mean(1))
        cls_prob = torch.sigmoid(cls_out)
        g = self.gate(cls_prob).unsqueeze(1)
        guided = self.norm(seq * g + seq)
        if not USE_HARD_LABELS:
            cls_out = cls_prob
        start = torch.sigmoid(self.start(guided)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(guided)).permute(0, 2, 1)
        return cls_out, start, end

class BertLSTM_Guided(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH)
        if FREEZE_BERT:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(256, NUM_LABELS)
        self.start = nn.Linear(768, NUM_LABELS)
        self.end = nn.Linear(768, NUM_LABELS)
        self.gate = nn.Sequential(nn.Linear(NUM_LABELS, 768), nn.Sigmoid())
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        o, (h, _) = self.lstm(seq)
        cls_out = self.cls(torch.cat([h[-2], h[-1]], dim=1))
        cls_prob = torch.sigmoid(cls_out)
        g = self.gate(cls_prob).unsqueeze(1)
        guided = self.norm(seq * g + seq)
        if not USE_HARD_LABELS:
            cls_out = cls_prob
        start = torch.sigmoid(self.start(guided)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(guided)).permute(0, 2, 1)
        return cls_out, start, end

MODEL_CONFIGS = {
    "bert_cls_span": BertCLS_Span,
    "bert_avg_span": BertAvg_Span,
    "bert_att_span": BertAtt_Span,
    "bert_lstm_span": BertLSTM_Span,
    "bert_cls_guided": BertCLS_Guided,
    "bert_avg_guided": BertAvg_Guided,
    "bert_att_guided": BertAtt_Guided,
    "bert_lstm_guided": BertLSTM_Guided,
}

# ===================== Training =====================
def train_one_fold(fold):
    suffix = "_with_synth" if USE_SYNTHETIC else ""
    train_file = f"train_fold{fold}{suffix}.xlsx"
    test_file = f"test_fold{fold}{suffix}.xlsx"
    fold_res = []
    for name, model_cls in MODEL_CONFIGS.items():
        print(f"\n========== Fold {fold} | {name} ==========")
        model = model_cls().to(device)
        data = load_data(DATA_PATH, train_file, test_file)
        loaders = get_dataloaders(data, BATCH_SIZE)
        opt = optim.Adam(model.parameters(), lr=LR)
        cri = JointLoss()
        best = {}

        for ep in range(EPOCHS):
            model.train()
            total_loss = 0
            pbar = tqdm(loaders['train'], desc=f"Ep{ep + 1} Train")
            for b in pbar:
                x, m, y = b["input_ids"].to(device), b["mask"].to(device), b["label"].to(device)
                st, et = b["start"].to(device), b["end"].to(device)

                if USE_HARD_LABELS:
                    y = (y >= 0.2).float()

                cls_pred, s_pred, e_pred = model(x, m)
                loss = cri(cls_pred, y, s_pred, st, e_pred, et)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})

            model.eval()
            ps, ts, spans = [], [], []
            with torch.no_grad():
                for b in loaders['val']:
                    x, m, y = b["input_ids"].to(device), b["mask"].to(device), b["label"].to(device)
                    st, et = b["start"].to(device), b["end"].to(device)
                    c, s, e = model(x, m)
                    ps.append(c.cpu().numpy())
                    ts.append(y.cpu().numpy())
                    for i in range(c.shape[0]):
                        pred, true = [], []
                        for ci in range(NUM_LABELS):
                            s_p = torch.where(s[i, ci] > 0.4)[0].cpu().numpy()
                            e_p = torch.where(e[i, ci] > 0.4)[0].cpu().numpy()
                            for si in s_p:
                                ei = e_p[e_p > si]
                                if len(ei) > 0:
                                    pred.append((si, ei[0], ci))
                            s_t = torch.where(st[i, ci] == 1)[0].cpu().numpy()
                            e_t = torch.where(et[i, ci] == 1)[0].cpu().numpy()
                            for si in s_t:
                                ei = e_t[e_t > si]
                                if len(ei) > 0:
                                    true.append((si, ei[0], ci))
                        spans.append((pred, true))

            ps = np.concatenate(ps)
            ts = np.concatenate(ts)
            pred_spans = [p for p, _ in spans]
            true_spans = [t for _, t in spans]
            met = compute_metrics(ps, ts)
            span_met = compute_span_metrics(pred_spans, true_spans)
            best = {**met, **span_met}
            print(f"[Val] Ep{ep + 1} | ClsF1:{met['F1_0.2']:.4f} | SpanF1:{span_met['Span_F1']:.4f}")

        fold_res.append({"fold": fold, "model": name, **best})
    return fold_res

def run():
    all_res = []
    for f in range(FOLD_NUM):
        all_res += train_one_fold(f)
    df = pd.DataFrame(all_res)
    avg = df.groupby("model").mean().sort_values("F1_0.2", ascending=False)

    save_path = os.path.join(SAVE_ROOT, f"baseline_result{SAVE_SUFFIX}.xlsx")
    avg.to_excel(save_path)
    print(f"\nResults saved to: {save_path}")
    print("\n==== Final Baseline Results ====")
    print(avg[["F1_0.2", "Span_F1"]])

if __name__ == "__main__":
    run()