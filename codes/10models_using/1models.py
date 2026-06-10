import torch
import torch.nn as nn
from transformers import BertModel

# =============================================================================
# 1. BertCLS_Span
# =============================================================================
class BertCLS_Span(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls = nn.Linear(768, num_labels)
        self.start = nn.Linear(768, num_labels)
        self.end = nn.Linear(768, num_labels)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_out = self.cls(seq[:, 0, :])
        start = torch.sigmoid(self.start(seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(seq)).permute(0, 2, 1)
        return cls_out, start, end

# =============================================================================
# 2. BertAvg_Span
# =============================================================================
class BertAvg_Span(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls = nn.Linear(768, num_labels)
        self.start = nn.Linear(768, num_labels)
        self.end = nn.Linear(768, num_labels)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_out = self.cls(seq.mean(1))
        start = torch.sigmoid(self.start(seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(seq)).permute(0, 2, 1)
        return cls_out, start, end

# =============================================================================
# 3. BertAtt_Span
# =============================================================================
class BertAtt_Span(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.att = nn.MultiheadAttention(768, 1, batch_first=True)
        self.cls = nn.Linear(768, num_labels)
        self.start = nn.Linear(768, num_labels)
        self.end = nn.Linear(768, num_labels)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        att_out, _ = self.att(seq, seq, seq)
        cls_out = self.cls(att_out.mean(1))
        start = torch.sigmoid(self.start(seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(seq)).permute(0, 2, 1)
        return cls_out, start, end

# =============================================================================
# 4. BertLSTM_Span
# =============================================================================
class BertLSTM_Span(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(256, num_labels)
        self.start = nn.Linear(768, num_labels)
        self.end = nn.Linear(768, num_labels)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        o, (h, _) = self.lstm(seq)
        cls_out = self.cls(torch.cat([h[-2], h[-1]], dim=1))
        start = torch.sigmoid(self.start(seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(seq)).permute(0, 2, 1)
        return cls_out, start, end

# =============================================================================
# 5. BertCLS_Guided
# =============================================================================
class BertCLS_Guided(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls = nn.Linear(768, num_labels)
        self.start = nn.Linear(768, num_labels)
        self.end = nn.Linear(768, num_labels)
        self.gate = nn.Sequential(nn.Linear(num_labels, 768), nn.Sigmoid())
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_out = self.cls(seq[:, 0, :])
        prob = torch.sigmoid(cls_out)
        g = self.gate(prob).unsqueeze(1)
        guided = self.norm(seq * g + seq)
        start = torch.sigmoid(self.start(guided)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(guided)).permute(0, 2, 1)
        return cls_out, start, end

# =============================================================================
# 6. BertAvg_Guided  
# =============================================================================
class BertAvg_Guided(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls = nn.Linear(768, num_labels)
        self.start = nn.Linear(768, num_labels)
        self.end = nn.Linear(768, num_labels)
        self.gate = nn.Sequential(nn.Linear(num_labels, 768), nn.Sigmoid())
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_out = self.cls(seq.mean(1))
        prob = torch.sigmoid(cls_out)
        g = self.gate(prob).unsqueeze(1)
        guided = self.norm(seq * g + seq)
        start = torch.sigmoid(self.start(guided)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(guided)).permute(0, 2, 1)
        return cls_out, start, end

# =============================================================================
# 7. BertAtt_Guided
# =============================================================================
class BertAtt_Guided(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.att = nn.MultiheadAttention(768, 1, batch_first=True)
        self.cls = nn.Linear(768, num_labels)
        self.start = nn.Linear(768, num_labels)
        self.end = nn.Linear(768, num_labels)
        self.gate = nn.Sequential(nn.Linear(num_labels, 768), nn.Sigmoid())
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        att_out, _ = self.att(seq, seq, seq)
        cls_out = self.cls(att_out.mean(1))
        prob = torch.sigmoid(cls_out)
        g = self.gate(prob).unsqueeze(1)
        guided = self.norm(seq * g + seq)
        start = torch.sigmoid(self.start(guided)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(guided)).permute(0, 2, 1)
        return cls_out, start, end

# =============================================================================
# 8. BertLSTM_Guided
# =============================================================================
class BertLSTM_Guided(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.cls = nn.Linear(256, num_labels)
        self.start = nn.Linear(768, num_labels)
        self.end = nn.Linear(768, num_labels)
        self.gate = nn.Sequential(nn.Linear(num_labels, 768), nn.Sigmoid())
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        o, (h, _) = self.lstm(seq)
        cls_out = self.cls(torch.cat([h[-2], h[-1]], dim=1))
        prob = torch.sigmoid(cls_out)
        g = self.gate(prob).unsqueeze(1)
        guided = self.norm(seq * g + seq)
        start = torch.sigmoid(self.start(guided)).permute(0, 2, 1)
        end = torch.sigmoid(self.end(guided)).permute(0, 2, 1)
        return cls_out, start, end

# =============================================================================
# 9. OurJointGuided
# =============================================================================
class OurJointGuided(nn.Module):
    def __init__(self, num_labels=13, bert_path="bert-base-chinese", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.cls_fuse = nn.Sequential(
            nn.Linear(768 * 2, 768), nn.LayerNorm(768), nn.GELU(), nn.Linear(768, num_labels)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(num_labels, 768), nn.LayerNorm(768), nn.Sigmoid()
        )
        self.start_head = nn.Linear(768, num_labels)
        self.end_head = nn.Linear(768, num_labels)
        self.norm = nn.LayerNorm(768)

    def forward(self, ids, mask):
        seq = self.bert(ids, mask).last_hidden_state
        cls_token = seq[:, 0, :]
        avg_pool = seq.mean(1)
        cls_out = self.cls_fuse(torch.cat([cls_token, avg_pool], dim=-1))
        prob = torch.sigmoid(cls_out)
        g = self.gate_net(prob).unsqueeze(1)
        guided_seq = self.norm(seq * g + seq)
        start = torch.sigmoid(self.start_head(guided_seq)).permute(0, 2, 1)
        end = torch.sigmoid(self.end_head(guided_seq)).permute(0, 2, 1)
        return cls_out, start, end