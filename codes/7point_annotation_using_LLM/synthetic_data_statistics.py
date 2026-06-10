import os
import pandas as pd
import numpy as np

# ===================== Path Configuration (Relative) =====================
# Current script directory: codes/7point_annotation_using_LLM
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Root project directory
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")

# Input annotated data paths
REAL_PATH = os.path.join(ROOT_DIR, "datas_after_annotation", "soft_annotated_13labels_deepseek-reasoner.xlsx")
SYN_PATH = os.path.join(ROOT_DIR, "datas_after_annotation", "soft_annotated_13labels_deepseek-reasoner_synthetic_dense.xlsx")

# Output directory for distribution analysis
OUT_FOLDER = os.path.join(ROOT_DIR, "data_distribution_analysis")
os.makedirs(OUT_FOLDER, exist_ok=True)

# ===================== Label Definitions =====================
# English labels for academic paper
label_eng = [
    "Content difficulty mismatch",
    "Excessively fast course pace",
    "Inappropriate explanation methods",
    "Homework and testing issues",
    "Errors in courseware and materials",
    "Abnormal learning progress statistics",
    "Platform playback abnormalities",
    "Platform function defects",
    "Poor online teaching effect",
    "Course content quality issues",
    "Unfair assessment and scoring",
    "Poor overall learning experience",
    "Expectations for teaching improvement"
]

# Corresponding Chinese column names in dataset
label_cn = [
    "内容难度不适",
    "课程节奏过快",
    "讲解方式不佳",
    "作业与测试问题",
    "课件资料错误",
    "学习进度异常",
    "平台播放问题",
    "平台功能缺陷",
    "线上教学效果差",
    "课程内容问题",
    "考核评分不公",
    "学习体验糟糕",
    "希望改进教学"
]

# ===================== Load Real and Synthetic Datasets =====================
df_real = pd.read_excel(REAL_PATH)
df_syn = pd.read_excel(SYN_PATH)

# ===================== Statistical Function =====================
def calculate_distribution(df):
    mean_values = []
    positive_ratios = []
    for col in label_cn:
        valid_vals = df[col].dropna()
        mean_val = valid_vals.mean()
        positive_ratio = (valid_vals > 0.01).mean()
        mean_values.append(round(mean_val, 3))
        positive_ratios.append(round(positive_ratio, 3))
    return mean_values, positive_ratios

# Calculate statistics for both datasets
real_mean, real_pos = calculate_distribution(df_real)
syn_mean, syn_pos = calculate_distribution(df_syn)

# ===================== Build Comparison Table =====================
comparison_table = pd.DataFrame({
    "Problem Category": label_eng,
    "Real_Mean": real_mean,
    "Syn_Mean": syn_mean,
    "Real_Positive_Ratio": real_pos,
    "Syn_Positive_Ratio": syn_pos
})

# ===================== Save Result =====================
output_path = os.path.join(OUT_FOLDER, "synthetic_vs_real_distribution.xlsx")
comparison_table.to_excel(output_path, index=False)

# ===================== Log Output =====================
print("✅ Analysis completed! File saved to:")
print(output_path)
print("\n=== Distribution Comparison (Ready for Paper) ===")
print(comparison_table.round(3))