import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold

# ===================== Core Configuration =====================
# Whether to include synthetic data in training
# True: full mixed dataset (real + synthetic)
# False: only real data (first 4002 samples)
USE_SYNTHETIC = False

# ===================== Path Configuration (Relative) =====================
# Current script directory: codes/8k_fold_split
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root directory
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")

# Input annotated dataset
INPUT_FILE = os.path.join(
    ROOT_DIR,
    "datas_after_annotation",
    "final_mixed_dataset_70real_30synth_char_span_annotated_full.xlsx"
)

# Output directory for cross-validation datasets
OUTPUT_FOLDER = os.path.join(ROOT_DIR, "cv_datasets")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== 13 Teaching Problem Labels =====================
LABEL_COLS = [
    "内容难度不适", "课程节奏过快", "讲解方式不佳", "作业与测试问题",
    "课件资料错误", "学习进度异常", "平台播放问题", "平台功能缺陷",
    "线上教学效果差", "课程内容问题", "考核评分不公", "学习体验糟糕",
    "希望改进教学"
]

# ===================== Load and Filter Dataset =====================
df = pd.read_excel(INPUT_FILE)
df = df.reset_index(drop=True)

# Keep only real data (first 4002 samples) if synthetic is disabled
if not USE_SYNTHETIC:
    df = df.iloc[:4002].copy()
    print(f"📌 Using real data only (first 4002 samples)")

# Create stratification target: samples with at least one problem
df["has_issue"] = df[LABEL_COLS].max(axis=1) > 0.05
y = df["has_issue"].astype(int)

print(f"📊 Total samples: {len(df)} | Problem samples: {round(y.mean() * 100, 2)}%")

# ===================== 5-Fold Stratified Cross-Validation =====================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(df, y)):
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # File naming based on synthetic usage
    suffix = "_with_synth" if USE_SYNTHETIC else "_without_synth"
    train_path = os.path.join(OUTPUT_FOLDER, f"train_fold{fold}{suffix}.xlsx")
    test_path = os.path.join(OUTPUT_FOLDER, f"test_fold{fold}{suffix}.xlsx")

    # Save full dataset (including span annotations)
    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)

    print(f"\n✅ Fold {fold} split completed")
    print(f"   Train: {len(train_df)} | Problem ratio: {round(train_df['has_issue'].mean() * 100, 2)}%")
    print(f"   Test:  {len(test_df)} | Problem ratio: {round(test_df['has_issue'].mean() * 100, 2)}%")

# ===================== Final Summary =====================
print("\n" + "=" * 80)
print("🎉 5-fold cross-validation datasets generated successfully!")
print(f"📁 Output directory: {OUTPUT_FOLDER}")
if USE_SYNTHETIC:
    print("📌 Dataset type: Real + Synthetic (suffix: _with_synth)")
else:
    print("📌 Dataset type: Real only (suffix: _without_synth)")
print("=" * 80)