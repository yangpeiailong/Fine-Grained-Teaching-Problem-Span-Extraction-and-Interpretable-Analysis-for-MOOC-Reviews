import pandas as pd
import os

# ===================== Path Configuration (Relative) =====================
# Current script directory: codes/6concat_real_and_synth
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Root directory for annotated data
ANNOTATION_FOLDER = os.path.join(BASE_DIR, "..", "..", "datas_after_annotation")

# File names
REAL_FILE = "soft_annotated_13labels_deepseek-reasoner.xlsx"
SYNTH_FILE = "soft_annotated_13labels_deepseek-reasoner_synthetic_dense.xlsx"
OUT_FILE = "final_mixed_dataset_70real_30synth.xlsx"

# Full paths
real_path = os.path.join(ANNOTATION_FOLDER, REAL_FILE)
synth_path = os.path.join(ANNOTATION_FOLDER, SYNTH_FILE)
out_path = os.path.join(ANNOTATION_FOLDER, OUT_FILE)

# ===================== Load Datasets =====================
df_real = pd.read_excel(real_path)
df_synth = pd.read_excel(synth_path)

print(f"✅ Real data samples: {len(df_real)}")
print(f"✅ Synthetic dense data samples: {len(df_synth)}")

# ===================== Concatenate Real and Synthetic Data =====================
df_combined = pd.concat([df_real, df_synth], ignore_index=True)

# ===================== Save Merged Dataset =====================
df_combined.to_excel(out_path, index=False)

# ===================== Log Results =====================
print(f"\n🎉 Combination completed! Total samples: {len(df_combined)}")
print(f"✅ Mixed dataset saved to:\n{out_path}")
print(f"\n📊 Data ratio: Real {len(df_real)/len(df_combined):.1%} + Synthetic {len(df_synth)/len(df_combined):.1%}")