import pandas as pd
import os

# ===================== Path Configuration (Relative) =====================
# Current script directory: ./codes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input path: ../datas_for_annotation/selected_reviews_for_annotation.xlsx
INPUT_FILE = os.path.join(BASE_DIR, "..", "datas_for_annotation", "selected_reviews_for_annotation.xlsx")

# Output directory: ../datas_negative
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "datas_negative")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output file path
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "negative_reviews_for_topic.xlsx")

# ===================== Data Loading =====================
# Load the preprocessed review dataset
df_all = pd.read_excel(INPUT_FILE)

# ===================== Core Filtering =====================
# Retain only reviews with rating ≤ 4 (negative and problem-oriented reviews)
df_negative = df_all[df_all["rating"] <= 4].copy()

# ===================== Save Result =====================
# Export filtered data for BERTopic topic modeling
df_negative.to_excel(OUTPUT_FILE, index=False)

# ===================== Log Information =====================
print(f"✅ Extraction completed!")
print(f"Total original records: {len(df_all)}")
print(f"Negative reviews (rating ≤ 4): {len(df_negative)}")
print(f"File saved to: negative_reviews_for_topic.xlsx")