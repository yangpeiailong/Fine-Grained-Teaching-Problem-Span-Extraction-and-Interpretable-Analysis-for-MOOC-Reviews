import os
import pandas as pd
import random

# ===================== Path Configuration (Relative Paths) =====================
# Current script directory: ./codes
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input directory: ../raw_datas
INPUT_DIR = os.path.join(BASE_DIR, "..", "raw_datas")

# Output directory: ../datas_for_annotation
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "datas_for_annotation")

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===================== Load all Excel files and extract course names =====================
all_reviews = []

# Iterate all files in input directory
for fname in os.listdir(INPUT_DIR):
    if fname.endswith("_reviews.xlsx"):
        # Automatically extract course name from filename
        course_name = fname.replace("_reviews.xlsx", "").strip()
        file_path = os.path.join(INPUT_DIR, fname)

        # Read Excel file
        df = pd.read_excel(file_path)

        # Keep only necessary columns if they exist
        if "review_content" in df.columns and "rating" in df.columns:
            df = df[["review_content", "rating", "course_term", "user_nickname", "review_time"]].copy()
            # Add course name as a new column
            df["course_name"] = course_name
            all_reviews.append(df)

# Merge all data into one DataFrame
total_df = pd.concat(all_reviews, ignore_index=True)

# ===================== Sampling Strategy =====================
# Split reviews by rating
low_rating = total_df[total_df["rating"] <= 4].copy()
high_rating = total_df[total_df["rating"] == 5].copy()

# Calculate sampling numbers
n_low = len(low_rating)
n_high = min(len(high_rating), int(n_low * 0.7))

# Sample high-rating reviews
high_sampled = high_rating.sample(n=n_high, random_state=42)

# Combine low-rated and sampled high-rated data
final_df = pd.concat([low_rating, high_sampled], ignore_index=True)

# Remove duplicate reviews based on content
final_df = final_df.drop_duplicates(subset=["review_content"])

# Shuffle the final dataset
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ===================== Save Output =====================
output_file = os.path.join(OUTPUT_DIR, "selected_reviews_for_annotation.xlsx")
final_df.to_excel(output_file, index=False)

# ===================== Print Summary =====================
print("=" * 60)
print("Processing completed! Course names have been automatically added ✅")
print(f"Reviews with rating ≤ 4: {len(low_rating)}")
print(f"Sampled 5-star reviews: {len(high_sampled)}")
print(f"Total records after deduplication: {len(final_df)}")
print(f"Output saved to: {output_file}")
print("=" * 60)