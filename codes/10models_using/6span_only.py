import os
import pandas as pd
from collections import Counter
import jieba

# Automatic relative path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

INPUT_DIR = os.path.join(project_root, "predicted_results")
OUTPUT_SPAN = os.path.join(project_root, "mid_results", "output_qualitative_spans")
os.makedirs(OUTPUT_SPAN, exist_ok=True)

# Label mappings in Chinese and English
CHN_COLS = [
    "内容难度不适", "课程节奏过快", "讲解方式不佳", "作业与测试问题",
    "课件资料错误", "学习进度异常", "平台播放问题", "平台功能缺陷",
    "线上教学效果差", "课程内容问题", "考核评分不公", "学习体验糟糕", "希望改进教学"
]

ENG_COLS = [
    "Content difficulty mismatch", "Excessively fast pace", "Inappropriate explanation methods",
    "Homework and testing issues", "Errors in courseware and materials", "Abnormal learning progress",
    "Platform playback abnormalities", "Platform function defects", "Poor online teaching effect",
    "Course content quality issues", "Unfair assessment and scoring", "Poor overall learning experience",
    "Expectations for teaching improvement"
]

# Course name mapping
COURSE_EN_MAP = {
    "人工智能原理": "Principles of AI",
    "人工智能导论": "Introduction to AI",
    "人工智能导论2": "Introduction to AI II",
    "人工智能：模型与算法": "AI Models & Algorithms"
}

SPAN_COLS = [f"{c}_片段" for c in CHN_COLS]
STOPWORDS = {"的", "了", "是", "在", "我", "你", "这", "那", "就", "都", "也", "有", "不", "很", "太", "有点"}

def clean_span(s):
    """Clean and validate extracted text spans"""
    if not isinstance(s, str):
        return None
    s = s.strip()
    return s if 4 <= len(s) <= 40 else None

def main():
    """Extract representative spans and high-frequency words from predicted results"""
    course_dict = {}
    all_spans = []

    for fname in os.listdir(INPUT_DIR):
        if not fname.endswith(".xlsx"):
            continue
        df = pd.read_excel(os.path.join(INPUT_DIR, fname))
        c_cn = fname.replace("_reviews_predictions.xlsx", "")
        c_en = COURSE_EN_MAP.get(c_cn, c_cn)

        if c_en not in course_dict:
            course_dict[c_en] = []

        for idx, span_col in enumerate(SPAN_COLS):
            raw_spans = [clean_span(s) for s in df[span_col].dropna() if clean_span(s)]
            unique_spans = list(dict.fromkeys(raw_spans))[:5]
            course_dict[c_en].extend(unique_spans)
            all_spans.extend(unique_spans)

    # Aggregate all spans for each course into one row
    table_rows = []
    for course_name, examples_list in course_dict.items():
        if not examples_list:
            continue
        example_str = "; ".join([f"{i+1}: {txt}" for i, txt in enumerate(examples_list)])
        table_rows.append({
            "Course": course_name,
            "Typical Examples": example_str
        })

    final_df = pd.DataFrame(table_rows)
    final_df.to_excel(os.path.join(OUTPUT_SPAN, "Course Typical Examples.xlsx"), index=False)

    # Extract high-frequency words from all valid spans
    word_counter = Counter()
    for text in all_spans:
        for word in jieba.lcut(text):
            if len(word) >= 2 and word not in STOPWORDS:
                word_counter[word] += 1

    pd.DataFrame(word_counter.most_common(50), columns=["Keyword", "Frequency"]).to_excel(
        os.path.join(OUTPUT_SPAN, "Global HighFreq Words.xlsx"), index=False)

    print("✅ Qualitative analysis completed successfully.")

if __name__ == "__main__":
    main()