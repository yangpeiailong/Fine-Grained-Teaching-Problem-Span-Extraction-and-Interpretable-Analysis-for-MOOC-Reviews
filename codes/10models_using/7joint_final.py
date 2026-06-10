import os
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# Automatic relative path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

INPUT_DIR = os.path.join(project_root, "predicted_results")
OUTPUT_FOLDER = os.path.join(project_root, "mid_results", "output_joint_final")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.2

# Chinese to English label mapping
CHN_ENG_MAP = {
    "内容难度不适": "Content difficulty mismatch",
    "课程节奏过快": "Excessively fast pace",
    "讲解方式不佳": "Inappropriate explanation methods",
    "作业与测试问题": "Homework and testing issues",
    "课件资料错误": "Errors in courseware and materials",
    "学习进度异常": "Abnormal learning progress",
    "平台播放问题": "Platform playback abnormalities",
    "平台功能缺陷": "Platform function defects",
    "线上教学效果差": "Poor online teaching effect",
    "课程内容问题": "Course content quality issues",
    "考核评分不公": "Unfair assessment and scoring",
    "学习体验糟糕": "Poor overall learning experience",
    "希望改进教学": "Expectations for teaching improvement"
}

# Course name mapping
COURSE_NAME_MAP = {
    "人工智能原理": "Principles of AI",
    "人工智能导论2": "Introduction to AI II",
    "人工智能导论": "Introduction to AI",
    "人工智能：模型与算法": "AI Models & Algorithms"
}

CHN_COLS = list(CHN_ENG_MAP.keys())
ENG_COLS = list(CHN_ENG_MAP.values())
SPAN_COLS = [f"{c}_片段" for c in CHN_COLS]

# Load embedding model
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


def clean_span(text):
    """Clean and validate text spans by length and validity"""
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    return cleaned if 4 <= len(cleaned) <= 60 else None


def cluster_and_select(sentences, scores, top_n=10):
    """Cluster sentences and select representative samples with highest confidence"""
    if len(sentences) <= top_n:
        sorted_pairs = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
        return sorted_pairs[:top_n]

    embeddings = embedding_model.encode(sentences, convert_to_numpy=True)
    n_clusters = min(top_n, len(sentences) // 2 + 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    selected = []
    for cluster_id in range(n_clusters):
        indices = [i for i, lab in enumerate(labels) if lab == cluster_id]
        scores_in_cluster = [scores[i] for i in indices]
        best_idx = indices[np.argmax(scores_in_cluster)]
        selected.append((sentences[best_idx], scores[best_idx]))

    return sorted(selected, key=lambda x: x[1], reverse=True)[:top_n]


def generate_analysis_tables():
    """Generate two analytical tables for qualitative result summarization"""
    global_samples = {label: [] for label in ENG_COLS}
    course_analysis = {}

    # Process all result files
    file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(".xlsx")]
    for filename in file_list:
        df = pd.read_excel(os.path.join(INPUT_DIR, filename))
        course_cn = filename.replace("_reviews_predictions.xlsx", "")
        course_en = COURSE_NAME_MAP.get(course_cn, course_cn)

        problem_totals = {label: 0.0 for label in ENG_COLS}
        problem_examples = {label: [] for label in ENG_COLS}

        for _, row in df.iterrows():
            for chn_label, eng_label, span_col in zip(CHN_COLS, ENG_COLS, SPAN_COLS):
                try:
                    confidence = float(row[chn_label]) if not pd.isna(row[chn_label]) else 0.0
                    span = clean_span(row[span_col])

                    if span and confidence >= CONFIDENCE_THRESHOLD:
                        global_samples[eng_label].append((span, confidence))
                        problem_totals[eng_label] += confidence
                        problem_examples[eng_label].append(span)
                except:
                    continue

        course_analysis[course_en] = {
            "totals": problem_totals,
            "examples": problem_examples
        }

    # --------------------------
    # Table 1: Global representative feedback samples
    # --------------------------
    typical_feedback = {}
    for label in ENG_COLS:
        samples = global_samples[label]
        if not samples:
            continue
        sents, scs = zip(*samples)
        selected = cluster_and_select(sents, scs, top_n=10)
        typical_feedback[label] = "; ".join(
            [f"{i + 1}: {text}({score:.3f})" for i, (text, score) in enumerate(selected)]
        )

    df_global = pd.DataFrame(list(typical_feedback.items()),
                             columns=["Problem", "Typical Feedback with Confidence"])
    df_global.to_excel(os.path.join(OUTPUT_FOLDER, "Table1 Typical Span Samples.xlsx"), index=False)

    # --------------------------
    # Table 2: Course-level diagnosis (Top5 examples per category)
    # --------------------------
    course_list = list(COURSE_NAME_MAP.values())
    table_rows = []

    for label in ENG_COLS:
        row_entry = {"Problem Category": label}
        for course in course_list:
            total_val = course_analysis[course]["totals"][label]
            examples = course_analysis[course]["examples"][label]
            unique_examples = list(dict.fromkeys(examples))[:5]
            example_str = "|".join(unique_examples) if unique_examples else "-"
            row_entry[course] = f"{total_val:.2f} ({example_str})"
        table_rows.append(row_entry)

    df_course = pd.DataFrame(table_rows)
    df_course.to_excel(os.path.join(OUTPUT_FOLDER, "Table2 Course Diagnosis Transposed.xlsx"), index=False)

    print("Qualitative analysis tables generated successfully.")


if __name__ == "__main__":
    generate_analysis_tables()