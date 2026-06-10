import pandas as pd
import os
import jieba
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer

# ===================== Path Configuration (Relative) =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, "..", "datas_negative", "negative_reviews_for_topic.xlsx")
SAVE_DIR = os.path.join(BASE_DIR, "..", "bertopic_results")

os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== Load Negative Review Data =====================
df = pd.read_excel(INPUT_FILE)
reviews = df["review_content"].dropna().astype(str).tolist()
print(f"✅ Number of reviews: {len(reviews)}")

# ===================== Chinese Tokenization Function =====================
def tokenize_zh(text):
    return jieba.lcut(text)

# Custom stop words for Chinese
stop_words = ["我", "的", "了", "是", "在", "有", "和", "就", "都", "不", "也", "很", "这个", "那个"]

vectorizer_model = CountVectorizer(
    tokenizer=tokenize_zh,
    stop_words=stop_words
)

# ===================== Sentence Embedding Model =====================
# Use online Hugging Face model instead of local path
embedding_model = SentenceTransformer("bert-base-chinese", device="cpu")
embeddings = embedding_model.encode(reviews, show_progress_bar=True)

# ===================== UMAP & HDBSCAN Configuration =====================
umap_model = umap.UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, prediction_data=True)

# ===================== BERTopic Model Training =====================
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    language="chinese",
)

topics, probs = topic_model.fit_transform(reviews, embeddings)

# ===================== Print Topic Keywords =====================
print("\n" + "="*60)
print("📌 Topic Keywords")
topic_info = topic_model.get_topic_info()
for tid in topic_info["Topic"]:
    if tid == -1:
        continue
    words = topic_model.get_topic(tid)
    word_list = [w for w, _ in words]
    print(f"Topic {tid:2d}: {' | '.join(word_list)}")

# ===================== Save Results =====================
output_path = os.path.join(SAVE_DIR, "topic_result_with_jieba.xlsx")
topic_info.to_excel(output_path, index=False)
print(f"\n🎉 Results saved to: {SAVE_DIR}")