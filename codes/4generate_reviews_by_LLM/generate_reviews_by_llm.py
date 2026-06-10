import openpyxl
import time
import re
import os
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from tenacity import (
    stop_after_attempt, wait_exponential, AsyncRetrying
)
from langchain_openai import ChatOpenAI

# ===================== Path Configuration (Relative) =====================
# Current script directory: codes/4generate_reviews_by_LLM
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Output directory: ../datas_for_annotation
OUTPUT_FOLDER = os.path.join(BASE_DIR, "..", "..", "datas_for_annotation")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== Generation Configuration =====================
GENERATE_COUNT = 1700

# ===================== LLM Model Configuration =====================
MODEL_NAME = "deepseek-reasoner"
ASYNC_TIMEOUT = 300
RETRY_ATTEMPTS = 2
semaphore = asyncio.Semaphore(10)

# ===================== Fixed Synthetic Sample Identifiers =====================
# Unique markers to distinguish synthetic data from real reviews
FIXED_RATING = 1
FIXED_COURSE_TERM = "Synthetic Course"
FIXED_USER_NICKNAME = "SYNTHETIC_USER"
FIXED_REVIEW_TIME = "2025-01-01"
FIXED_COURSE_NAME = "Synthetic Multi-label Dense Sample"

# ===================== Generation Prompt (Chinese for review generation) =====================
GENERATE_PROMPT = """
你是大学MOOC课程评论生成器，只生成真实、简短、口语化的学生负面评论。

必须严格遵守：
1. 每条评论必须 **同时包含至少 3~4 个以下问题**：
   内容难度不适、课程节奏过快、讲解方式不佳、作业与测试问题、课件资料错误、学习进度异常、平台播放问题、平台功能缺陷、线上教学效果差、课程内容问题、考核评分不公、学习体验糟糕、希望改进教学

2. 风格必须是真实学生评论，简短、自然、口语化
3. 只输出纯文本

生成1条符合要求的评论：
"""

# ===================== Generate Single Review with Retry =====================
async def generate_one_review(llm: ChatOpenAI, index: int):
    async with semaphore:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=2),
            reraise=False
        ):
            with attempt:
                try:
                    resp = await llm.ainvoke(GENERATE_PROMPT)
                    content = resp.content.strip()
                    print(f"[{index}] Generated: {content}")
                    return content
                except Exception as e:
                    print(f"[{index}] Generation failed")
                    return ""

# ===================== Main Generation Pipeline =====================
async def run_generate_only():
    print("===== Generating Dense Multi-label Synthetic Reviews =====")
    api_deepseek = os.getenv("DEEPSEEK_API_KEY")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_deepseek,
        base_url="https://api.deepseek.com",
        timeout=ASYNC_TIMEOUT
    )

    tasks = [generate_one_review(llm, i+1) for i in range(GENERATE_COUNT)]
    reviews = await tqdm_asyncio.gather(*tasks)

    # Filter invalid and empty reviews
    reviews = [r.strip() for r in reviews if r and len(r) >= 5]

    # Construct standardized DataFrame
    df = pd.DataFrame({
        "review_content": reviews,
        "rating": FIXED_RATING,
        "course_term": FIXED_COURSE_TERM,
        "user_nickname": FIXED_USER_NICKNAME,
        "review_time": FIXED_REVIEW_TIME,
        "course_name": FIXED_COURSE_NAME
    })

    out_path = os.path.join(OUTPUT_FOLDER, "synthetic_dense_reviews_fixed_cols.xlsx")
    df.to_excel(out_path, index=False)

    print("\n✅ Generation completed!")
    print("✅ Synthetic samples are clearly marked")
    print("✅ Output path:", out_path)

# ===================== Entry Point =====================
if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_generate_only())