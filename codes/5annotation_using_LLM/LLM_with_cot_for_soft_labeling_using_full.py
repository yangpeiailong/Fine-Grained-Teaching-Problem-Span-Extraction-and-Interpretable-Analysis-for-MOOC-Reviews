import openpyxl
import time
import re
import os
from typing import List
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from tenacity import (
    stop_after_attempt, wait_exponential, AsyncRetrying
)
from langchain_openai import ChatOpenAI

# ===================== Path Configuration (Relative) =====================
# Current script directory: codes/5annotation_using_LLM
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input: synthetic dense reviews
INPUT_FILE = os.path.join(BASE_DIR, "..", "..", "datas_for_annotation", "synthetic_dense_reviews_fixed_cols.xlsx")

# Output directory for annotated results
OUTPUT_FOLDER = os.path.join(BASE_DIR, "..", "..", "datas_after_annotation")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== 13 Teaching Problem Labels =====================
LABELS = [
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

# ===================== Model and Execution Configuration =====================
MODEL_LIST = ["deepseek-reasoner"]
ASYNC_TIMEOUT = 300
RETRY_ATTEMPTS = 2
semaphore = asyncio.Semaphore(5)

# ===================== Strict Annotation Prompt =====================
PROMPT = """
你是课程评论细粒度打分员，必须严格遵守以下铁律，违反即不合格。

【标签定义与打分规则】
只允许使用 4 个分值：0 = 无问题, 0.3 = 轻微, 0.6 = 明显, 0.9 = 严重
1 内容难度不适：内容过难、听不懂、跟不上
2 课程节奏过快：讲课太快、赶进度、来不及思考
3 讲解方式不佳：讲解混乱、照念PPT、不解释、不通俗
4 作业与测试问题：作业/试卷缺解析、错误、超纲、不合理
5 课件资料错误：PPT/讲义有错字、公式错、知识点错
6 学习进度异常：课程跳章、漏内容、前后不连贯、安排混乱
7 平台播放问题：视频卡顿、音画不同步、无法播放
8 平台功能缺陷：平台难用、bug多、功能缺失
9 线上教学效果差：整体线上效果差、效率低、无互动
10 课程内容问题：内容陈旧、重点不清、结构混乱
11 考核评分不公：给分随意、压分、标准不透明
12 学习体验糟糕：整体体验差、烦躁、无收获
13 希望改进教学：明确提出改进、建议、希望优化

【绝对强制规则】
1. 评论中【完全没提到】的维度，必须填 0，严禁乱打分
2. 严禁因为情绪差就把所有维度都打高分
3. 只输出 JSON 数组，13个数字，顺序严格不变
4. 只能出现 0、0.3、0.6、0.9，不能出现其他数值
5. 不要解释、不要思考、不要多余字符

【示例1：无问题】
评论：老师讲得很清楚，内容适中，平台流畅
输出：[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

【示例2：没有解析】
评论：没有解析，完全看不懂答案怎么来
输出：[0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.3]

【示例3：平台卡顿】
评论：视频一直卡顿，平台经常卡死
输出：[0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.9,0.3,0.0,0.0,0.3,0.3]

--------------------------------
评论内容：{review}
输出：
"""

# ===================== Helper: Extract JSON Score Array =====================
def extract_json_array(text: str) -> List[float]:
    try:
        text = re.sub(r'```json|```', '', text).strip()
        import json
        return json.loads(text)
    except:
        return [0.0] * 13

# ===================== Asynchronous LLM Annotation =====================
async def async_llm_call(llm: ChatOpenAI, review: str):
    start_time = time.time()
    async with semaphore:
        prompt = PROMPT.format(review=review)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=2),
            reraise=False
        ):
            with attempt:
                try:
                    resp = await asyncio.wait_for(
                        llm.ainvoke(prompt),
                        timeout=ASYNC_TIMEOUT
                    )
                    scores = extract_json_array(resp.content)
                    legal_scores = {0.0, 0.3, 0.6, 0.9}
                    scores = [s if s in legal_scores else 0.0 for s in scores]
                    return scores[:13], time.time() - start_time
                except Exception as e:
                    if attempt.retry_state.attempt_number >= RETRY_ATTEMPTS:
                        return [0.0] * 13, time.time() - start_time
                    raise

# ===================== Main Full Annotation Pipeline =====================
async def run_annotation():
    print("===== Full Soft Label Annotation for Synthetic Reviews =====")
    df = pd.read_excel(INPUT_FILE)
    reviews = df["review_content"].astype(str).tolist()
    print(f"✅ Loaded reviews: {len(reviews)}")

    # Load API keys from environment variables
    api_qwen = os.getenv("TONGYI_API_KEY")
    api_deepseek = os.getenv("DEEPSEEK_API_KEY")

    for model in MODEL_LIST:
        print(f"\n===== Processing model: {model} =====")
        try:
            if model.startswith("deepseek"):
                llm = ChatOpenAI(
                    model=model,
                    api_key=api_deepseek,
                    base_url="https://api.deepseek.com",
                    temperature=0.01,
                    timeout=ASYNC_TIMEOUT
                )
            else:
                llm = ChatOpenAI(
                    model=model,
                    api_key=api_qwen,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                    temperature=0.01,
                    timeout=ASYNC_TIMEOUT
                )
        except Exception as e:
            print(f"❌ Failed to initialize {model}: {e}")
            continue

        # Execute concurrent annotation tasks
        tasks = [async_llm_call(llm, r) for r in reviews]
        results = await tqdm_asyncio.gather(*tasks)

        # Build result DataFrame
        scores_list = [s for s, _ in results]
        score_df = pd.DataFrame(scores_list, columns=LABELS)
        final_df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

        # Save fully annotated results
        out_path = os.path.join(OUTPUT_FOLDER, f"soft_annotated_13labels_{model}_synthetic_dense.xlsx")
        final_df.to_excel(out_path, index=False)
        print(f"✅ Saved to: {out_path}")

# ===================== Entry Point =====================
if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_annotation())