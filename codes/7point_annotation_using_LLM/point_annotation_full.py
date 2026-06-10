import os
import json
import asyncio
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from tenacity import stop_after_attempt, wait_exponential, AsyncRetrying
from langchain_openai import ChatOpenAI

# ===================== Path Configuration (Relative) =====================
# Current script directory: codes/7point_annotation_using_LLM
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Full dataset input path
INPUT_FILE = os.path.join(BASE_DIR, "..", "..", "datas_after_annotation", "final_mixed_dataset_70real_30synth.xlsx")

# Output directory for fully annotated results
OUTPUT_FOLDER = os.path.join(BASE_DIR, "..", "..", "datas_after_annotation")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===================== Annotation Configuration =====================
THRESHOLD = 0.2
MODEL_LIST = ["qwen3.6-plus"]
ASYNC_TIMEOUT = 300
RETRY_TIMES = 2
semaphore = asyncio.Semaphore(5)

# ===================== 13 Teaching Problem Categories =====================
PROBLEM_COLS = [
    "内容难度不适", "课程节奏过快", "讲解方式不佳", "作业与测试问题",
    "课件资料错误", "学习进度异常", "平台播放问题", "平台功能缺陷",
    "线上教学效果差", "课程内容问题", "考核评分不公", "学习体验糟糕", "希望改进教学"
]

# ===================== Character-level Splitting =====================
def char_split(text):
    text = str(text).strip()
    return list(text)

# ===================== Character-level Span Annotation Prompt =====================
PROMPT = """
你是专业的教学评论问题片段标注员。
任务：根据【按字拆分后的评论】和【允许标注的问题】，输出每个问题对应的连续字片段的 start_char_idx、end_char_idx（从0开始，左闭右闭）。

规则：
1. 下标必须严格对应字序列位置！
2. 只标注给定问题，不漏标、不新增！
3. 同一片段可属于多个问题，分别输出！
4. 输出严格JSON，无多余内容！
5. 无问题则返回空 spans！

强制约束：
1. 给出的允许问题有几类，你就必须输出几条不同或相同的文本span，**不能缺任何一类**。
2. 每一类允许问题，都必须单独对应一段原文片段，不能遗漏、不能合并省略。
3. 若多个问题类型对应同一句原文，也要重复生成多条span，分别归属对应问题类型。
4. 严格按给定问题列表逐条匹配，不得自行删减问题类别。

输出格式：
{{
  "spans": [
    {{
      "problem_type": "问题名称",
      "span_text": "片段原文",
      "start_char_idx": 0,
      "end_char_idx": 0
    }}
  ]
}}

按字拆分后的评论：{char_sequence}
允许标注的问题：{allowed_problems}
"""

# ===================== Helper: Get Valid Problems Above Threshold =====================
def get_allowed_problems(row):
    allowed = []
    for col in PROBLEM_COLS:
        if col in row and not pd.isna(row[col]) and row[col] >= THRESHOLD:
            allowed.append(col)
    return allowed

# ===================== Asynchronous Annotation Function =====================
async def annotate(llm, char_sequence, allowed):
    async with semaphore:
        prompt = PROMPT.format(
            char_sequence=char_sequence,
            allowed_problems=allowed
        )
        async for attempt in AsyncRetrying(
                stop=stop_after_attempt(RETRY_TIMES),
                wait=wait_exponential(multiplier=1, min=1, max=2)
        ):
            with attempt:
                try:
                    resp = await asyncio.wait_for(llm.ainvoke(prompt), timeout=ASYNC_TIMEOUT)
                    return json.loads(resp.content.strip())
                except:
                    return {"spans": []}

# ===================== Full Annotation Pipeline =====================
async def run_full_annotation():
    print("===== Starting Full Character-level Annotation =====")

    # Load full dataset
    df = pd.read_excel(INPUT_FILE).copy()
    api_qwen = os.getenv("TONGYI_API_KEY")

    for model in MODEL_LIST:
        print(f"\nProcessing with model: {model} | Total records: {len(df)}")
        llm = ChatOpenAI(
            model=model,
            api_key=api_qwen,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            temperature=0.1
        )

        # Prepare tasks
        tasks = []
        char_sequences = []
        allowed_list = []

        for _, row in df.iterrows():
            text = str(row["review_content"]).strip()
            chars = char_split(text)
            char_sequences.append(" | ".join(chars))
            allowed = get_allowed_problems(row)
            allowed_list.append(" | ".join(allowed))
            tasks.append(annotate(llm, chars, allowed))

        # Execute concurrent tasks
        results = await tqdm_asyncio.gather(*tasks)

        # Save annotation results
        df["char_sequence"] = char_sequences
        df["allowed_problems(≥0.2)"] = allowed_list
        df["span_annotations(char_level)"] = [json.dumps(r, ensure_ascii=False) for r in results]

        # Final output path
        save_path = os.path.join(OUTPUT_FOLDER, "final_mixed_dataset_70real_30synth_char_span_annotated_full.xlsx")
        df.to_excel(save_path, index=False)

        print(f"\n✅ Full annotation completed!")
        print(f"✅ Total annotated records: {len(df)}")
        print(f"✅ File saved to: {save_path}")

# ===================== Entry Point =====================
if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_full_annotation())