import os
import pandas as pd

# ===================== 自动相对路径（无任何本地硬编码）=====================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

INPUT_DIR = os.path.join(project_root, "predicted_results")
OUTPUT_SOFT = os.path.join(project_root, "mid_results", "output_softonly")
os.makedirs(OUTPUT_SOFT, exist_ok=True)

# 和excel真实中文列名完全匹配
CHN_PROB_COLS = [
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

# 移除下划线后的标准英文输出名称
ENG_PROB_COLS = [
    "Content difficulty mismatch",
    "Excessively fast course pace",
    "Inappropriate explanation methods",
    "Homework and testing issues",
    "Errors in courseware and materials",
    "Abnormal learning progress statistics",
    "Platform playback abnormalities",
    "Platform function defects",
    "Poor online teaching effect",
    "Course content quality issues",
    "Unfair assessment and scoring",
    "Poor overall learning experience",
    "Expectations for teaching improvement"
]

# 移除下划线后的课程英文名称
COURSE_EN_MAP = {
    "人工智能原理": "Principles of AI",
    "人工智能导论": "Introduction to AI",
    "人工智能导论2": "Introduction to AI II",
    "人工智能：模型与算法": "AI Models & Algorithms"
}
THRESH = 0.2

def main():
    file_list = [f for f in os.listdir(INPUT_DIR) if f.endswith(".xlsx")]
    all_raw_df = []
    course_info = []
    course_perf_dict = {}
    course_imp_dict = {}

    # 逐课程解析
    for fname in file_list:
        fpath = os.path.join(INPUT_DIR, fname)
        df = pd.read_excel(fpath)
        c_cn = fname.replace("_reviews_predictions.xlsx","")
        c_en = COURSE_EN_MAP.get(c_cn, c_cn)
        all_raw_df.append(df[CHN_PROB_COLS].copy())
        total = len(df)

        # 1.单课程：每类问题均值(Performance)、出现占比(Importance)
        per_mean = df[CHN_PROB_COLS].mean(axis=0)
        per_imp = (df[CHN_PROB_COLS] >= THRESH).sum(axis=0) / total

        course_perf_dict[c_en] = per_mean
        course_imp_dict[c_en] = per_imp

        course_info.append({
            "Course": c_en,
            "Total Reviews": total
        })

    # 全局合并计算全局权重（和原论文一致）
    global_df = pd.concat(all_raw_df)
    global_total = len(global_df)
    global_imp = (global_df[CHN_PROB_COLS] >= THRESH).sum(axis=0)/global_total
    global_weight = (global_imp / global_imp.sum()).round(4)

    # 输出1：各课程13项指标明细
    df_perf = pd.DataFrame(course_perf_dict).T
    df_perf.columns = ENG_PROB_COLS
    df_perf.to_excel(os.path.join(OUTPUT_SOFT,"Course Performance Matrix.xlsx"),index=True)

    df_imp = pd.DataFrame(course_imp_dict).T
    df_imp.columns = ENG_PROB_COLS
    df_imp.to_excel(os.path.join(OUTPUT_SOFT,"Course Importance Matrix.xlsx"),index=True)

    # 输出2：全局权重表
    df_global = pd.DataFrame({
        "Problem":ENG_PROB_COLS,
        "Global Importance":global_imp.values,
        "Global Weight":global_weight.values
    })
    df_global.to_excel(os.path.join(OUTPUT_SOFT,"Global Importance Weight.xlsx"),index=False)

    # 输出3：课程综合严重得分&风险分级
    res_list = []
    for item in course_info:
        c_name = item["Course"]
        total = item["Total Reviews"]
        perf_ser = df_perf.loc[c_name]
        # 加权综合严重度：越高课程问题越严重
        severity = (perf_ser * global_weight.values).sum()

        # 风险划分
        if severity >= 0.3:
            risk = "High Risk"
        elif severity >= 0.15:
            risk = "Medium Risk"
        else:
            risk = "Low Risk"

        res_list.append({
            "Course":c_name,
            "Total Reviews":total,
            "Weighted Severity":round(severity,4),
            "Risk Level":risk
        })
    df_final = pd.DataFrame(res_list).sort_values("Weighted Severity",ascending=False)
    df_final.to_excel(os.path.join(OUTPUT_SOFT,"Course Severity Risk.xlsx"),index=False)
    print("✅ 分项+加权综合全部输出完毕")

if __name__ == "__main__":
    main()