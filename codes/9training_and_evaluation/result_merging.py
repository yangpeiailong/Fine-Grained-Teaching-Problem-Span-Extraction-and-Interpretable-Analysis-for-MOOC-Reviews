import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== Path Configuration (Relative) =====================
# Current script directory: codes/9training_and_evaluation
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "..")

# Result directory
SAVE_ROOT = os.path.join(ROOT_DIR, "results_baseline")

# ===================== Font Settings =====================
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===================== Filename Parser =====================
def parse_filename(filename):
    core = filename.replace("baseline_result_", "").replace(".xlsx", "")
    parts = core.split("_")
    synth = "With Synthetic" if parts[0] == "synth" else "No Synthetic"
    freeze = "Unfrozen BERT" if parts[1] == "nofreeze" else "Frozen BERT"
    label = "Soft Label" if parts[2] == "soft" else "Hard Label"
    return f"{synth}, {freeze}, {label}"


# ===================== Standard Configuration Order =====================
config_order = [
    "No Synthetic, Frozen BERT, Hard Label",
    "No Synthetic, Frozen BERT, Soft Label",
    "No Synthetic, Unfrozen BERT, Hard Label",
    "No Synthetic, Unfrozen BERT, Soft Label",
    "With Synthetic, Frozen BERT, Hard Label",
    "With Synthetic, Frozen BERT, Soft Label",
    "With Synthetic, Unfrozen BERT, Hard Label",
    "With Synthetic, Unfrozen BERT, Soft Label",
]

# ===================== Load All Result Files =====================
all_dfs = []
files = [f for f in os.listdir(SAVE_ROOT) if f.endswith(".xlsx") and "baseline_result_" in f]

for f in files:
    file_path = os.path.join(SAVE_ROOT, f)
    df = pd.read_excel(file_path, index_col="model")
    df = df[["F1_0.2", "Span_F1"]].reset_index()
    df["Config"] = parse_filename(f)
    all_dfs.append(df)

# Combine and reshape data
combined = pd.concat(all_dfs, ignore_index=True)
pivot_cls = combined.pivot(index="Config", columns="model", values="F1_0.2").reindex(config_order)
pivot_span = combined.pivot(index="Config", columns="model", values="Span_F1").reindex(config_order)


# ===================== Plot Function with Dividers =====================
def plot_bar_with_divider(pivot_df, title, ylabel, save_path):
    configs = pivot_df.index
    models = pivot_df.columns
    n_configs = len(configs)
    n_models = len(models)
    width = 0.1
    x = np.arange(n_configs) * (n_models * width + 0.3)

    fig, ax = plt.subplots(figsize=(18, 9))

    # Draw bars
    for i, model in enumerate(models):
        offset = i * width
        ax.bar(x + offset, pivot_df[model], width, label=model)

    # Draw vertical dividers
    for i in range(1, n_configs):
        line_x = x[i] - 0.15
        ax.axvline(x=line_x, color='gray', linestyle='--', alpha=0.5)

    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)
    ax.set_xticks(x + (n_models * width) / 2 - width / 2)
    ax.set_xticklabels(configs, rotation=15, fontsize=10)
    ax.legend(fontsize=10, bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ===================== Generate Figures =====================
# Classification F1 figure
cls_fig_path = os.path.join(SAVE_ROOT, "cls_f1_comparison_divider.png")
plot_bar_with_divider(
    pivot_cls,
    "Classification F1 under Different Configurations",
    "F1 Score",
    cls_fig_path
)

# Span F1 figure
span_fig_path = os.path.join(SAVE_ROOT, "span_f1_comparison_divider.png")
plot_bar_with_divider(
    pivot_span,
    "Span F1 under Different Configurations",
    "Span F1 Score",
    span_fig_path
)

# ===================== Save Final Summary Excel =====================
summary_path = os.path.join(SAVE_ROOT, "summary_results_final.xlsx")
with pd.ExcelWriter(summary_path) as writer:
    pivot_cls.to_excel(writer, sheet_name="Classification_F1")
    pivot_span.to_excel(writer, sheet_name="Span_F1")

# ===================== Log Output =====================
print("✅ Figures and summary generated successfully!")
print(f"📊 Classification plot: {cls_fig_path}")
print(f"📊 Span prediction plot: {span_fig_path}")
print(f"📊 Final summary Excel: {summary_path}")