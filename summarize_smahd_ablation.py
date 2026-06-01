import argparse
import os

import pandas as pd
from scipy.stats import ttest_rel


DISPLAY = {
    "SMAHD_full": "SMAHD (GAT, view-specific, micro-cluster)",
    "equal_weights": "Equal reconstruction weights",
    "early_concat_single_encoder": "Early concatenation, single encoder",
    "full_batch_no_microcluster": "Full-batch training",
    "GCN_encoder": "GCN encoder",
    "SAGE_encoder": "SAGE encoder",
    "GraphConv_encoder": "GraphConv encoder",
}

ORDER = [
    "SMAHD_full",
    "equal_weights",
    "early_concat_single_encoder",
    "full_batch_no_microcluster",
    "GCN_encoder",
    "SAGE_encoder",
    "GraphConv_encoder",
]


def fmt(mean, std, digits=3):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/ablation_simulated_5seed")
    args = parser.parse_args()

    raw_path = os.path.join(args.output_dir, "smahd_simulated_ablation_raw.csv")
    summary_path = os.path.join(args.output_dir, "smahd_simulated_ablation_summary.csv")
    report_path = os.path.join(args.output_dir, "smahd_simulated_ablation_report.md")
    latex_path = os.path.join(args.output_dir, "smahd_simulated_ablation_table.tex")

    raw = pd.read_csv(raw_path)
    summary = pd.read_csv(summary_path).set_index("variant").loc[ORDER].reset_index()
    table = pd.DataFrame(
        {
            "Variant": [DISPLAY[x] for x in summary["variant"]],
            "ARI": [fmt(r.ARI_mean, r.ARI_std) for r in summary.itertuples()],
            "NMI": [fmt(r.NMI_mean, r.NMI_std) for r in summary.itertuples()],
            "AMI": [fmt(r.AMI_mean, r.AMI_std) for r in summary.itertuples()],
            "FMI": [fmt(r.FMI_mean, r.FMI_std) for r in summary.itertuples()],
            "Runtime_s": [fmt(r.elapsed_sec_mean, r.elapsed_sec_std, 2) for r in summary.itertuples()],
            "Peak_memory_MiB": [fmt(r.peak_memory_mib_mean, r.peak_memory_mib_std, 1) for r in summary.itertuples()],
        }
    )

    pivot_ari = raw.pivot(index="seed", columns="variant", values="ARI")
    full_ari = pivot_ari["SMAHD_full"]
    comparisons = []
    for variant in ORDER:
        if variant == "SMAHD_full":
            continue
        delta = float((full_ari - pivot_ari[variant]).mean())
        pval = float(ttest_rel(full_ari, pivot_ari[variant]).pvalue)
        comparisons.append(f"- {DISPLAY[variant]}: mean ARI delta = {delta:.4f}, paired t-test p = {pval:.4g}.")

    report = [
        "# SMAHD simulated ablation report",
        "",
        "Configuration: synthetic three-view spatial multi-omics benchmark; KNN spatial graph with k=15; post hoc KMeans evaluation.",
        "",
        "## Summary table",
        "",
        to_markdown(table),
        "",
        "## Paired comparisons against SMAHD_full",
        "",
        *comparisons,
        "",
        "## Interpretation",
        "",
        "These results should be interpreted as module sensitivity analyses. Micro-cluster mini-batch training reduces memory relative to full-batch training, and encoder choice is dataset dependent.",
    ]

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report) + "\n")
    with open(latex_path, "w", encoding="utf-8") as handle:
        handle.write(table.to_latex(index=False, escape=True))
    print(report_path)
    print(latex_path)


if __name__ == "__main__":
    main()
