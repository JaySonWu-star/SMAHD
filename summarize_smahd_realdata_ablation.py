import argparse
import os

import pandas as pd


DISPLAY = {
    "SMAHD_full": "SMAHD full",
    "equal_weights": "Equal weights",
    "early_concat_single_encoder": "Early concat",
    "full_batch_no_microcluster": "Full-batch",
    "RNA_only": "RNA only",
    "ADT_only": "Other modality only",
    "GCN_encoder": "GCN encoder",
    "SAGE_encoder": "SAGE encoder",
}

ORDER = [
    "SMAHD_full",
    "equal_weights",
    "early_concat_single_encoder",
    "full_batch_no_microcluster",
    "RNA_only",
    "ADT_only",
    "GCN_encoder",
    "SAGE_encoder",
]


def fmt(mean, std, digits=3):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/realdata_ablation")
    args = parser.parse_args()

    summary_path = os.path.join(args.output_dir, "smahd_realdata_ablation_summary.csv")
    report_path = os.path.join(args.output_dir, "smahd_realdata_ablation_report.md")
    table_path = os.path.join(args.output_dir, "smahd_realdata_ablation_table.tex")

    summary = pd.read_csv(summary_path)
    rows = []
    for dataset in summary["dataset"].drop_duplicates():
        indexed = summary[summary.dataset == dataset].set_index("variant")
        for variant in [v for v in ORDER if v in indexed.index]:
            r = indexed.loc[variant]
            rows.append(
                {
                    "Dataset": dataset,
                    "Variant": DISPLAY.get(variant, variant),
                    "ARI": fmt(r.ARI_mean, r.ARI_std),
                    "NMI": fmt(r.NMI_mean, r.NMI_std),
                    "AMI": fmt(r.AMI_mean, r.AMI_std),
                    "FMI": fmt(r.FMI_mean, r.FMI_std),
                    "Peak memory MiB": fmt(r.peak_memory_mib_mean, r.peak_memory_mib_std, 1),
                }
            )
    table = pd.DataFrame(rows)
    across = summary.groupby("variant")[["ARI_mean", "NMI_mean", "peak_memory_mib_mean"]].mean()
    across = across.reindex([v for v in ORDER if v in across.index]).reset_index()
    across["Variant"] = across["variant"].map(lambda x: DISPLAY.get(x, x))
    across = across[["Variant", "ARI_mean", "NMI_mean", "peak_memory_mib_mean"]]

    report = [
        "# SMAHD real-data ablation report",
        "",
        "Real-data ablations compare multimodal SMAHD against single-modality, early-concatenation, full-batch, and graph-backbone variants.",
        "",
        "## Per-dataset summary",
        "",
        markdown(table),
        "",
        "## Average across datasets",
        "",
        markdown(across.round(4)),
        "",
        "## Interpretation",
        "",
        "These results support reporting graph backbone and reconstruction weights as explicit sensitivity parameters rather than universal constants.",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=True))
    print(report_path)
    print(table_path)


if __name__ == "__main__":
    main()
