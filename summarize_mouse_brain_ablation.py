import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


DISPLAY = {
    "SMAHD_full": "SMAHD full",
    "equal_weights": "Equal weights",
    "early_concat_single_encoder": "Early concat",
    "full_batch_no_microcluster": "Full-batch",
    "RNA_only": "RNA only",
    "ADT_only": "ATAC only",
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

DATASET_DISPLAY = {
    "mouse_brain_atac": "Mouse brain ATAC",
}


def fmt(mean, std, digits=3):
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def summarize(raw):
    requested_metrics = ["ARI", "NMI", "AMI", "FMI", "peak_memory_mib", "total_time_sec"]
    metrics = [metric for metric in requested_metrics if metric in raw.columns]
    return (
        raw.groupby(["dataset", "variant"], as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
        .pipe(lambda df: df.set_axis(["_".join(c).strip("_") for c in df.columns], axis=1))
    )


def make_table(summary):
    rows = []
    for dataset in summary["dataset"].drop_duplicates():
        indexed = summary[summary.dataset == dataset].set_index("variant")
        for variant in [v for v in ORDER if v in indexed.index]:
            r = indexed.loc[variant]
            rows.append(
                {
                    "Dataset": DATASET_DISPLAY.get(dataset, dataset),
                    "Variant": DISPLAY.get(variant, variant),
                    "ARI": fmt(r.ARI_mean, r.ARI_std),
                    "NMI": fmt(r.NMI_mean, r.NMI_std),
                    "AMI": fmt(r.AMI_mean, r.AMI_std),
                    "FMI": fmt(r.FMI_mean, r.FMI_std),
                    "Peak memory MiB": fmt(r.peak_memory_mib_mean, r.peak_memory_mib_std, 1),
                }
            )
    return pd.DataFrame(rows)


def make_plot(summary, out_path):
    plot_df = summary.copy()
    plot_df["Dataset"] = plot_df["dataset"].map(lambda x: DATASET_DISPLAY.get(x, x))
    plot_df["Variant"] = plot_df["variant"].map(lambda x: DISPLAY.get(x, x))
    variants = [DISPLAY[v] for v in ORDER if DISPLAY[v] in set(plot_df["Variant"])]
    datasets = list(plot_df["Dataset"].drop_duplicates())

    fig, axes = plt.subplots(1, len(datasets), figsize=(12, 4.8), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = plot_df[plot_df.Dataset == dataset].set_index("Variant").reindex(variants).dropna()
        ax.barh(sub.index, sub.ARI_mean, xerr=sub.ARI_std, color="#4f8f8b", ecolor="#2b2b2b")
        ax.set_title(dataset)
        ax.set_xlabel("ARI vs reference labels")
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)


def write_report(summary, table, out_dir):
    across = (
        summary.groupby("variant")[["ARI_mean", "NMI_mean", "peak_memory_mib_mean"]]
        .mean()
        .reindex([v for v in ORDER if v in set(summary["variant"])])
        .reset_index()
    )
    across["Variant"] = across["variant"].map(lambda x: DISPLAY.get(x, x))
    across = across[["Variant", "ARI_mean", "NMI_mean", "peak_memory_mib_mean"]]

    report = [
        "# SMAHD mouse brain real-data ablation",
        "",
        "Dataset: mouse brain ATAC from the configured SMART data directory. Each variant was run with seeds 101, 202 and 303 for 80 epochs. Evaluation uses the available `mclust` reference clustering labels, so these values should be interpreted as agreement with reference clusters rather than manual anatomical ground truth.",
        "",
        "## Per-dataset summary",
        "",
        markdown(table),
        "",
        "## Average across mouse brain datasets",
        "",
        markdown(across.round(4)),
        "",
        "## Interpretation",
        "",
        "- The mouse brain datasets provide a broader real-data stress test than the previous RNA+ADT sections because the second modality is ATAC/peak-derived rather than protein-derived.",
        "- Micro-cluster training should be interpreted primarily as a scalability result: compare SMAHD full with full-batch memory at similar ARI/NMI.",
        "- Modality and backbone behavior is dataset-dependent under the available reference clustering labels. Treat backbone choice and modality weights as sensitivity settings rather than universal constants.",
    ]
    report_path = os.path.join(out_dir, "smahd_mouse_brain_ablation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/mouse_brain_ablation")
    args = parser.parse_args()

    raw_path = os.path.join(args.output_dir, "smahd_realdata_ablation_raw.csv")
    summary_path = os.path.join(args.output_dir, "smahd_mouse_brain_ablation_summary.csv")
    table_path = os.path.join(args.output_dir, "smahd_mouse_brain_ablation_table.tex")
    plot_path = os.path.join(args.output_dir, "smahd_mouse_brain_ablation_ari.png")

    raw = pd.read_csv(raw_path)
    summary = summarize(raw)
    table = make_table(summary)

    summary.to_csv(summary_path, index=False)
    table.to_latex(table_path, index=False, escape=True)
    make_plot(summary, plot_path)
    report_path = write_report(summary, table, args.output_dir)

    print(summary_path)
    print(table_path)
    print(plot_path)
    print(report_path)


if __name__ == "__main__":
    main()
