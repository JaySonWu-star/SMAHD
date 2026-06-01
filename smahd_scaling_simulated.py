import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments_smahd_ablation_simulated import RunConfig, make_spatial_multiomics, score_embedding, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/scaling_simulated")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--sizes", default="2000,5000,10000,20000")
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--infer-batch-size", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_sizes = [int(x) for x in args.sizes.split(",") if x.strip()]
    configs = [
        RunConfig("microcluster", "GAT", "view_specific", "cluster", (0.45, 0.45, 0.10)),
        RunConfig("full_batch", "GAT", "view_specific", "full", (0.45, 0.45, 0.10)),
    ]
    rows = []
    for total_n in total_sizes:
        n_per_domain = max(1, total_n // 4)
        coords, labels, features, edge = make_spatial_multiomics(args.seed + total_n, n_per_domain=n_per_domain)
        actual_n = len(labels)
        print("DATASET", total_n, "actual", actual_n, "edges", edge.shape[1], flush=True)
        for config in configs:
            row = {
                "target_n": total_n,
                "actual_n": actual_n,
                "edges": int(edge.shape[1]),
                "variant": config.variant,
                "epochs": args.epochs,
            }
            try:
                start = time.time()
                emb, elapsed, peak_mib = train_model(
                    features=features,
                    edge=edge,
                    config=config,
                    seed=args.seed + total_n,
                    epochs=args.epochs,
                    train_batch_size=args.train_batch_size,
                    infer_batch_size=args.infer_batch_size,
                    device=device,
                )
                row.update(score_embedding(emb, labels))
                row.update(
                    {
                        "elapsed_sec": elapsed,
                        "wall_sec": time.time() - start,
                        "peak_memory_mib": peak_mib,
                        "status": "completed",
                    }
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    row.update(
                        {
                            "elapsed_sec": np.nan,
                            "wall_sec": np.nan,
                            "peak_memory_mib": np.nan,
                            "ARI": np.nan,
                            "NMI": np.nan,
                            "AMI": np.nan,
                            "FMI": np.nan,
                            "V_measure": np.nan,
                            "homogeneity": np.nan,
                            "status": "OOM",
                            "error": str(exc)[:500],
                        }
                    )
                else:
                    raise
            rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "smahd_scaling_simulated_raw.csv"), index=False)
            print(row, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output_dir, "smahd_scaling_simulated_raw.csv"), index=False)
    completed = df[df["status"] == "completed"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=180)
    for variant, sub in completed.groupby("variant"):
        sub = sub.sort_values("actual_n")
        axes[0].plot(sub["actual_n"], sub["peak_memory_mib"], marker="o", label=variant)
        axes[1].plot(sub["actual_n"], sub["elapsed_sec"], marker="o", label=variant)
        axes[2].plot(sub["actual_n"], sub["ARI"], marker="o", label=variant)
    axes[0].set_ylabel("Peak memory (MiB)")
    axes[1].set_ylabel("Runtime (s)")
    axes[2].set_ylabel("ARI")
    for ax in axes:
        ax.set_xlabel("Number of spots")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "smahd_scaling_simulated_summary.png"))
    print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
