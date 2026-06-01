import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, fowlkes_mallows_score
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiments_smahd_ablation_simulated import RunConfig, train_model


DATA_ROOT = os.environ.get("SMAHD_DATA_ROOT", "data")


def data_path(*parts):
    return os.path.join(DATA_ROOT, *parts)


DATASETS = {
    "tonsil_s1": {
        "rna": data_path("human-tonsil", "raw", "slice1", "adata_rna.h5ad"),
        "adt": data_path("human-tonsil", "raw", "slice1", "adata_adt.h5ad"),
        "label_col": "final_annot",
        "combined": None,
    },
    "tonsil_s2": {
        "rna": data_path("human-tonsil", "raw", "slice2", "adata_rna.h5ad"),
        "adt": data_path("human-tonsil", "raw", "slice2", "adata_adt.h5ad"),
        "label_col": "final_annot",
        "combined": None,
    },
    "hln_A1": {
        "rna": data_path("SMART", "datasets", "Human_Lymph_Node_A1", "adata_RNA.h5ad"),
        "adt": data_path("SMART", "datasets", "Human_Lymph_Node_A1", "adata_ADT.h5ad"),
        "label_col": "anno",
        "combined": data_path("SMART", "results", "Human_Lymph_Node_A1", "adata_all.h5ad"),
    },
    "hln_D1": {
        "rna": data_path("SMART", "datasets", "Human_Lymph_Node_D1", "adata_RNA.h5ad"),
        "adt": data_path("SMART", "datasets", "Human_Lymph_Node_D1", "adata_ADT.h5ad"),
        "label_col": "anno",
        "combined": data_path("SMART", "results", "Human_Lymph_Node_D1", "adata_all.h5ad"),
        "rna_obsm": "X_RNA_pca",
        "other_obsm": "X_ADT_pca",
    },
    "mouse_brain_atac": {
        "rna": data_path("SMART", "datasets", "Dataset7_Mouse_Brain_ATAC", "adata_RNA.h5ad"),
        "adt": data_path("SMART", "datasets", "Dataset7_Mouse_Brain_ATAC", "adata_peaks_normalized.h5ad"),
        "label_col": "mclust",
        "combined": data_path("SMART", "results", "Dataset7_Mouse_Brain_ATAC", "adata_all.h5ad"),
        "rna_obsm": "X_RNA_pca",
        "other_obsm": "X_ATAC_pca",
    },
}


def dense_array(x):
    if sp.issparse(x):
        return x.toarray()
    return np.asarray(x)


def pca_features(adata, n_comps=30, is_rna=True):
    if "X_pca" in adata.obsm and adata.obsm["X_pca"].shape[1] >= min(n_comps, adata.n_vars):
        return np.asarray(adata.obsm["X_pca"][:, : min(n_comps, adata.obsm["X_pca"].shape[1])], dtype=np.float32)
    adata = adata.copy()
    adata.var_names_make_unique()
    if is_rna:
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        n_top = min(3000, adata.n_vars)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="cell_ranger")
        if "highly_variable" in adata.var:
            adata = adata[:, adata.var["highly_variable"]].copy()
    x = dense_array(adata.X).astype(np.float32)
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    n = min(n_comps, x.shape[0] - 1, x.shape[1])
    return PCA(n_components=n, random_state=0).fit_transform(x).astype(np.float32)


def load_dataset(name):
    cfg = DATASETS[name]
    rna = sc.read_h5ad(cfg["rna"])
    adt = sc.read_h5ad(cfg["adt"])
    rna.var_names_make_unique()
    adt.var_names_make_unique()

    if cfg["combined"]:
        combined = sc.read_h5ad(cfg["combined"])
        labels = combined.obs[cfg["label_col"]].astype(str)
        keep = (~labels.str.lower().isin(["nan", "exclude"])) & labels.notna()
        common = [x for x in combined.obs_names[keep] if x in rna.obs_names and x in adt.obs_names]
        combined = combined[common].copy()
        rna = rna[common].copy()
        adt = adt[common].copy()
        labels = combined.obs[cfg["label_col"]].astype(str).to_numpy()
        rna_key = cfg.get("rna_obsm", "X_RNA_pca")
        other_key = cfg.get("other_obsm", "X_ADT_pca")
        if rna_key in combined.obsm and other_key in combined.obsm:
            rna_feat = np.asarray(combined.obsm[rna_key][:, :30], dtype=np.float32)
            adt_feat = np.asarray(combined.obsm[other_key][:, :30], dtype=np.float32)
        else:
            rna_feat = pca_features(rna, 30, True)
            adt_feat = pca_features(adt, 30, False)
        spatial = np.asarray(combined.obsm["spatial"], dtype=np.float32)
    else:
        labels = rna.obs[cfg["label_col"]].astype(str)
        keep = (~labels.str.lower().isin(["nan", "exclude"])) & labels.notna()
        common = [x for x in rna.obs_names[keep] if x in adt.obs_names]
        rna = rna[common].copy()
        adt = adt[common].copy()
        labels = rna.obs[cfg["label_col"]].astype(str).to_numpy()
        rna_feat = pca_features(rna, 30, True)
        adt_feat = pca_features(adt, 30, False)
        spatial = np.asarray(rna.obsm["spatial"], dtype=np.float32)

    # Standardize feature matrices after any precomputed PCA so all variants see comparable scales.
    rna_feat = StandardScaler().fit_transform(rna_feat).astype(np.float32)
    adt_feat = StandardScaler().fit_transform(adt_feat).astype(np.float32)
    y = LabelEncoder().fit_transform(labels)
    edge = np.asarray(kneighbors_graph(spatial, n_neighbors=15, mode="connectivity", include_self=False).nonzero(), dtype=np.int64)
    edge = np.concatenate([edge, edge[::-1]], axis=1)
    edge = np.unique(edge, axis=1)
    return {
        "name": name,
        "features": [torch.tensor(rna_feat), torch.tensor(adt_feat)],
        "labels": y,
        "label_names": labels,
        "spatial": spatial,
        "edge": torch.tensor(edge, dtype=torch.long),
        "n_clusters": len(np.unique(y)),
    }


def configs():
    return [
        RunConfig("SMAHD_full", "GAT", "view_specific", "cluster", (0.4, 0.6)),
        RunConfig("equal_weights", "GAT", "view_specific", "cluster", (1.0, 1.0)),
        RunConfig("early_concat_single_encoder", "GAT", "concat", "cluster", (1.0,)),
        RunConfig("full_batch_no_microcluster", "GAT", "view_specific", "full", (0.4, 0.6)),
        RunConfig("RNA_only", "GAT", "view_specific", "cluster", (1.0,)),
        RunConfig("ADT_only", "GAT", "view_specific", "cluster", (1.0,)),
        RunConfig("GCN_encoder", "GCN", "view_specific", "cluster", (0.4, 0.6)),
        RunConfig("SAGE_encoder", "SAGE", "view_specific", "cluster", (0.4, 0.6)),
    ]


def feature_subset(all_features, variant):
    if variant == "RNA_only":
        return [all_features[0]]
    if variant == "ADT_only":
        return [all_features[1]]
    return all_features


def score(embedding, labels, n_clusters):
    pred = KMeans(n_clusters=n_clusters, n_init=30, random_state=0).fit_predict(embedding)
    return {
        "ARI": adjusted_rand_score(labels, pred),
        "NMI": normalized_mutual_info_score(labels, pred),
        "AMI": adjusted_mutual_info_score(labels, pred),
        "FMI": fowlkes_mallows_score(labels, pred),
    }


def plot_summary(summary, out_dir):
    order = ["SMAHD_full", "RNA_only", "ADT_only", "early_concat_single_encoder", "equal_weights", "full_batch_no_microcluster", "GCN_encoder", "SAGE_encoder"]
    datasets = list(summary["dataset"].unique())
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=180)
    x = np.arange(len(datasets))
    width = 0.1
    for idx, variant in enumerate(order):
        vals = []
        errs = []
        for dataset in datasets:
            row = summary[(summary["dataset"] == dataset) & (summary["variant"] == variant)]
            vals.append(float(row["ARI_mean"].iloc[0]) if len(row) else np.nan)
            errs.append(float(row["ARI_std"].iloc[0]) if len(row) else 0)
        ax.bar(x + (idx - len(order) / 2) * width + width / 2, vals, width, yerr=errs, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("ARI vs annotation")
    ax.set_ylim(0, 1)
    ax.legend(ncol=4, fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "smahd_realdata_ablation_ari.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/realdata_ablation")
    parser.add_argument("--datasets", default="tonsil_s1,tonsil_s2,hln_A1,hln_D1")
    parser.add_argument("--seeds", default="101,202,303")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--infer-batch-size", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        data = load_dataset(dataset_name)
        print("DATASET", dataset_name, "n", len(data["labels"]), "clusters", data["n_clusters"], "edges", data["edge"].shape[1], flush=True)
        for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
            for cfg in configs():
                feats = feature_subset(data["features"], cfg.variant)
                local_cfg = cfg
                if cfg.variant in {"RNA_only", "ADT_only"}:
                    local_cfg = RunConfig(cfg.variant, cfg.encoder, "view_specific", cfg.sampling, (1.0,))
                print("RUN", dataset_name, seed, local_cfg, flush=True)
                started = time.time()
                emb, elapsed, peak = train_model(
                    features=feats,
                    edge=data["edge"],
                    config=local_cfg,
                    seed=seed,
                    epochs=args.epochs,
                    train_batch_size=args.train_batch_size,
                    infer_batch_size=args.infer_batch_size,
                    device=device,
                )
                row = {
                    "dataset": dataset_name,
                    "n_obs": len(data["labels"]),
                    "n_clusters": data["n_clusters"],
                    "seed": seed,
                    "variant": cfg.variant,
                    "encoder": local_cfg.encoder,
                    "view_mode": local_cfg.view_mode,
                    "sampling": local_cfg.sampling,
                    "weights": str(local_cfg.weights),
                    "elapsed_sec": elapsed,
                    "wall_sec": time.time() - started,
                    "peak_memory_mib": peak,
                }
                row.update(score(emb, data["labels"], data["n_clusters"]))
                rows.append(row)
                pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "smahd_realdata_ablation_raw.csv"), index=False)
                print(row, flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(args.output_dir, "smahd_realdata_ablation_raw.csv"), index=False)
    summary = raw.groupby(["dataset", "variant"])[["ARI", "NMI", "AMI", "FMI", "elapsed_sec", "peak_memory_mib"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.output_dir, "smahd_realdata_ablation_summary.csv"), index=False)
    plot_summary(summary, args.output_dir)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
