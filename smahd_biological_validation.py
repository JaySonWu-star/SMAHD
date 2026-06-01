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
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from experiments_smahd_ablation_simulated import RunConfig, train_model
from smahd_realdata_ablation import DATASETS, DATA_ROOT, dense_array, pca_features


OUT_DIR = "experiments/biological_validation"
SEED = 101
EPOCHS = 80
TRAIN_BATCH_SIZE = 512
INFER_BATCH_SIZE = 2048

MOUSE_DATASETS = {
    "mouse_brain_atac": "Mouse brain ATAC",
}

MOUSE_MARKERS = [
    "Reln",
    "Cux2",
    "Rorb",
    "Bcl11b",
    "Fezf2",
    "Tle4",
    "Foxp2",
    "Mbp",
    "Plp1",
    "Gad1",
    "Gad2",
    "Aqp4",
]

MOUSE_SPATIAL_MARKERS = ["Reln", "Rorb", "Bcl11b", "Tle4", "Mbp"]

SPLEEN_MARKERS = [
    "CD3",
    "CD4",
    "CD8",
    "CD19",
    "B220_CD45R",
    "CD20",
    "IgM",
    "IgD",
    "F4_80",
    "CD68",
    "CD163",
    "CD31",
    "CD105",
    "MadCAM1",
]

SPLEEN_SPATIAL_MARKERS = ["CD3", "B220_CD45R", "IgM", "F4_80", "CD68", "CD31"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_edge(spatial, n_neighbors=15):
    edge = np.asarray(kneighbors_graph(spatial, n_neighbors=n_neighbors, mode="connectivity", include_self=False).nonzero(), dtype=np.int64)
    edge = np.concatenate([edge, edge[::-1]], axis=1)
    edge = np.unique(edge, axis=1)
    return torch.tensor(edge, dtype=torch.long)


def expression_matrix(adata, markers):
    adata.var_names_make_unique()
    present = [marker for marker in markers if marker in set(map(str, adata.var_names))]
    x = dense_array(adata[:, present].X).astype(np.float32)
    x = np.log1p(np.maximum(x, 0))
    return present, x


def robust_scale(values):
    values = np.asarray(values, dtype=np.float32)
    lo, hi = np.nanpercentile(values, [2, 98])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0, 1)


def cluster_marker_means(clusters, marker_names, expr):
    rows = []
    for cluster in sorted(np.unique(clusters)):
        mask = clusters == cluster
        for idx, marker in enumerate(marker_names):
            rows.append({"cluster": int(cluster), "marker": marker, "mean_expression": float(expr[mask, idx].mean())})
    return pd.DataFrame(rows)


def zscore_table(mean_df):
    table = mean_df.pivot(index="cluster", columns="marker", values="mean_expression").sort_index()
    values = table.to_numpy(dtype=np.float32)
    values = (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-6)
    return pd.DataFrame(values, index=table.index, columns=table.columns)


def plot_marker_heatmap(zscores, title, out_path, cmap="RdBu_r"):
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * zscores.shape[1]), max(3.5, 0.35 * zscores.shape[0])), dpi=220)
    im = ax.imshow(zscores.to_numpy(), aspect="auto", cmap=cmap, vmin=-2, vmax=2)
    ax.set_xticks(np.arange(zscores.shape[1]))
    ax.set_xticklabels(zscores.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(zscores.shape[0]))
    ax.set_yticklabels([f"C{c}" for c in zscores.index], fontsize=8)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Marker")
    ax.set_ylabel("SMAHD domain")
    fig.colorbar(im, ax=ax, shrink=0.75, label="z-scored domain mean")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mouse_spatial(mouse_results, out_path):
    n_rows = len(mouse_results)
    n_cols = 1 + len(MOUSE_SPATIAL_MARKERS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.35 * n_cols, 2.7 * n_rows), dpi=220)
    if n_rows == 1:
        axes = axes[None, :]
    for row_idx, result in enumerate(mouse_results):
        spatial = result["spatial"]
        clusters = result["clusters"]
        marker_names = result["marker_names"]
        expr = result["expr"]
        ax = axes[row_idx, 0]
        ax.scatter(spatial[:, 0], spatial[:, 1], c=clusters, s=3, cmap="tab20", linewidths=0)
        ax.set_title(f"{result['display']}\nSMAHD domains", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        for col_idx, marker in enumerate(MOUSE_SPATIAL_MARKERS, start=1):
            ax = axes[row_idx, col_idx]
            if marker in marker_names:
                vals = robust_scale(expr[:, marker_names.index(marker)])
                ax.scatter(spatial[:, 0], spatial[:, 1], c=vals, s=3, cmap="magma", linewidths=0)
            else:
                ax.text(0.5, 0.5, "not detected", ha="center", va="center", fontsize=8)
            ax.set_title(marker, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def load_mouse_dataset(name):
    cfg = DATASETS[name]
    rna = sc.read_h5ad(cfg["rna"])
    adt = sc.read_h5ad(cfg["adt"])
    combined = sc.read_h5ad(cfg["combined"])
    rna.var_names_make_unique()
    adt.var_names_make_unique()
    combined.var_names_make_unique()

    labels = combined.obs[cfg["label_col"]].astype(str)
    keep = (~labels.str.lower().isin(["nan", "exclude"])) & labels.notna()
    common = [x for x in combined.obs_names[keep] if x in rna.obs_names and x in adt.obs_names]
    combined = combined[common].copy()
    rna = rna[common].copy()
    adt = adt[common].copy()

    rna_key = cfg.get("rna_obsm", "X_RNA_pca")
    other_key = cfg.get("other_obsm", "X_ADT_pca")
    if rna_key in combined.obsm and other_key in combined.obsm:
        rna_feat = np.asarray(combined.obsm[rna_key][:, :30], dtype=np.float32)
        adt_feat = np.asarray(combined.obsm[other_key][:, :30], dtype=np.float32)
    else:
        rna_feat = pca_features(rna, 30, True)
        adt_feat = pca_features(adt, 30, False)
    rna_feat = StandardScaler().fit_transform(rna_feat).astype(np.float32)
    adt_feat = StandardScaler().fit_transform(adt_feat).astype(np.float32)
    spatial = np.asarray(combined.obsm["spatial"], dtype=np.float32)
    return {
        "rna": rna,
        "features": [torch.tensor(rna_feat), torch.tensor(adt_feat)],
        "spatial": spatial,
        "edge": build_edge(spatial),
        "n_clusters": len(np.unique(combined.obs[cfg["label_col"]].astype(str))),
    }


def run_smahd(features, edge, n_clusters, device):
    cfg = RunConfig("SMAHD_full", "GAT", "view_specific", "cluster", (0.4, 0.6))
    emb, elapsed, peak = train_model(
        features=features,
        edge=edge,
        config=cfg,
        seed=SEED,
        epochs=EPOCHS,
        train_batch_size=TRAIN_BATCH_SIZE,
        infer_batch_size=INFER_BATCH_SIZE,
        device=device,
    )
    clusters = KMeans(n_clusters=n_clusters, n_init=30, random_state=0).fit_predict(emb)
    return emb, clusters, elapsed, peak


def run_mouse_validation(device):
    results = []
    metric_rows = []
    marker_rows = []
    for name, display in MOUSE_DATASETS.items():
        print("MOUSE", name, flush=True)
        data = load_mouse_dataset(name)
        emb_path = os.path.join(OUT_DIR, f"{name}_smahd_embedding.npy")
        cluster_path = os.path.join(OUT_DIR, f"{name}_smahd_clusters.csv")
        if os.path.exists(emb_path) and os.path.exists(cluster_path):
            emb = np.load(emb_path)
            clusters = pd.read_csv(cluster_path)["smahd_domain"].to_numpy()
            elapsed = np.nan
            peak = np.nan
        else:
            emb, clusters, elapsed, peak = run_smahd(data["features"], data["edge"], data["n_clusters"], device)
            np.save(emb_path, emb)
            pd.DataFrame({"spot": data["rna"].obs_names, "smahd_domain": clusters}).to_csv(cluster_path, index=False)
        markers, expr = expression_matrix(data["rna"], MOUSE_MARKERS)
        means = cluster_marker_means(clusters, markers, expr)
        means["dataset"] = name
        marker_rows.append(means)
        metric_rows.append(
            {
                "dataset": name,
                "display": display,
                "n_spots": data["rna"].n_obs,
                "n_domains": data["n_clusters"],
                "seed": SEED,
                "epochs": EPOCHS,
                "elapsed_sec": elapsed,
                "peak_memory_mib": peak,
            }
        )
        results.append({"name": name, "display": display, "spatial": data["spatial"], "clusters": clusters, "marker_names": markers, "expr": expr})

    all_means = pd.concat(marker_rows, ignore_index=True)
    all_means.to_csv(os.path.join(OUT_DIR, "mouse_brain_smahd_marker_means.csv"), index=False)
    pd.DataFrame(metric_rows).to_csv(os.path.join(OUT_DIR, "mouse_brain_biological_validation_runs.csv"), index=False)
    plot_mouse_spatial(results, os.path.join(OUT_DIR, "smahd_mouse_brain_marker_spatial.png"))
    for name, display in MOUSE_DATASETS.items():
        zscores = zscore_table(all_means[all_means["dataset"] == name])
        zscores.to_csv(os.path.join(OUT_DIR, f"{name}_marker_zscores.csv"))
        plot_marker_heatmap(zscores, f"{display}: RNA marker enrichment by SMAHD domain", os.path.join(OUT_DIR, f"{name}_marker_heatmap.png"))


def pca_no_hvg(adata, n_comps=30, log_norm=True):
    adata = adata.copy()
    adata.var_names_make_unique()
    if log_norm:
        sc.pp.filter_genes(adata, min_cells=10)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    x = dense_array(adata.X).astype(np.float32)
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)
    n = min(n_comps, x.shape[0] - 1, x.shape[1])
    return PCA(n_components=n, random_state=0).fit_transform(x).astype(np.float32)


def run_spleen_validation(device):
    print("SPLEEN Mouse_Spleen1", flush=True)
    spleen_dir = os.path.join(DATA_ROOT, "SMART", "datasets", "Mouse_Spleen1")
    rna = sc.read_h5ad(os.path.join(spleen_dir, "adata_RNA.h5ad"))
    adt = sc.read_h5ad(os.path.join(spleen_dir, "adata_ADT.h5ad"))
    rna.var_names_make_unique()
    adt.var_names_make_unique()
    common = [x for x in rna.obs_names if x in adt.obs_names]
    rna = rna[common].copy()
    adt = adt[common].copy()
    spatial = np.asarray(rna.obsm["spatial"], dtype=np.float32)
    rna_feat = pca_no_hvg(rna, 30, True)
    adt_feat = pca_no_hvg(adt, 21, False)
    features = [torch.tensor(rna_feat), torch.tensor(adt_feat)]
    edge = build_edge(spatial)
    emb_path = os.path.join(OUT_DIR, "mouse_spleen1_smahd_embedding.npy")
    cluster_path = os.path.join(OUT_DIR, "mouse_spleen1_smahd_clusters.csv")
    if os.path.exists(emb_path) and os.path.exists(cluster_path):
        emb = np.load(emb_path)
        clusters = pd.read_csv(cluster_path)["smahd_domain"].to_numpy()
    else:
        emb, clusters, _, _ = run_smahd(features, edge, 7, device)
        np.save(emb_path, emb)
        pd.DataFrame({"spot": rna.obs_names, "smahd_domain": clusters}).to_csv(cluster_path, index=False)
    markers, expr = expression_matrix(adt, SPLEEN_MARKERS)
    means = cluster_marker_means(clusters, markers, expr)
    means.to_csv(os.path.join(OUT_DIR, "mouse_spleen1_adt_marker_means.csv"), index=False)
    zscores = zscore_table(means)
    zscores.to_csv(os.path.join(OUT_DIR, "mouse_spleen1_adt_marker_zscores.csv"))
    plot_marker_heatmap(zscores, "Mouse spleen: ADT marker enrichment by SMAHD domain", os.path.join(OUT_DIR, "smahd_mouse_spleen_adt_marker_heatmap.png"))

    n_cols = 1 + len(SPLEEN_SPATIAL_MARKERS)
    fig, axes = plt.subplots(1, n_cols, figsize=(2.25 * n_cols, 2.6), dpi=220)
    axes[0].scatter(spatial[:, 0], spatial[:, 1], c=clusters, s=5, cmap="tab20", linewidths=0)
    axes[0].set_title("SMAHD domains", fontsize=9)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for idx, marker in enumerate(SPLEEN_SPATIAL_MARKERS, start=1):
        ax = axes[idx]
        if marker in markers:
            vals = robust_scale(expr[:, markers.index(marker)])
            ax.scatter(spatial[:, 0], spatial[:, 1], c=vals, s=5, cmap="viridis", linewidths=0)
        else:
            ax.text(0.5, 0.5, "not detected", ha="center", va="center", fontsize=8)
        ax.set_title(marker, fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "smahd_mouse_spleen_adt_marker_spatial.png"))
    plt.close(fig)


def write_report():
    lines = [
        "# SMAHD biological validation",
        "",
        "This analysis adds marker-level biological support for the real-data results.",
        "",
        "Mouse brain validation uses RNA expression of canonical layer and cell-type markers (Reln, Cux2, Rorb, Bcl11b, Fezf2, Tle4, Foxp2, Mbp, Plp1, Gad1, Gad2 and Aqp4) over SMAHD domains inferred from RNA+ATAC data.",
        "",
        "Mouse spleen validation uses antibody-derived tags from Mouse_Spleen1. T-cell markers (CD3, CD4, CD8), B-cell/follicle markers (CD19, B220/CD45R, CD20, IgM, IgD), macrophage/red-pulp markers (F4/80, CD68, CD163) and vascular/stromal markers (CD31, CD105, MadCAM1) were summarized by SMAHD domain.",
        "",
        "These marker analyses should be reported as biological plausibility checks rather than supervised validation, because the markers were not used for model training and the SMAHD domains were inferred in an unsupervised manner.",
    ]
    with open(os.path.join(OUT_DIR, "smahd_biological_validation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ensure_dir(OUT_DIR)
    start = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device, "seed", SEED, "epochs", EPOCHS, flush=True)
    run_mouse_validation(device)
    run_spleen_validation(device)
    write_report()
    print("done", OUT_DIR, "seconds", time.time() - start, flush=True)


if __name__ == "__main__":
    main()
