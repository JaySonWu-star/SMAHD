import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))

from layer import (
    GATConv_Decoder,
    GATConv_Encoder,
    GCNConv_Decoder,
    GCNConv_Encoder,
    GraphConv_Decoder,
    GraphConv_Encoder,
    SAGEConv_Decoder,
    SAGEConv_Encoder,
    SMAHD,
)


ENCODERS = {
    "GAT": (GATConv_Encoder, GATConv_Decoder),
    "GCN": (GCNConv_Encoder, GCNConv_Decoder),
    "SAGE": (SAGEConv_Encoder, SAGEConv_Decoder),
    "GraphConv": (GraphConv_Encoder, GraphConv_Decoder),
}


@dataclass
class RunConfig:
    variant: str
    encoder: str = "GAT"
    view_mode: str = "view_specific"
    sampling: str = "cluster"
    weights: tuple = (0.4, 0.4, 0.2)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_spatial_multiomics(seed, n_per_domain=900, n_features=(30, 30, 120)):
    set_seed(seed)
    centers = np.array([[0.0, 0.0], [2.2, 0.25], [1.1, 1.9], [3.25, 2.1]], dtype=np.float32)
    labels = []
    coords = []
    for label, center in enumerate(centers):
        local = np.random.normal(loc=center, scale=0.9, size=(n_per_domain, 2))
        coords.append(local)
        labels.extend([label] * n_per_domain)
    coords = np.vstack(coords).astype(np.float32)
    labels = np.asarray(labels)
    order = np.random.permutation(coords.shape[0])
    coords = coords[order]
    labels = labels[order]

    views = []
    for view_idx, dim in enumerate(n_features):
        domain_effect = np.random.normal(0, 1, size=(len(centers), dim)).astype(np.float32)
        if view_idx == 2:
            # A high-dimensional weak/noisy view mimics sparse epigenomic peaks or low-quality ADTs.
            # It contains a partly discordant structure, so equal weighting should be less robust.
            noisy_labels = (labels + np.random.randint(0, len(centers), size=labels.shape)) % len(centers)
            shared = 0.35 * domain_effect[labels] + 0.65 * domain_effect[noisy_labels]
        else:
            shared = domain_effect[labels]
        spatial_signal = np.stack(
            [
                np.sin(coords[:, 0] * (view_idx + 1)),
                np.cos(coords[:, 1] * (view_idx + 1)),
            ],
            axis=1,
        ).astype(np.float32)
        loadings = np.random.normal(0, 0.35, size=(2, dim)).astype(np.float32)
        x = shared + spatial_signal @ loadings
        noise_scale = [0.75, 0.9, 1.45][view_idx]
        x += np.random.normal(0, noise_scale, size=x.shape).astype(np.float32)

        # Mimic modality-specific sparsity; RNA-like view is sparse, ADT-like view is denser.
        dropout = [0.45, 0.18, 0.68][view_idx]
        mask = np.random.binomial(1, 1 - dropout, size=x.shape).astype(np.float32)
        x = x * mask
        x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
        views.append(torch.tensor(x, dtype=torch.float32))

    edge_mat = kneighbors_graph(coords, n_neighbors=15, mode="connectivity", include_self=False)
    edge = np.asarray(edge_mat.nonzero(), dtype=np.int64)
    edge = np.concatenate([edge, edge[::-1]], axis=1)
    edge = np.unique(edge, axis=1)
    return coords, labels, views, torch.tensor(edge, dtype=torch.long)


def train_model(features, edge, config, seed, epochs, train_batch_size, infer_batch_size, device):
    set_seed(seed)
    enc_cls, dec_cls = ENCODERS[config.encoder]
    if config.view_mode == "concat":
        train_features = [torch.cat(features, dim=1)]
        weights = [1.0]
    else:
        train_features = features
        weights = list(config.weights)

    hidden_dims = [x.shape[1] for x in train_features] + [64]
    model = SMAHD(hidden_dims=hidden_dims, device=device, Conv_Encoder=enc_cls, Conv_Decoder=dec_cls)
    features_cat = torch.cat(train_features, dim=1)
    data = Data(x=features_cat, edge_index=edge)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    model.to(device)

    if config.sampling == "cluster":
        cluster_data = ClusterData(
            data,
            num_parts=int(np.ceil(data.num_nodes / train_batch_size)) * 10,
            recursive=False,
            log=False,
        )
        train_loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True, num_workers=0)
    elif config.sampling == "full":
        train_loader = [data]
    else:
        raise ValueError(config.sampling)

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            x_split = torch.split(batch.x, hidden_dims[:-1], dim=1)
            x_split = [x.to(device) for x in x_split]
            z, x_rec = model(x_split, batch.edge_index.to(device))
            loss = 0
            for idx, (x, x_r) in enumerate(zip(x_split, x_rec)):
                loss = loss + weights[idx] * F.mse_loss(x, x_r)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

    elapsed = time.time() - start_time
    peak_memory_mib = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024 if torch.cuda.is_available() else np.nan
    )

    model.eval()
    model.to("cpu")
    for encoder in model.encoders:
        encoder.to("cpu")
    for decoder in model.decoders:
        decoder.to("cpu")
    cpu_data = data.to("cpu")
    subgraph_loader = NeighborLoader(cpu_data, num_neighbors=[-1], batch_size=infer_batch_size, shuffle=False)
    embeddings = []
    with torch.no_grad():
        for batch in subgraph_loader:
            x_split = torch.split(batch.x, hidden_dims[:-1], dim=1)
            z, _ = model(list(x_split), batch.edge_index)
            embeddings.append(z[: batch.batch_size].cpu().numpy())
    return np.vstack(embeddings), elapsed, peak_memory_mib


def score_embedding(embedding, labels):
    pred = KMeans(n_clusters=len(np.unique(labels)), n_init=20, random_state=0).fit_predict(embedding)
    return {
        "ARI": adjusted_rand_score(labels, pred),
        "NMI": normalized_mutual_info_score(labels, pred),
        "AMI": adjusted_mutual_info_score(labels, pred),
        "FMI": fowlkes_mallows_score(labels, pred),
        "V_measure": v_measure_score(labels, pred),
        "homogeneity": homogeneity_score(labels, pred),
    }


def build_configs():
    return [
        RunConfig("SMAHD_full", "GAT", "view_specific", "cluster", (0.45, 0.45, 0.10)),
        RunConfig("equal_weights", "GAT", "view_specific", "cluster", (1.0, 1.0, 1.0)),
        RunConfig("early_concat_single_encoder", "GAT", "concat", "cluster", (1.0,)),
        RunConfig("full_batch_no_microcluster", "GAT", "view_specific", "full", (0.45, 0.45, 0.10)),
        RunConfig("GCN_encoder", "GCN", "view_specific", "cluster", (0.45, 0.45, 0.10)),
        RunConfig("SAGE_encoder", "SAGE", "view_specific", "cluster", (0.45, 0.45, 0.10)),
        RunConfig("GraphConv_encoder", "GraphConv", "view_specific", "cluster", (0.45, 0.45, 0.10)),
    ]


def plot_results(summary, output_dir):
    order = [
        "SMAHD_full",
        "equal_weights",
        "early_concat_single_encoder",
        "full_batch_no_microcluster",
        "GCN_encoder",
        "SAGE_encoder",
        "GraphConv_encoder",
    ]
    summary = summary.set_index("variant").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=180)
    axes[0].bar(summary["variant"], summary["ARI_mean"], yerr=summary["ARI_std"], color="#4C78A8")
    axes[0].set_ylabel("ARI")
    axes[0].set_ylim(0, 1)
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[1].bar(summary["variant"], summary["elapsed_sec_mean"], yerr=summary["elapsed_sec_std"], color="#F58518")
    axes[1].set_ylabel("Runtime (s)")
    axes[1].tick_params(axis="x", rotation=45, labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "smahd_simulated_ablation_summary.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/ablation_simulated")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--n-per-domain", type=int, default=900)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--infer-batch-size", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device", device)
    print("seeds", seeds)
    print("epochs", args.epochs)

    rows = []
    configs = build_configs()
    for data_seed in seeds:
        coords, labels, features, edge = make_spatial_multiomics(data_seed, n_per_domain=args.n_per_domain)
        np.savez_compressed(
            os.path.join(args.output_dir, f"simulated_dataset_seed{data_seed}.npz"),
            coords=coords,
            labels=labels,
            edge=edge.numpy(),
        )
        for cfg in configs:
            print("RUN", data_seed, cfg)
            emb, elapsed, peak_mib = train_model(
                features=features,
                edge=edge,
                config=cfg,
                seed=data_seed,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                infer_batch_size=args.infer_batch_size,
                device=device,
            )
            metrics = score_embedding(emb, labels)
            row = {
                "seed": data_seed,
                "variant": cfg.variant,
                "encoder": cfg.encoder,
                "view_mode": cfg.view_mode,
                "sampling": cfg.sampling,
                "weights": json.dumps(cfg.weights),
                "elapsed_sec": elapsed,
                "peak_memory_mib": peak_mib,
            }
            row.update(metrics)
            rows.append(row)
            pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "smahd_simulated_ablation_raw.csv"), index=False)
            np.save(os.path.join(args.output_dir, f"embedding_{cfg.variant}_seed{data_seed}.npy"), emb)
            print(row)

    raw = pd.DataFrame(rows)
    metric_cols = ["ARI", "NMI", "AMI", "FMI", "V_measure", "homogeneity", "elapsed_sec", "peak_memory_mib"]
    summary = raw.groupby("variant")[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(args.output_dir, "smahd_simulated_ablation_summary.csv"), index=False)
    plot_results(summary, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
