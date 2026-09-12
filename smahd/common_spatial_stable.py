import json
import hashlib
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader

from .layer import (
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
from .train_cluster import soft_graph_contrastive_loss


ENCODERS = {
    "GAT": (GATConv_Encoder, GATConv_Decoder),
    "GCN": (GCNConv_Encoder, GCNConv_Decoder),
    "SAGE": (SAGEConv_Encoder, SAGEConv_Decoder),
    "GraphConv": (GraphConv_Encoder, GraphConv_Decoder),
}


@dataclass(frozen=True)
class RunConfig:
    variant: str
    encoder: str = "GAT"
    view_mode: str = "view_specific"
    sampling: str = "cluster"
    weights: tuple = (0.45, 0.45, 0.10)
    selected_views: tuple = (0, 1, 2)
    graph_weight: float = 0.1
    spatial_weight: float = 0.0
    geometry_weight: float = 0.0
    keep_inter_cluster_edges: bool = False


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def score_embedding(embedding, labels, seed=0):
    prediction = KMeans(
        n_clusters=np.unique(labels).size,
        n_init=20,
        random_state=seed,
    ).fit_predict(embedding)
    return {
        "ARI": float(adjusted_rand_score(labels, prediction)),
        "NMI": float(normalized_mutual_info_score(labels, prediction)),
        "AMI": float(adjusted_mutual_info_score(labels, prediction)),
        "FMI": float(fowlkes_mallows_score(labels, prediction)),
    }, prediction


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _make_loader(
    data,
    sampling,
    batch_size,
    seed,
    partition_dir=None,
    keep_inter_cluster_edges=False,
):
    if sampling == "full":
        return [data], {"sampling": "full"}
    parts = max(1, int(math.ceil(data.num_nodes / batch_size)) * 10)
    save_dir = None
    if partition_dir is not None:
        save_dir = str(Path(partition_dir))
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    clustered = ClusterData(
        data,
        num_parts=parts,
        recursive=False,
        save_dir=save_dir,
        log=False,
        keep_inter_cluster_edges=keep_inter_cluster_edges,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = ClusterLoader(
        clustered,
        batch_size=min(10, parts),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    cache_files = []
    if save_dir is not None:
        for path in sorted(Path(save_dir).glob("*")):
            if path.is_file():
                cache_files.append({
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                })
    return loader, {
        "sampling": "cluster",
        "num_parts": parts,
        "loader_batch_parts": min(10, parts),
        "loader_seed": seed,
        "partition_dir": save_dir,
        "keep_inter_cluster_edges": keep_inter_cluster_edges,
        "partition_cache_files": cache_files,
    }


def spatial_consistency_loss(z, edge_index):
    """Penalize normalized embedding differences across observed spatial edges."""
    source, target = edge_index
    valid = source != target
    source = source[valid]
    target = target[valid]
    if source.numel() == 0:
        return z.new_zeros(())
    normalized = F.normalize(z, p=2, dim=1)
    return (
        (normalized[source] - normalized[target]).pow(2).sum(dim=1).mean()
    )


def input_geometry_loss(z, input_views, edge_index, max_pairs=50000):
    """Preserve multimodal pairwise similarity without using labels."""
    source, target = edge_index
    valid = source != target
    source = source[valid]
    target = target[valid]

    random_count = min(max_pairs, max(1, z.shape[0] * 4))
    random_source = torch.randint(0, z.shape[0], (random_count,), device=z.device)
    random_target = torch.randint(0, z.shape[0], (random_count,), device=z.device)
    random_valid = random_source != random_target
    source = torch.cat([source[:max_pairs], random_source[random_valid]])
    target = torch.cat([target[:max_pairs], random_target[random_valid]])
    if source.numel() == 0:
        return z.new_zeros(())

    normalized_z = F.normalize(z, p=2, dim=1)
    embedding_similarity = (normalized_z[source] * normalized_z[target]).sum(dim=1)
    with torch.no_grad():
        view_similarities = []
        for view in input_views:
            normalized_view = F.normalize(view, p=2, dim=1)
            view_similarities.append(
                (normalized_view[source] * normalized_view[target]).sum(dim=1)
            )
        target_similarity = torch.stack(view_similarities, dim=0).mean(dim=0)
    return F.smooth_l1_loss(embedding_similarity, target_similarity)


def train_model(
    all_features,
    edge,
    config,
    seed,
    epochs,
    train_batch_size,
    infer_batch_size,
    device,
    emb_dim=64,
    lr=0.001,
    weight_decay=1e-6,
    partition_dir=None,
):
    set_seed(seed)
    features = [all_features[index] for index in config.selected_views]
    if config.view_mode == "concat":
        features = [torch.cat(features, dim=1)]
        weights = [1.0]
    else:
        weights = list(config.weights)
    if len(features) != len(weights):
        raise ValueError(f"Feature/weight mismatch for {config.variant}")

    encoder, decoder = ENCODERS[config.encoder]
    hidden_dims = [feature.shape[1] for feature in features] + [emb_dim]
    model = SMAHD(hidden_dims, device, encoder, decoder).to(device)
    data = Data(x=torch.cat(features, dim=1), edge_index=edge)
    loader, loader_metadata = _make_loader(
        data,
        config.sampling,
        train_batch_size,
        seed,
        partition_dir=partition_dir,
        keep_inter_cluster_edges=config.keep_inter_cluster_edges,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    updates = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_reconstruction = 0.0
        total_spatial = 0.0
        total_geometry = 0.0
        batches = 0
        for batch in loader:
            batch = batch.to(device)
            split = torch.split(batch.x, hidden_dims[:-1], dim=1)
            optimizer.zero_grad(set_to_none=True)
            z, reconstructions = model(split, batch.edge_index)
            reconstruction = z.new_zeros(())
            for weight, original, reconstructed in zip(weights, split, reconstructions):
                reconstruction = reconstruction + weight * F.mse_loss(reconstructed, original)
            graph = (
                soft_graph_contrastive_loss(
                    z,
                    batch.edge_index,
                    split,
                    temperature=0.5,
                    max_pairs=100000,
                )
                if config.graph_weight > 0
                else z.new_zeros(())
            )
            spatial = (
                spatial_consistency_loss(z, batch.edge_index)
                if config.spatial_weight > 0
                else z.new_zeros(())
            )
            geometry = (
                input_geometry_loss(z, split, batch.edge_index)
                if config.geometry_weight > 0
                else z.new_zeros(())
            )
            loss = (
                reconstruction
                + config.graph_weight * graph
                + config.spatial_weight * spatial
                + config.geometry_weight * geometry
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_reconstruction += float(reconstruction.detach().cpu())
            total_spatial += float(spatial.detach().cpu())
            total_geometry += float(geometry.detach().cpu())
            batches += 1
            updates += 1
        history.append({
            "epoch": epoch,
            "loss": total_loss / max(1, batches),
            "reconstruction_loss": total_reconstruction / max(1, batches),
            "spatial_consistency_loss": total_spatial / max(1, batches),
            "input_geometry_loss": total_geometry / max(1, batches),
        })

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_memory_mib = (
        float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
        if torch.cuda.is_available()
        else None
    )

    model.to("cpu").eval()
    cpu_data = data.to("cpu")
    infer_started = time.perf_counter()
    inference_loader = NeighborLoader(
        cpu_data,
        num_neighbors=[-1],
        batch_size=infer_batch_size,
        shuffle=False,
    )
    embeddings = []
    with torch.no_grad():
        for batch in inference_loader:
            split = torch.split(batch.x, hidden_dims[:-1], dim=1)
            z, _ = model(split, batch.edge_index)
            embeddings.append(z[: batch.batch_size].cpu().numpy())
    embedding = np.vstack(embeddings)
    inference_seconds = time.perf_counter() - infer_started
    metadata = {
        "config": asdict(config),
        "epochs": epochs,
        "updates": updates,
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "model_seconds": training_seconds + inference_seconds,
        "peak_gpu_memory_mib": peak_memory_mib,
        "registered_parameters": int(sum(p.numel() for p in model.parameters())),
        "history": history,
        "lr": lr,
        "weight_decay": weight_decay,
        "loader": loader_metadata,
    }
    del model, loader, data, cpu_data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embedding, metadata


def write_incremental(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(value, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
