import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader

from .layer import SMAHD


def _forward_with_views(model, features, edge_index):
    views = [
        encoder(feature, edge_index)
        for encoder, feature in zip(model.encoders, features)
    ]
    fused = views[0] if len(views) == 1 else torch.cat(views, dim=1)
    z = model.fc(fused)
    reconstructions = [decoder(z, edge_index) for decoder in model.decoders]
    return z, reconstructions, views


def _sample_edges(edge_index, max_pairs):
    source, target = edge_index
    valid = source != target
    source = source[valid]
    target = target[valid]
    if source.numel() > max_pairs:
        selection = torch.randperm(source.numel(), device=source.device)[:max_pairs]
        source = source[selection]
        target = target[selection]
    return source, target


def soft_graph_contrastive_loss(
    z,
    edge_index,
    input_views,
    temperature=0.5,
    negative_ratio=1.0,
    max_pairs=100000,
):
    source, target = _sample_edges(edge_index, max_pairs)
    if source.numel() == 0:
        return z.new_zeros(())

    z_normalized = F.normalize(z, p=2, dim=1)
    positive_logits = (
        z_normalized[source] * z_normalized[target]
    ).sum(dim=1) / temperature

    with torch.no_grad():
        view_confidences = []
        for view in input_views:
            view_normalized = F.normalize(view, p=2, dim=1)
            similarity = (
                view_normalized[source] * view_normalized[target]
            ).sum(dim=1)
            view_confidences.append((similarity + 1.0) * 0.5)
        confidence = torch.stack(view_confidences, dim=0).mean(dim=0)
        confidence = confidence.clamp(0.05, 1.0)

    positive_loss = -(
        confidence * F.logsigmoid(positive_logits)
    ).sum() / confidence.sum().clamp_min(1e-8)

    negative_count = max(1, int(source.numel() * negative_ratio))
    negative_source = torch.randint(
        0, z.shape[0], (negative_count,), device=z.device
    )
    negative_target = torch.randint(
        0, z.shape[0], (negative_count,), device=z.device
    )
    keep = negative_source != negative_target
    negative_source = negative_source[keep]
    negative_target = negative_target[keep]
    negative_logits = (
        z_normalized[negative_source] * z_normalized[negative_target]
    ).sum(dim=1) / temperature
    negative_loss = -F.logsigmoid(-negative_logits).mean()
    return 0.5 * (positive_loss + negative_loss)


def view_alignment_loss(view_embeddings):
    if len(view_embeddings) < 2:
        return view_embeddings[0].new_zeros(())
    normalized = [F.normalize(view, p=2, dim=1) for view in view_embeddings]
    losses = []
    for first in range(len(normalized)):
        for second in range(first + 1, len(normalized)):
            losses.append(
                1.0 - (normalized[first] * normalized[second]).sum(dim=1).mean()
            )
    return torch.stack(losses).mean()


def variance_floor_loss(z, target_std=1.0):
    if z.shape[0] < 2:
        return z.new_zeros(())
    standard_deviation = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    return F.relu(target_std - standard_deviation).mean()


def _initialize_cluster_centers(z, n_clusters, seed, max_samples=50000):
    values = z.detach().cpu().numpy()
    if values.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        selected = rng.choice(values.shape[0], max_samples, replace=False)
        values = values[selected]
    estimator = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=20,
        batch_size=min(4096, values.shape[0]),
        random_state=seed,
    )
    estimator.fit(values)
    return torch.as_tensor(estimator.cluster_centers_, dtype=z.dtype, device=z.device)


def prototype_clustering_loss(z, centers, alpha=1.0):
    squared_distance = torch.sum(
        (z.unsqueeze(1) - centers.unsqueeze(0)) ** 2,
        dim=2,
    )
    q = (1.0 + squared_distance / alpha) ** (-(alpha + 1.0) / 2.0)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
    frequency = q.sum(dim=0).clamp_min(1e-8)
    target = (q.detach() ** 2) / frequency.detach()
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
    sharpening = F.kl_div(q.clamp_min(1e-8).log(), target, reduction="batchmean")

    mean_assignment = q.mean(dim=0)
    uniform = torch.full_like(mean_assignment, 1.0 / mean_assignment.numel())
    balance = torch.sum(
        mean_assignment
        * (mean_assignment.clamp_min(1e-8).log() - uniform.log())
    )
    return sharpening + 0.1 * balance


def infer_embeddings(model, data, hidden_dims, batch_size, device):
    loader = NeighborLoader(
        data,
        num_neighbors=[-1],
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    model.to(device)
    embeddings = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            split = torch.split(batch.x, hidden_dims[:-1], dim=1)
            z, _, _ = _forward_with_views(model, split, batch.edge_index)
            embeddings.append(z[: batch.batch_size].cpu())
    return torch.cat(embeddings, dim=0).numpy()


def train_cluster_aware_smahd(
    features,
    edge,
    emb_dim,
    weights,
    n_epochs,
    lr,
    train_batch_size,
    infer_batch_size,
    weight_decay,
    train_device,
    infer_device,
    Conv_Encoder,
    Conv_Decoder,
    graph_weight=0.0,
    alignment_weight=0.0,
    variance_weight=0.0,
    cluster_weight=0.0,
    n_clusters=None,
    cluster_warmup_epochs=20,
    graph_temperature=0.5,
    max_graph_pairs=100000,
    seed=0,
):
    hidden_dims = [feature.shape[1] for feature in features] + [emb_dim]
    model = SMAHD(
        hidden_dims=hidden_dims,
        device=train_device,
        Conv_Encoder=Conv_Encoder,
        Conv_Decoder=Conv_Decoder,
    ).to(train_device)

    data = Data(x=torch.cat(features, dim=1), edge_index=edge)
    number_of_parts = max(
        1,
        int(math.ceil(data.num_nodes / train_batch_size)) * 10,
    )
    cluster_data = ClusterData(
        data,
        num_parts=number_of_parts,
        recursive=False,
        log=False,
    )
    train_loader = ClusterLoader(
        cluster_data,
        batch_size=min(10, number_of_parts),
        shuffle=True,
        num_workers=2,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    if cluster_weight > 0 and not n_clusters:
        raise ValueError("n_clusters is required when cluster_weight is positive")

    history = []
    centers_initialized = False
    started = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        totals = {
            "total": 0.0,
            "reconstruction": 0.0,
            "graph": 0.0,
            "alignment": 0.0,
            "variance": 0.0,
            "cluster": 0.0,
        }
        batch_count = 0
        for batch in train_loader:
            batch = batch.to(train_device)
            split = torch.split(batch.x, hidden_dims[:-1], dim=1)
            optimizer.zero_grad()
            z, reconstructions, views = _forward_with_views(
                model,
                split,
                batch.edge_index,
            )

            reconstruction = z.new_zeros(())
            for index, (original, reconstructed) in enumerate(
                zip(split, reconstructions)
            ):
                reconstruction = reconstruction + weights[index] * F.mse_loss(
                    reconstructed,
                    original,
                )

            graph = (
                soft_graph_contrastive_loss(
                    z,
                    batch.edge_index,
                    split,
                    temperature=graph_temperature,
                    max_pairs=max_graph_pairs,
                )
                if graph_weight > 0
                else z.new_zeros(())
            )
            alignment = (
                view_alignment_loss(views)
                if alignment_weight > 0
                else z.new_zeros(())
            )
            variance = (
                variance_floor_loss(z)
                if variance_weight > 0
                else z.new_zeros(())
            )

            cluster = z.new_zeros(())
            if cluster_weight > 0 and epoch > cluster_warmup_epochs:
                if not centers_initialized:
                    centers = _initialize_cluster_centers(z, n_clusters, seed)
                    model.register_parameter(
                        "cluster_centers",
                        nn.Parameter(centers),
                    )
                    optimizer.add_param_group({"params": [model.cluster_centers]})
                    centers_initialized = True
                cluster = prototype_clustering_loss(z, model.cluster_centers)

            objective = (
                reconstruction
                + graph_weight * graph
                + alignment_weight * alignment
                + variance_weight * variance
                + cluster_weight * cluster
            )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            totals["total"] += float(objective.detach().cpu())
            totals["reconstruction"] += float(reconstruction.detach().cpu())
            totals["graph"] += float(graph.detach().cpu())
            totals["alignment"] += float(alignment.detach().cpu())
            totals["variance"] += float(variance.detach().cpu())
            totals["cluster"] += float(cluster.detach().cpu())
            batch_count += 1

        history.append(
            {
                "epoch": epoch,
                **{
                    name: value / max(batch_count, 1)
                    for name, value in totals.items()
                },
            }
        )

    training_seconds = time.time() - started
    embedding = infer_embeddings(
        model,
        data,
        hidden_dims,
        infer_batch_size,
        infer_device,
    )
    metadata = {
        "training_seconds": training_seconds,
        "registered_parameters": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "cluster_centers_initialized": centers_initialized,
        "history": history,
    }
    return embedding, metadata
