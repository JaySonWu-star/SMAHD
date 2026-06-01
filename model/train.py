from scipy import stats
import torch
from tqdm import tqdm

from layer import SMAHD, GCNConv_Encoder, GCNConv_Decoder, GATConv_Encoder, GATConv_Decoder
import torch.nn.functional as F
import numpy as np

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader


def train_SMAHD(features,
                edge,
                emb_dim=64,
                weights=[1, 1, 1],
                n_epochs=500,
                lr=0.0001,
                train_batch_size=100,
                infer_batch_size=100,
                weight_decay=1e-5,
                train_device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                infer_device=torch.device('cpu'),
                Conv_Encoder=GATConv_Encoder,
                Conv_Decoder=GATConv_Decoder
                ):
    hidden_dims = [x.shape[1] for x in features] + [emb_dim]

    model = SMAHD(hidden_dims=hidden_dims, device=train_device, Conv_Encoder=Conv_Encoder, Conv_Decoder=Conv_Decoder)

    features_cat = torch.cat(features, dim=1)

    data = Data(x=features_cat, edge_index=edge)
    cluster_data = ClusterData(data, num_parts=int(np.ceil(data.num_nodes / train_batch_size)) * 10, recursive=False,
                               log=True)
    train_loader = ClusterLoader(cluster_data, batch_size=10, shuffle=True, num_workers=2)
    subgraph_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=infer_batch_size, shuffle=False)

    model.to(train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_list = []
    loop = tqdm(range(1, n_epochs + 1))
    for epoch in loop:
        model.train()
        loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch.to(train_device)
            edge_index = batch.edge_index.to(train_device)
            x_split = torch.split(batch.x, hidden_dims[:-1], dim=1)
            x_split = [x.to(train_device) for x in x_split]
            z, x_rec = model(x_split, edge_index)

            rec_loss = 0
            for i, (x, x_r) in enumerate(zip(x_split, x_rec)):
                rec_output = F.mse_loss(x, x_r)
                w = weights[i]
                rec_loss += weights[i] * rec_output
            loss += rec_loss.item()
            rec_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        loop.set_description(f'Epoch [{epoch}/{n_epochs}]')
        loop.set_postfix(loss=loss)

    with torch.no_grad():
        model.to(infer_device)
        for encoder in model.encoders:
            encoder.to(infer_device)
        for decoder in model.decoders:
            decoder.to(infer_device)

        z_list = []
        for batch in subgraph_loader:
            batch.to(infer_device)
            edge_index = batch.edge_index.to(infer_device)
            x_split = torch.split(batch.x, hidden_dims[:-1], dim=1)
            x_split = [x.to(infer_device) for x in x_split]
            z, x_rec = model(x_split, edge_index)
            z_list.append(z[:batch.batch_size].cpu())
        z_all = torch.cat(z_list, dim=0).numpy()

    return z_all
