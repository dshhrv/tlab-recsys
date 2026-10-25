import torch
from torch import nn
import numpy as np
import math
from tgn.model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder,
                 time_encoder, n_layers, n_node_features, n_edge_features,
                 n_time_features, embedding_dimension, device, dropout,
                 use_edge_features):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.memory = memory
        self.neighbor_finder = neighbor_finder
        self.time_encoder = time_encoder
        self.n_layers = n_layers
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features
        self.n_time_features = n_time_features
        self.dropout = dropout
        self.embedding_dimension = embedding_dimension
        self.device = device
        self.use_edge_features = use_edge_features

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20, time_diffs=None, use_time_proj=True):
        return NotImplemented


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20, time_diffs=None, use_time_proj=True):
        return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder,
                 time_encoder, n_layers, n_node_features, n_edge_features,
                 n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1,
                 use_edge_features=True):
        super(TimeEmbedding, self).__init__(
            node_features, edge_features, memory, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features,
            n_time_features, embedding_dimension, device, dropout,
            use_edge_features
        )

        class NormalLinear(nn.Linear):
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.n_node_features)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20, time_diffs=None, use_time_proj=True):
        time_proj = self.embedding_layer(time_diffs.unsqueeze(1))
        source_embeddings = memory[source_nodes, :] * (1 + time_proj)
        return source_embeddings


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder,
                 time_encoder, n_layers, n_node_features, n_edge_features,
                 n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True,
                 use_edge_features=True):
        super(GraphEmbedding, self).__init__(
            node_features, edge_features, memory, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features,
            n_time_features, embedding_dimension, device, dropout,
            use_edge_features
        )
        self.use_memory = use_memory
        self.device = device

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers,
                          n_neighbors=20, time_diffs=None, use_time_proj=True):
        assert (n_layers >= 0)

        source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
        timestamps_torch = torch.unsqueeze(
            torch.tensor(
                timestamps, dtype=torch.float32, device=self.device
            ),
            dim=1
        )

        source_nodes_time_embedding = self.time_encoder(
            torch.zeros_like(timestamps_torch)
        )

        source_node_features = self.node_features[source_nodes_torch, :]

        if self.use_memory:
            source_node_features = memory[source_nodes, :] + source_node_features

        if n_layers == 0:
            return source_node_features
        else:
            source_node_conv_embeddings = self.compute_embedding(
                memory, source_nodes, timestamps,
                n_layers=n_layers - 1, n_neighbors=n_neighbors
            )

            neighbors, edge_idxs, edge_times = (
                self.neighbor_finder.get_temporal_neighbor(
                    source_nodes, timestamps, n_neighbors=n_neighbors
                )
            )

            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
            edge_deltas = timestamps[:, np.newaxis] - edge_times
            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

            neighbors_flat = neighbors.flatten()
            neighbor_embeddings = self.compute_embedding(
                memory, neighbors_flat, np.repeat(timestamps, n_neighbors),
                n_layers=n_layers - 1, n_neighbors=n_neighbors
            )

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(
                len(source_nodes), effective_n_neighbors, -1
            )
            edge_time_embeddings = self.time_encoder(edge_deltas_torch)
            edge_features = None
            if self.use_edge_features:
                edge_idxs_torch = torch.from_numpy(edge_idxs).long().to(self.device)
                edge_features = self.edge_features[edge_idxs_torch, :]
            mask = neighbors_torch == 0

            source_embedding = self.aggregate(
                n_layers, source_node_conv_embeddings,
                source_nodes_time_embedding, neighbor_embeddings,
                edge_time_embeddings, edge_features, mask
            )

            return source_embedding

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder,
                 time_encoder, n_layers, n_node_features, n_edge_features,
                 n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True,
                 use_edge_features=True):
        super(GraphSumEmbedding, self).__init__(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory, neighbor_finder=neighbor_finder,
            time_encoder=time_encoder, n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features, embedding_dimension=embedding_dimension,
            device=device, n_heads=n_heads, dropout=dropout, use_memory=use_memory,
            use_edge_features=use_edge_features
        )
        self.linear_1 = torch.nn.ModuleList([
            torch.nn.Linear(embedding_dimension + n_time_features +
                            (n_edge_features if use_edge_features else 0),
                            embedding_dimension)
            for _ in range(n_layers)
        ])
        self.linear_2 = torch.nn.ModuleList([
            torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                            embedding_dimension)
            for _ in range(n_layers)
        ])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        features = [neighbor_embeddings, edge_time_embeddings]
        if self.use_edge_features:
            features.append(edge_features)
        neighbors_features = torch.cat(features, dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat(
            [source_node_features, source_nodes_time_embedding.squeeze()], dim=1
        )
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class NGCFGraphEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder,
                 time_encoder, n_layers, n_node_features, n_edge_features,
                 n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True,
                 use_edge_features=True):
        super(NGCFGraphEmbedding, self).__init__(
            node_features=node_features,
            edge_features=edge_features,
            memory=memory, neighbor_finder=neighbor_finder,
            time_encoder=time_encoder, n_layers=n_layers,
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_time_features=n_time_features, embedding_dimension=embedding_dimension,
            device=device, n_heads=n_heads, dropout=dropout, use_memory=use_memory,
            use_edge_features=use_edge_features
        )
        self.W1 = torch.nn.ModuleList([
            torch.nn.Linear(embedding_dimension, embedding_dimension)
            for _ in range(n_layers)
        ])
        self.W2 = torch.nn.ModuleList([
            torch.nn.Linear(embedding_dimension, embedding_dimension)
            for _ in range(n_layers)
        ])
        self.leaky_relu = torch.nn.LeakyReLU()

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        sum_embeddings = torch.sum(neighbor_embeddings, dim=1)
        sum_transformed = self.W1[n_layer - 1](sum_embeddings)
        source_transformed = self.W2[n_layer - 1](source_node_features)
        bi_interaction = sum_transformed * source_transformed
        return self.leaky_relu(bi_interaction)


class LightGCNEmbedding(GraphEmbedding):
    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        return torch.sum(neighbor_embeddings, dim=1)


class UltraGCNEmbedding(GraphEmbedding):
    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        sum_neighbors = torch.sum(neighbor_embeddings, dim=1)
        return sum_neighbors + source_node_features


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder,
                 time_encoder, n_layers, n_node_features, n_edge_features,
                 n_time_features, embedding_dimension, device,
                 n_heads=2, dropout=0.1, use_memory=True,
                 use_edge_features=True):
        super(GraphAttentionEmbedding, self).__init__(
            node_features, edge_features, memory, neighbor_finder,
            time_encoder, n_layers, n_node_features, n_edge_features,
            n_time_features, embedding_dimension, device,
            n_heads, dropout, use_memory, use_edge_features
        )

        self.attention_models = torch.nn.ModuleList([
            TemporalAttentionLayer(
                n_node_features=n_node_features,
                n_neighbors_features=n_node_features,
                n_edge_features=n_edge_features,
                time_dim=n_time_features,
                n_head=n_heads,
                dropout=dropout,
                output_dimension=n_node_features,
                use_edge_features=use_edge_features
            )
            for _ in range(n_layers)
        ])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings, edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]
        source_embedding, _ = attention_model(
            source_node_features, source_nodes_time_embedding,
            neighbor_embeddings, edge_time_embeddings, edge_features, mask
        )
        return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory,
                         neighbor_finder, time_encoder, n_layers,
                         n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device, n_heads=2,
                         dropout=0.1, n_neighbors=None, use_memory=True,
                         use_edge_features=True):
    params = {
        "node_features": node_features,
        "edge_features": edge_features,
        "memory": memory, "neighbor_finder": neighbor_finder,
        "time_encoder": time_encoder, "n_layers": n_layers,
        "n_node_features": n_node_features,
        "n_edge_features": n_edge_features,
        "n_time_features": n_time_features, "embedding_dimension": embedding_dimension,
        "device": device, "dropout": dropout,
        "use_edge_features": use_edge_features
    }

    if module_type == "graph_attention":
        return GraphAttentionEmbedding(**params, n_heads=n_heads, use_memory=use_memory)
    elif module_type == "graph_NGCF":
        return NGCFGraphEmbedding(**params, n_heads=n_heads, use_memory=use_memory)
    elif module_type == 'graph_LightGCN':
        return LightGCNEmbedding(**params, n_heads=n_heads, use_memory=use_memory)
    elif module_type == 'graph_UltraGCN':
        return UltraGCNEmbedding(**params, n_heads=n_heads, use_memory=use_memory)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(**params, n_heads=n_heads, use_memory=use_memory)
    elif module_type == "identity":
        return IdentityEmbedding(**params)
    elif module_type == "time":
        return TimeEmbedding(**params, n_neighbors=n_neighbors)
    else:
        raise ValueError(f"Embedding Module {module_type} not supported")
