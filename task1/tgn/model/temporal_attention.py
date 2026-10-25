import torch
from torch import nn
from tgn.utils.utils import MergeLayer


class TemporalAttentionLayer(torch.nn.Module):
    def __init__(self, n_node_features, n_neighbors_features,
                 n_edge_features, time_dim, output_dimension, n_head=2,
                 dropout=0.1, use_edge_features=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_head = n_head
        self.feat_dim = n_node_features
        self.time_dim = time_dim
        self.use_edge_features = use_edge_features
        self.query_dim = n_node_features + time_dim
        self.key_dim = (
            n_neighbors_features + time_dim +
            (n_edge_features if use_edge_features else 0)
        )
        self.merger = MergeLayer(self.query_dim, n_node_features,
                                 n_node_features, output_dimension)
        self.multi_head_target = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            kdim=self.key_dim,
            vdim=self.key_dim,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, src_node_features, src_time_features, neighbors_features,
                neighbors_time_features, edge_features,
                neighbors_padding_mask):
        src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)
        query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
        key_features = [neighbors_features, neighbors_time_features]
        if self.use_edge_features:
            key_features.append(edge_features)
        key = torch.cat(key_features, dim=2)

        invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
        neighbors_padding_mask = neighbors_padding_mask.clone()
        neighbors_padding_mask[:, 0] &= ~invalid_neighborhood_mask.squeeze(1)

        attn_output, _ = self.multi_head_target(
            query=query,
            key=key,
            value=key,
            key_padding_mask=neighbors_padding_mask,
            need_weights=False
        )

        attn_output = attn_output.squeeze(1)

        attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)

        attn_output = self.merger(attn_output, src_node_features)
        return attn_output, None


class LightGCN(torch.nn.Module):
    def __init__(self, n_node_features, embedding_size, layers):
        super(LightGCN, self).__init__()
        self.embedding_size = embedding_size
        self.layers = layers
        self.embedding = torch.nn.Linear(n_node_features, embedding_size)

    def forward(self, src_node_features, neighbors_features, 
                neighbors_padding_mask=None):
        src_node_embedding = self.embedding(src_node_features)
        all_embeddings = [src_node_embedding]
        ego_embeddings = all_embeddings[-1]

        for _ in range(self.layers):
            side_embeddings = torch.mm(neighbors_features, ego_embeddings)
            if neighbors_padding_mask is not None:
                side_embeddings = side_embeddings.masked_fill(
                    neighbors_padding_mask, 0
                )
            all_embeddings += [side_embeddings]
            ego_embeddings = side_embeddings

        all_embeddings = torch.stack(all_embeddings, 1)
        all_embeddings = torch.mean(all_embeddings, dim=1, keepdim=False)
        return all_embeddings


class NGCF(torch.nn.Module):
    def __init__(self, n_node_features, layers):
        super(NGCF, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.num_layers = len(layers)

        for From, To in zip(layers[:-1], layers[1:]):
            self.layers.append(torch.nn.Sequential(
                torch.nn.Linear(From, To),
                torch.nn.LeakyReLU()
            ))

        self.initial_layer = torch.nn.Linear(n_node_features, layers[0])

    def forward(self, src_node_features, neighbors_features, 
                neighbors_padding_mask=None):
        ego_embeddings = self.initial_layer(src_node_features)

        for layer in self.layers:
            side_embeddings = torch.mm(neighbors_features, ego_embeddings)
            if neighbors_padding_mask is not None:
                side_embeddings = side_embeddings.masked_fill(
                    neighbors_padding_mask, 0
                )
            summed_embeddings = side_embeddings + ego_embeddings
            ego_embeddings = layer(summed_embeddings)

        return ego_embeddings
