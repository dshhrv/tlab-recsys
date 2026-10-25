import logging
from collections import defaultdict

import numpy as np
import torch

from tgn.model.time_encoding import TimeEncode
from tgn.modules.embedding_module import get_embedding_module
from tgn.modules.memory import Memory
from tgn.modules.memory_updater import get_memory_updater
from tgn.modules.message_aggregator import get_message_aggregator
from tgn.modules.message_function import get_message_function
from tgn.modules.memory import TrainableMemory
from tgn.utils.utils import MergeLayer
import torch.nn as nn

class TGN(torch.nn.Module):
    def __init__(
        self, neighbor_finder, node_features, edge_features, device, n_layers=2,
        n_heads=2, dropout=0.1, use_memory=False,
        memory_update_at_start=True, message_dimension=100,
        memory_dimension=100, embedding_module_type="graph_attention",
        message_function="mlp",
        mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
        # std_time_shift_dst=1, n_neighbors=None, aggregator_type="exponential",
        std_time_shift_dst=1, n_neighbors=None, aggregator_type="attention",
        aggregation_half_life=600.0,
        memory_updater_type="gru",
        use_destination_embedding_in_message=False,
        use_source_embedding_in_message=False,
        dyrep=False, use_edge_features=True, trainable_node_features=True
    ):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        node_raw_features = torch.as_tensor(
            node_features, dtype=torch.float32, device=device
        )
        if trainable_node_features:
            self.node_raw_features = nn.Parameter(node_raw_features)
        else:
            self.register_buffer(
                "node_raw_features", node_raw_features, persistent=False
            )
        self.edge_raw_features = torch.from_numpy(
            edge_features.astype(np.float32)
        ).to(device)
        self.n_node_features = self.node_raw_features.shape[1]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.use_edge_features = use_edge_features
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.aggregator_type = aggregator_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_node_features)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dim = (
                2 * self.memory_dimension + self.time_encoder.dimension +
                (self.n_edge_features if self.use_edge_features else 0)
            )
            message_dimension = (
                message_dimension if message_function != "identity"
                else raw_message_dim
            )
            self.memory = Memory(
                n_nodes=self.n_nodes,
                memory_dimension=self.memory_dimension,
                input_dimension=message_dimension,
                message_dimension=message_dimension,
                device=device
            )
            # self.memory = TrainableMemory(
            #     n_nodes=self.n_nodes,
            #     memory_dimension=self.memory_dimension,
            #     device=self.device,
            # ).to(self.device)

            self.message_aggregator = get_message_aggregator(
                aggregator_type=aggregator_type,
                device=device,
                memory_dim=self.memory_dimension,
                message_dim=raw_message_dim,
                n_heads=n_heads,
                half_life=aggregation_half_life
            )
            self.message_function = get_message_function(
                module_type=message_function,
                raw_message_dimension=raw_message_dim,
                message_dimension=message_dimension
            )
            self.memory_updater = get_memory_updater(
                module_type=memory_updater_type,
                memory=self.memory,
                message_dimension=message_dimension,
                memory_dimension=self.memory_dimension,
                device=device
            )

        self.embedding_module = get_embedding_module(
            module_type=embedding_module_type,
            node_features=self.node_raw_features,
            edge_features=self.edge_raw_features,
            memory=self.memory,
            neighbor_finder=self.neighbor_finder,
            time_encoder=self.time_encoder,
            n_layers=self.n_layers,
            n_node_features=self.n_node_features,
            n_edge_features=self.n_edge_features,
            n_time_features=self.n_node_features,
            embedding_dimension=self.embedding_dimension,
            device=self.device,
            n_heads=n_heads,
            dropout=dropout,
            use_memory=use_memory,
            n_neighbors=self.n_neighbors,
            use_edge_features=self.use_edge_features
        )

        self.affinity_score = MergeLayer(
            self.n_node_features, self.n_node_features,
            self.n_node_features, 1
        )
        
    def compute_temporal_embeddings_read_only(self, source_nodes, candidate_nodes, edge_times, n_neighbors=10, memory_snapshot=None):
        nodes = np.concatenate([source_nodes, candidate_nodes])
        timestamps = np.concatenate([edge_times, np.repeat(edge_times, len(candidate_nodes))])
        memory = None
        time_diffs = None
        if self.use_memory:
            if memory_snapshot is not None:
                memory, last_update = memory_snapshot
            elif self.memory_update_at_start:
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)), self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update
            # memory = self.memory.get_memory(list(range(self.n_nodes)))
            # last_update = self.memory.last_update.clone()

            source_time_diffs = torch.tensor(
                edge_times, dtype=torch.long, device=self.device
            ) - last_update[source_nodes].long()
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src

            candidate_times = np.repeat(edge_times, len(candidate_nodes))
            candidate_time_diffs = torch.tensor(
                candidate_times, dtype=torch.long, device=self.device
            ) - last_update[candidate_nodes].long()
            candidate_time_diffs = (candidate_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst

            time_diffs = torch.cat([source_time_diffs, candidate_time_diffs], dim=0)

        node_embedding = self.embedding_module.compute_embedding(memory=memory, source_nodes=nodes, timestamps=timestamps, n_layers=self.n_layers, n_neighbors=n_neighbors, time_diffs=time_diffs)

        source_embedding = node_embedding[:len(source_nodes)]
        candidate_embedding = node_embedding[len(source_nodes):]

        if self.use_memory and self.dyrep:
            source_embedding = memory[source_nodes]
            candidate_embedding = memory[candidate_nodes]

        return source_embedding, candidate_embedding

    
    def process_real_interaction(
        self, source_nodes, destination_nodes, edge_times, edge_idxs,
        n_neighbors
    ):
        empty_candidates = np.empty(0, dtype=np.int64)
        return self.compute_temporal_embeddings(
            source_nodes,
            destination_nodes,
            empty_candidates,
            edge_times,
            edge_idxs,
            n_neighbors,
        )

    def compute_temporal_embeddings(
        self,
        source_nodes,
        destination_nodes,
        candidate_nodes,
        edge_times,
        edge_idxs,
        n_neighbors=10,
    ):
        candidates_per_source = len(candidate_nodes) // len(source_nodes)
        candidate_timestamps = np.repeat(edge_times, candidates_per_source)
        source_node_embedding, destination_node_embedding, memory, last_update = (
            self.compute_event_embeddings(
                source_nodes,
                destination_nodes,
                edge_times,
                edge_idxs,
                n_neighbors,
            )
        )
        candidate_node_embedding = self.compute_candidate_embeddings(
            candidate_nodes,
            candidate_timestamps,
            memory,
            last_update,
            n_neighbors,
        )

        return (
            source_node_embedding,
            destination_node_embedding,
            candidate_node_embedding
        )

    def compute_event_embeddings(
        self, source_nodes, destination_nodes, edge_times, edge_idxs,
        n_neighbors=10
    ):
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])
        positives = np.concatenate([source_nodes, destination_nodes])
        memory = None
        last_update = None
        time_diffs = None

        if self.use_memory:
            if self.memory_update_at_start:
                memory, last_update = self.get_updated_memory(
                    list(range(self.n_nodes)), self.memory.messages
                )
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update
            # memory = self.memory.get_memory(list(range(self.n_nodes)))
            # last_update = self.memory.last_update.clone()

            source_time_diffs = (
                torch.tensor(edge_times, dtype=torch.long, device=self.device) -
                last_update[source_nodes].long()
            )
            source_time_diffs = (
                (source_time_diffs - self.mean_time_shift_src) /
                self.std_time_shift_src
            )
            destination_time_diffs = (
                torch.tensor(edge_times, dtype=torch.long, device=self.device) -
                last_update[destination_nodes].long()
            )
            destination_time_diffs = (
                (destination_time_diffs - self.mean_time_shift_dst) /
                self.std_time_shift_dst
            )
            time_diffs = torch.cat(
                [source_time_diffs, destination_time_diffs], dim=0
            )

        node_embedding = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs,
        )
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples:]

        if self.use_memory:
            if self.memory_update_at_start:
                self.update_memory(positives, self.memory.messages)
                assert torch.allclose(
                    memory[positives],
                    self.memory.get_memory(positives),
                    atol=1e-5
                )
                self.memory.clear_messages(positives)
            
            unique_sources, source_id_to_messages = self.get_raw_messages(
                source_nodes, source_node_embedding,
                destination_nodes, destination_node_embedding,
                edge_times, edge_idxs
            )
            unique_destinations, dest_id_to_messages = self.get_raw_messages(
                destination_nodes, destination_node_embedding,
                source_nodes, source_node_embedding,
                edge_times, edge_idxs
            )
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, dest_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, dest_id_to_messages)

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]

            self._update_last_update(source_nodes, destination_nodes, edge_times)

        return source_node_embedding, destination_node_embedding, memory, last_update

    def _update_last_update(self, source_nodes, destination_nodes, edge_times):
        event_nodes = np.concatenate([source_nodes, destination_nodes])
        event_times = np.concatenate([edge_times, edge_times])
        with torch.no_grad():
            for node in np.unique(event_nodes):
                self.memory.last_update[node] = int(
                    event_times[event_nodes == node].max()
                )

    def compute_candidate_embeddings(
        self, candidate_nodes, candidate_timestamps, memory, last_update,
        n_neighbors=10
    ):
        if len(candidate_nodes) == 0:
            return torch.empty(
                (0, self.embedding_dimension), device=self.device
            )

        time_diffs = None
        if self.use_memory:
            candidate_time_diffs = (
                torch.tensor(
                    candidate_timestamps, dtype=torch.long, device=self.device
                ) -
                last_update[candidate_nodes].long()
            )
            time_diffs = (
                (candidate_time_diffs - self.mean_time_shift_dst) /
                self.std_time_shift_dst
            )

        candidate_node_embedding = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=candidate_nodes,
            timestamps=candidate_timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs,
        )

        if self.use_memory and self.dyrep:
            candidate_node_embedding = memory[candidate_nodes]

        return candidate_node_embedding

    def compute_temporal_embeddings_eval(
        self, source_nodes, destination_nodes, p_neg_nodes,
        edge_times, edge_idxs, n_neighbors=10
    ):
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, p_neg_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        size = int(len(p_neg_nodes) / len(source_nodes))
        edge_times_duplicated = [x for x in edge_times for _ in range(size)]
        timestamps = np.concatenate([edge_times, edge_times, edge_times_duplicated])

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                memory, last_update = self.get_updated_memory(
                    list(range(self.n_nodes)), self.memory.messages
                )
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update
            # memory = self.memory.get_memory(list(range(self.n_nodes)))
            # last_update = self.memory.last_update

            source_time_diffs = (
                torch.tensor(edge_times, dtype=torch.long, device=self.device) -
                last_update[source_nodes].long()
            )
            source_time_diffs = (
                (source_time_diffs - self.mean_time_shift_src) /
                self.std_time_shift_src
            )

            destination_time_diffs = (
                torch.tensor(edge_times, dtype=torch.long, device=self.device) -
                last_update[destination_nodes].long()
            )
            destination_time_diffs = (
                (destination_time_diffs - self.mean_time_shift_dst) /
                self.std_time_shift_dst
            )

            try:
                p_neg_time_diffs = (
                    torch.tensor(
                        edge_times_duplicated,
                        dtype=torch.long,
                        device=self.device
                    ) -
                    last_update[p_neg_nodes].long()
                )
            except RuntimeError:
                raise

            p_neg_time_diffs = (
                (p_neg_time_diffs - self.mean_time_shift_dst) /
                self.std_time_shift_dst
            )
            time_diffs = torch.cat(
                [source_time_diffs, destination_time_diffs, p_neg_time_diffs],
                dim=0
            )

        node_embedding = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs
        )

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        p_neg_node_embedding = node_embedding[2 * n_samples:]

        if self.use_memory:
            if self.memory_update_at_start:
                self.update_memory(positives, self.memory.messages)
                assert torch.allclose(
                    memory[positives],
                    self.memory.get_memory(positives),
                    atol=1e-5
                )
                self.memory.clear_messages(positives)
            
            unique_sources, source_id_to_messages = self.get_raw_messages(
                source_nodes, source_node_embedding,
                destination_nodes, destination_node_embedding,
                edge_times, edge_idxs
            )
            unique_destinations, dest_id_to_messages = self.get_raw_messages(
                destination_nodes, destination_node_embedding,
                source_nodes, source_node_embedding,
                edge_times, edge_idxs
            )
            
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, dest_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, dest_id_to_messages)

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                p_neg_node_embedding = memory[p_neg_nodes]
            self._update_last_update(source_nodes, destination_nodes, edge_times)

        return (
            source_node_embedding,
            destination_node_embedding,
            p_neg_node_embedding
        )

    def compute_temporal_embeddings_eval_jodie(
        self, source_nodes, destination_nodes, p_neg_nodes,
        edge_times, edge_idxs, n_neighbors=20
    ):
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes, p_neg_nodes])
        positives = np.concatenate([source_nodes, destination_nodes])
        edge_times_duplicated = [x for x in edge_times for _ in range(10)]
        timestamps = np.concatenate([edge_times, edge_times, edge_times_duplicated])

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start:
                memory, last_update = self.get_updated_memory(
                    list(range(self.n_nodes)), self.memory.messages
                )
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update
            # memory = self.memory.get_memory(list(range(self.n_nodes)))
            # last_update = self.memory.last_update

            source_time_diffs = (
                torch.tensor(edge_times, dtype=torch.long, device=self.device) -
                last_update[source_nodes].long()
            )
            source_time_diffs = (
                (source_time_diffs - self.mean_time_shift_src) /
                self.std_time_shift_src
            )
            destination_time_diffs = (
                torch.tensor(edge_times, dtype=torch.long, device=self.device) -
                last_update[destination_nodes].long()
            )
            destination_time_diffs = (
                (destination_time_diffs - self.mean_time_shift_dst) /
                self.std_time_shift_dst
            )
            p_neg_time_diffs = (
                torch.tensor(
                    edge_times_duplicated,
                    dtype=torch.long,
                    device=self.device
                ) -
                last_update[p_neg_nodes].long()
            )
            p_neg_time_diffs = (
                (p_neg_time_diffs - self.mean_time_shift_dst) /
                self.std_time_shift_dst
            )
            time_diffs = torch.cat(
                [source_time_diffs, destination_time_diffs, p_neg_time_diffs],
                dim=0
            )

        node_embedding = self.embedding_module.compute_embedding(
            memory=memory,
            source_nodes=nodes,
            timestamps=timestamps,
            n_layers=self.n_layers,
            n_neighbors=n_neighbors,
            time_diffs=time_diffs
        )

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        p_neg_node_embedding = node_embedding[2 * n_samples:]

        if self.use_memory:
            if self.memory_update_at_start:
                self.update_memory(positives, self.memory.messages)
                assert torch.allclose(
                    memory[positives],
                    self.memory.get_memory(positives),
                    atol=1e-5
                )
                self.memory.clear_messages(positives)
            
            unique_sources, source_id_to_messages = self.get_raw_messages(
                source_nodes, source_node_embedding,
                destination_nodes, destination_node_embedding,
                edge_times, edge_idxs
            )
            unique_destinations, dest_id_to_messages = self.get_raw_messages(
                destination_nodes, destination_node_embedding,
                source_nodes, source_node_embedding,
                edge_times, edge_idxs
            )
            
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, dest_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, dest_id_to_messages)

            if self.dyrep:
                source_node_embedding = memory[source_nodes]
                destination_node_embedding = memory[destination_nodes]
                p_neg_node_embedding = memory[p_neg_nodes]
            self._update_last_update(source_nodes, destination_nodes, edge_times)

        return (
            source_node_embedding,
            destination_node_embedding,
            p_neg_node_embedding
        )

    def compute_edge_probabilities(
        self, source_nodes, destination_nodes, negative_nodes,
        edge_times, edge_idxs, n_neighbors=20
    ):
        n_samples = len(source_nodes)
        (
            source_node_embedding,
            destination_node_embedding,
            negative_node_embedding
        ) = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes,
            edge_times, edge_idxs, n_neighbors
        )

        score = self.affinity_score(
            torch.cat([source_node_embedding, source_node_embedding], dim=0),
            torch.cat([destination_node_embedding, negative_node_embedding])
        ).squeeze(dim=0)
        pos_score = score[:n_samples]
        neg_score = score[n_samples:]

        return pos_score.sigmoid(), neg_score.sigmoid()

    def update_memory(self, nodes, messages):
        aggregation_memory = (
            self.memory.get_all_memory()
            if self.aggregator_type == "attention"
            else None
        )
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages,
                aggregation_memory
            )

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        self.memory_updater.update_memory(
            unique_nodes, unique_messages, timestamps=unique_timestamps
        )

    def get_updated_memory(self, nodes, messages):
        aggregation_memory = (
            self.memory.get_all_memory()
            if self.aggregator_type == "attention"
            else None
        )
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes, messages, aggregation_memory
            )

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = \
            self.memory_updater.get_updated_memory(
                unique_nodes, unique_messages, timestamps=unique_timestamps
            )

        return updated_memory, updated_last_update

    def get_raw_messages(
        self, source_nodes, source_node_embedding, destination_nodes,
        destination_node_embedding, edge_times, edge_idxs
    ):
        if self.aggregator_type == "exponential":
            message_timestamps = torch.as_tensor(edge_times, dtype=torch.float64)
        else:
            message_timestamps = torch.as_tensor(
                edge_times, dtype=torch.float32, device=self.device
            )
        edge_times = torch.as_tensor(
            edge_times, dtype=torch.float32, device=self.device
        )
        source_memory = (
            self.memory.get_memory(source_nodes)
            if not self.use_source_embedding_in_message
            else source_node_embedding
        )
        destination_memory = (
            self.memory.get_memory(destination_nodes)
            if not self.use_destination_embedding_in_message
            else destination_node_embedding
        )

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(
            source_time_delta.unsqueeze(dim=1)
        ).view(len(source_nodes), -1)

        message_features = [source_memory, destination_memory]
        if self.use_edge_features:
            edge_idxs = torch.tensor(
                edge_idxs, dtype=torch.long, device=self.device
            )
            message_features.append(self.edge_raw_features[edge_idxs])
        message_features.append(source_time_delta_encoding)
        source_message = torch.cat(message_features, dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], message_timestamps[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder
