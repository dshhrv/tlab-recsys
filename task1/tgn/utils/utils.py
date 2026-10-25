import numpy as np
import torch


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MLP(torch.nn.Module):
    def __init__(self, dim, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, 1)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor:
    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        self.max_round = max_round
        self.num_round = 0
        self.epoch_count = 0
        self.best_epoch = 0
        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1

        self.epoch_count += 1
        return self.num_round >= self.max_round


class RandEdgeSampler:
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = seed
        self.src_list = src_list
        self.dst_real_list = dst_list
        self.dst_list = np.unique(dst_list)
        self.src_to_dst = {src: [] for src in np.unique(src_list)}

        if seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        else:
            self.random_state = np.random

    def sample(self, size):
        dst_index = []
        for src, dst in zip(self.src_list, self.dst_real_list):
            self.src_to_dst[src].append(dst)
            current_dst = self.src_to_dst[src]
            diff = np.setdiff1d(self.dst_list, current_dst)
            sample_dst = self.random_state.choice(diff, size=size, replace=False)
            dst_index.append(sample_dst)

        return np.array(dst_index)


def get_neighbor_finder(data, uniform, max_node_idx=None):
    max_idx = max(data.sources.max(), data.destinations.max())
    max_node_idx = max_idx if max_node_idx is None else max_node_idx
    adj_list = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp in zip(
        data.sources, data.destinations, data.edge_idxs, data.timestamps
    ):
        adj_list[source].append((destination, edge_idx, timestamp))
        adj_list[destination].append((source, edge_idx, timestamp))

    return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
    def __init__(self, adj_list, uniform=False, seed=None):
        self.node_to_neighbors = []
        self.node_to_edge_idxs = []
        self.node_to_edge_timestamps = []

        for neighbors in adj_list:
            sorted_neighbors = sorted(neighbors, key=lambda x: x[2])
            self.node_to_neighbors.append(
                np.array([x[0] for x in sorted_neighbors])
            )
            self.node_to_edge_idxs.append(
                np.array([x[1] for x in sorted_neighbors])
            )
            self.node_to_edge_timestamps.append(
                np.array([x[2] for x in sorted_neighbors])
            )

        self.uniform = uniform
        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def find_before(self, src_idx, cut_time):
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
        return (self.node_to_neighbors[src_idx][:i],
                self.node_to_edge_idxs[src_idx][:i],
                self.node_to_edge_timestamps[src_idx][:i])

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        n_samples = len(source_nodes)
        neighbors = np.zeros((n_samples, tmp_n_neighbors), dtype=np.int32)
        edge_idxs = np.zeros((n_samples, tmp_n_neighbors), dtype=np.int32)
        edge_times = np.zeros((n_samples, tmp_n_neighbors), dtype=np.float32)

        for i, (node, ts) in enumerate(zip(source_nodes, timestamps)):
            s_neighbors, s_edge_idxs, s_edge_times = self.find_before(node, ts)

            if len(s_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:
                    sampled_idx = np.random.randint(0, len(s_neighbors), n_neighbors)
                    neighbors[i, :] = s_neighbors[sampled_idx]
                    edge_idxs[i, :] = s_edge_idxs[sampled_idx]
                    edge_times[i, :] = s_edge_times[sampled_idx]

                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                else:
                    s_edge_times = s_edge_times[-n_neighbors:]
                    s_neighbors = s_neighbors[-n_neighbors:]
                    s_edge_idxs = s_edge_idxs[-n_neighbors:]

                    n_found = len(s_neighbors)
                    neighbors[i, n_neighbors - n_found:] = s_neighbors
                    edge_idxs[i, n_neighbors - n_found:] = s_edge_idxs
                    edge_times[i, n_neighbors - n_found:] = s_edge_times

        return neighbors, edge_idxs, edge_times
