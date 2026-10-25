import random
import numpy as np
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)


def get_data_node_classification(dataset_name, use_validation=True):
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))
    edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
    node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.80, 0.90]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    random.seed(2020)

    train_mask = (timestamps <= val_time if use_validation
                  else timestamps <= test_time)
    test_mask = timestamps > test_time
    val_mask = (np.logical_and(timestamps <= test_time, timestamps > val_time)
                if use_validation else test_mask)

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    train_data = Data(sources[train_mask], destinations[train_mask],
                      timestamps[train_mask], edge_idxs[train_mask],
                      labels[train_mask])

    val_data = Data(sources[val_mask], destinations[val_mask],
                    timestamps[val_mask], edge_idxs[val_mask],
                    labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask],
                     timestamps[test_mask], edge_idxs[test_mask],
                     labels[test_mask])

    return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, dim, different_new_nodes_between_val_and_test=False,
             randomize_features=True):
    graph_df = pd.read_csv('./data/ml_{}.csv'.format(dataset_name))

    if randomize_features:
        edge_features = np.random.rand(int(graph_df.idx.max()) + 1, dim)
        n_nodes = max(int(graph_df.u.max()), int(graph_df.i.max())) + 1
        node_features = np.random.rand(n_nodes, dim)
    else:
        edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
        node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

    val_time, test_time = list(np.quantile(graph_df.ts, [0.80, 0.90]))

    sources = graph_df.u.values
    destinations = graph_df.i.values
    edge_idxs = graph_df.idx.values
    labels = graph_df.label.values
    timestamps = graph_df.ts.values

    full_data = Data(sources, destinations, timestamps, edge_idxs, labels)

    random.seed(2020)

    node_set = set(sources) | set(destinations)
    n_total_unique_nodes = len(node_set)

    test_node_set = set(sources[timestamps > val_time]).union(
        set(destinations[timestamps > val_time]))

    new_test_node_set = set(random.sample(test_node_set,
                                          int(0.1 * n_total_unique_nodes)))

    new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
    new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

    observed_edges_mask = np.logical_and(~new_test_source_mask,
                                         ~new_test_destination_mask)

    train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

    train_data = Data(sources[train_mask], destinations[train_mask],
                      timestamps[train_mask], edge_idxs[train_mask],
                      labels[train_mask])

    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
    test_mask = timestamps > test_time

    val_data = Data(sources[val_mask], destinations[val_mask],
                    timestamps[val_mask], edge_idxs[val_mask],
                    labels[val_mask])

    test_data = Data(sources[test_mask], destinations[test_mask],
                     timestamps[test_mask], edge_idxs[test_mask],
                     labels[test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(
        full_data.n_interactions, full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} nodes".format(
        train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} nodes".format(
        val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} nodes".format(
        test_data.n_interactions, test_data.n_unique_nodes))

    return node_features, edge_features, full_data, train_data, val_data, test_data


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []

    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]

        if source_id not in last_timestamp_sources:
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst:
            last_timestamp_dst[dest_id] = 0

        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])

        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp

    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return (mean_time_shift_src, std_time_shift_src,
            mean_time_shift_dst, std_time_shift_dst)
