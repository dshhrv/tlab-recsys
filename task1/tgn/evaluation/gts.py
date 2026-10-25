import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


def score_full_catalog(tgn, source, timestamp, num_users, num_items, n_neighbors, eval_item_batch_size):
    item_offset = num_users + 1
    scores = np.empty(num_items, dtype=np.float32)

    for start in range(0, num_items, eval_item_batch_size):
        end = min(num_items, start + eval_item_batch_size)
        candidate_nodes = np.arange(start, end, dtype=np.int64) + item_offset

        source_embedding, candidate_embedding = tgn.compute_temporal_embeddings_read_only(
            source_nodes=np.array([source], dtype=np.int64),
            candidate_nodes=candidate_nodes,
            edge_times=np.array([timestamp]),
            n_neighbors=n_neighbors
        )

        chunk_scores = torch.sum(source_embedding * candidate_embedding, dim=1)
        scores[start:end] = chunk_scores.detach().cpu().numpy()

    return scores


def mask_seen_keep_target(scores, seen_items, true_item):
    true_score = float(scores[true_item])

    if seen_items:
        seen = np.fromiter(seen_items, dtype=np.int64, count=len(seen_items))
        seen = seen[(seen >= 0) & (seen < len(scores))]
        scores[seen] = -np.inf

    scores[true_item] = true_score
    return scores


def topk_from_scores(scores, top_n):
    return np.argsort(-scores, kind="stable")[:top_n]


def store_seen_items(history_data, num_users, num_items):
    seen_items = defaultdict(set)
    item_offset = num_users + 1
    
    for source, destination in zip(history_data.sources, history_data.destinations):
        item = int(destination) - item_offset
        if 0 <= item < num_items:
            seen_items[int(source)].add(item)

    return seen_items


def process_real_event(tgn, data, event_idx, n_neighbors):
    tgn.process_real_interaction(
        source_nodes=np.array([int(data.sources[event_idx])], dtype=np.int64),
        destination_nodes=np.array([int(data.destinations[event_idx])], dtype=np.int64),
        edge_times=np.array([data.timestamps[event_idx]]),
        edge_idxs=np.array([int(data.edge_idxs[event_idx])], dtype=np.int64),
        n_neighbors=n_neighbors,
    )


def predict_gts_stream(tgn, stream_data, base_history, target_event_by_user, num_users, num_items, n_neighbors, eval_item_batch_size, top_n, max_targets=None, desc="TGN GTS"):
    item_offset = num_users + 1
    seen_items = store_seen_items(base_history, num_users, num_items)
    recommendations = np.full((num_users, top_n), -1, dtype=np.int64)
    scored_targets = 0

    tgn.eval()
    for event_idx in tqdm(range(len(stream_data.sources)), desc=desc, leave=False):
        source = int(stream_data.sources[event_idx])
        destination = int(stream_data.destinations[event_idx])
        timestamp = stream_data.timestamps[event_idx]
        true_item = destination - item_offset

        if target_event_by_user.get(source) == event_idx:
            scores = score_full_catalog(tgn, source, timestamp, num_users, num_items, n_neighbors, eval_item_batch_size)
            scores = mask_seen_keep_target(scores, seen_items[source], true_item)
            recommendations[source - 1] = topk_from_scores(scores, top_n)
            scored_targets += 1

        process_real_event(tgn, stream_data, event_idx, n_neighbors)
        if 0 <= true_item < num_items:
            seen_items[source].add(true_item)
        if max_targets is not None and scored_targets >= max_targets:
            break

    return recommendations


def predict_gts_stream_many(tgn, stream_data, base_history, target_maps, num_users, num_items, n_neighbors, eval_item_batch_size, top_n, max_targets=None, desc="TGN test GTS all holdouts"):
    item_offset = num_users + 1
    seen_items = store_seen_items(base_history, num_users, num_items)
    recommendations = [np.full((num_users, top_n), -1, dtype=np.int64) for _ in target_maps]
    scored_targets = [0 for _ in target_maps]
    total_targets = sum(len(target_event_by_user) for target_event_by_user in target_maps)

    event_to_holdouts = defaultdict(list)
    for holdout_id, target_event_by_user in enumerate(target_maps):
        for source, event_idx in target_event_by_user.items():
            event_to_holdouts[int(event_idx)].append((holdout_id, int(source)))

    tgn.eval()
    for event_idx in tqdm(range(len(stream_data.sources)), desc=desc, leave=False):
        source = int(stream_data.sources[event_idx])
        destination = int(stream_data.destinations[event_idx])
        timestamp = stream_data.timestamps[event_idx]
        true_item = destination - item_offset

        target_holdouts = [
            holdout_id
            for holdout_id, target_source in event_to_holdouts.get(event_idx, [])
            if target_source == source and (max_targets is None or scored_targets[holdout_id] < max_targets)
        ]

        if target_holdouts:
            scores = score_full_catalog(tgn, source, timestamp, num_users, num_items, n_neighbors, eval_item_batch_size)
            scores = mask_seen_keep_target(scores, seen_items[source], true_item)
            top_items = topk_from_scores(scores, top_n)

            for holdout_id in target_holdouts:
                recommendations[holdout_id][source - 1] = top_items
                scored_targets[holdout_id] += 1

        process_real_event(tgn, stream_data, event_idx, n_neighbors)
        if 0 <= true_item < num_items:
            seen_items[source].add(true_item)

        if max_targets is None and sum(scored_targets) >= total_targets:
            break
        if max_targets is not None and all(count >= max_targets for count in scored_targets):
            break

    return recommendations
