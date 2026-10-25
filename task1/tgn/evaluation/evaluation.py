import math
import numpy as np
import torch
from tqdm import tqdm
from tgn.utils.utils import RandEdgeSampler


def recall_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / min(k, len(test_items))


def ndcg_at_k(recommendations, test_items, k):
    dcg = 0
    idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(test_items)))])
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            dcg += 1 / np.log2(i + 2)
    return dcg / idcg


def mrr_at_k(recommendations, test_items, k):
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            return 1 / (i + 1)
    return 0


def hit_at_k(recommendations, test_items, k):
    for i, item in enumerate(recommendations[:k]):
        if item in test_items:
            return 1
    return 0


def precision_at_k(recommendations, test_items, k):
    hits = len(set(recommendations[:k]) & set(test_items))
    return hits / k


def eval_recommendation(tgn, data, batch_size, n_neighbors,
                        num_neg_eval, is_test_run):
    with torch.no_grad():
        tgn = tgn.eval()
        test_batch_size = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / test_batch_size)

        recalls, ndcgs, mrrs, hits, precisions = [], [], [], [], []

        for batch in tqdm(range(num_test_batch), desc="Progress: Eval Batch"):
            s_idx = batch * test_batch_size
            e_idx = min(num_test_instance, s_idx + test_batch_size)

            if e_idx == num_test_instance:
                continue

            if is_test_run and batch == 2:
                break

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = data.edge_idxs[s_idx:e_idx]

            test_rand_sampler = RandEdgeSampler(
                sources_batch, destinations_batch, seed=2023
            )
            negatives_batch = test_rand_sampler.sample(size=num_neg_eval)

            source_emb, dest_emb, neg_emb = tgn.compute_temporal_embeddings_eval(
                sources_batch,
                destinations_batch,
                negatives_batch.flatten(),
                timestamps_batch,
                edge_idxs_batch,
                n_neighbors
            )

            bs = source_emb.shape[0]
            source_emb = source_emb.view(bs, 1, -1)
            dest_emb = dest_emb.view(bs, 1, -1)
            neg_emb = neg_emb.view(bs, num_neg_eval, -1)

            pos_scores = torch.sum(source_emb * dest_emb, dim=2).cpu().numpy()
            neg_scores = torch.sum(source_emb * neg_emb, dim=2).cpu().numpy()

            for i in range(bs):
                p_score = pos_scores[i]
                n_score = neg_scores[i]

                scores = np.concatenate((p_score, n_score))
                ranking = np.argsort(scores)[::-1]

                pos_ranking = [0]
                topk = [1, 5, 10, 20]

                recalls.append([recall_at_k(ranking, pos_ranking, k) for k in topk])
                ndcgs.append([ndcg_at_k(ranking, pos_ranking, k) for k in topk])
                mrrs.append([mrr_at_k(ranking, pos_ranking, k) for k in topk])
                hits.append([hit_at_k(ranking, pos_ranking, k) for k in topk])
                precisions.append([precision_at_k(ranking, pos_ranking, k) for k in topk])

        return {
            'recalls': recalls,
            'ndcgs': ndcgs,
            'mrrs': mrrs,
            'hits': hits,
            'precisions': precisions
        }
