import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from dataset import PAD
from gpt2 import score_prompts
from reward_model import predict_components, reward_value

KS = (5, 10, 20)


def reward_components(models, prompts, candidates, max_length, device):
    if isinstance(models, (list, tuple)):
        return np.mean([predict_components(model, prompts, candidates, max_length, device) for model in models], axis=0)
    return predict_components(models, prompts, candidates, max_length, device)


def best_tokens(scores, available, count):
    tokens = np.flatnonzero(available)
    if len(tokens) == 0:
        return np.empty(0, dtype=np.int64)
    values = scores[tokens]
    count = min(count, len(tokens))
    selected = np.argpartition(-values, count - 1)[:count]
    selected = selected[np.lexsort((tokens[selected], -values[selected]))]
    return tokens[selected]


def kl_divergence(policy_scores, base_scores, available):
    if not available.any():
        return 0.0
    policy_values = policy_scores[available].astype(np.float64)
    base_values = base_scores[available].astype(np.float64)
    log_p = policy_values - logsumexp(policy_values)
    log_q = base_values - logsumexp(base_values)
    return float(np.sum(np.exp(log_p) * (log_p - log_q)))


def candidate_mask(store):
    mask = np.ones(store.catalog.vocab_size, dtype=bool)
    mask[:2] = False
    return mask


def score_holdout(store, policy, base, reward_model, audit_model, split, holdout, device, rerank_n=0, batch_size=16, reward_function='ratio', like_weight=0.3, rerank_alpha=None):
    if rerank_n and rerank_n < max(KS):
        raise ValueError('rerank_n must be at least 20')
    frame = store.holdout(split, holdout)
    rows = []
    for start in range(0, len(frame), batch_size):
        group = frame.iloc[start:start + batch_size]
        prompts = [store.prompt(row.user_id, row.timestamp) for row in group.itertuples(index=False)]
        policy_logits = score_prompts(policy, prompts, store.max_length, device, batch_size)
        base_logits = policy_logits if policy is base else score_prompts(base, prompts, store.max_length, device, batch_size)
        for offset, query in enumerate(group.itertuples(index=False)):
            prompt = prompts[offset]
            seen = store.seen_tokens(query.user_id, query.timestamp)
            available = candidate_mask(store)
            target = store.catalog.token(query.item_id)
            logits = policy_logits[offset]
            reference = base_logits[offset]
            top = best_tokens(logits, available, max(rerank_n, max(KS)))
            if target >= 2 and available[target]:
                positions = np.flatnonzero(top == target)
                rank = int(positions[0]) + 1 if len(positions) else float('inf')
            else:
                rank = float('inf')
            rows.append({'user_id': int(query.user_id), 'item_id': int(query.item_id), 'timestamp': int(query.timestamp), 'prompt': prompt, 'seen': seen, 'target': target, 'target_available': bool(target >= 2 and available[target]), 'top': top, 'top_scores': logits[top], 'rank': rank, 'kl': 0.0 if policy is base else kl_divergence(logits, reference, available)})
    if rerank_n and rerank_alpha != 0:
        if reward_model is None:
            raise ValueError('Reranking requires a reward model')
        prompts = [row['prompt'] for row in rows for _ in row['top']]
        candidates = np.concatenate([row['top'] for row in rows]).astype(np.int64)
        components = reward_components(reward_model, prompts, candidates, store.max_length, device)
        values = reward_value(components, reward_function, like_weight)
        cursor = 0
        for row in rows:
            size = len(row['top'])
            local = values[cursor:cursor + size]
            if rerank_alpha is None:
                combined = local
            else:
                base_scores = row['top_scores'].astype(np.float64)
                normalized_base = (base_scores - base_scores.mean()) / max(base_scores.std(), 1e-8)
                normalized_reward = (local - local.mean()) / max(local.std(), 1e-8)
                combined = normalized_base + rerank_alpha * normalized_reward
            order = np.argsort(-combined, kind='stable')
            positions = np.flatnonzero(row['top'][order] == row['target'])
            row['rank'] = int(positions[0]) + 1 if row['target_available'] and len(positions) else float('inf')
            row['top'] = row['top'][order]
            cursor += size
    for row in rows:
        row['top'] = row['top'][:max(KS)]
    if audit_model is not None:
        audit_prompts = [row['prompt'] for row in rows for _ in row['top']]
        audit_candidates = np.concatenate([row['top'] for row in rows]).astype(np.int64)
        audit_components = reward_components(audit_model, audit_prompts, audit_candidates, store.max_length, device)
        audit_rewards = reward_value(audit_components, reward_function, like_weight)
        if reward_model is audit_model or reward_model is None:
            optimization_rewards = audit_rewards
        else:
            optimization_components = reward_components(reward_model, audit_prompts, audit_candidates, store.max_length, device)
            optimization_rewards = reward_value(optimization_components, reward_function, like_weight)
    recommended = {k: set() for k in KS}
    metric = {'queries': len(rows), 'target_outside_catalog': sum(row['target'] == PAD for row in rows), 'target_previously_seen': sum(row['target'] >= 2 and row['target'] in row['seen'] for row in rows), 'eligible_queries': sum(row['target_available'] for row in rows), 'catalog_items': len(store.catalog.items)}
    ranks = np.asarray([row['rank'] for row in rows], dtype=np.float64)
    eligible_ranks = ranks[np.asarray([row['target_available'] for row in rows])]
    popular = store.catalog.popular_items
    for k in KS:
        for row in rows:
            recommended[k].update(int(store.catalog.token_to_item[token]) for token in row['top'][:k])
        metric[f'recall@{k}'] = float(np.mean(ranks <= k))
        metric[f'ndcg@{k}'] = float(np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)))
        metric[f'eligible_recall@{k}'] = float(np.mean(eligible_ranks <= k)) if len(eligible_ranks) else 0.0
        metric[f'eligible_ndcg@{k}'] = float(np.mean(np.where(eligible_ranks <= k, 1 / np.log2(eligible_ranks + 1), 0))) if len(eligible_ranks) else 0.0
        metric[f'coverage@{k}'] = len(recommended[k]) / len(store.catalog.items)
        metric[f'popular_fraction@{k}'] = float(np.mean([int(store.catalog.token_to_item[token]) in popular for row in rows for token in row['top'][:k]]))
        metric[f'repeat_fraction@{k}'] = float(np.mean([token in row['seen'] for row in rows for token in row['top'][:k]]))
        if audit_model is not None:
            offsets = np.cumsum([0] + [len(row['top']) for row in rows])
            metric[f'predicted_reward@{k}'] = float(np.mean([np.mean(optimization_rewards[offsets[i]:offsets[i] + min(k, len(row['top']))]) for i, row in enumerate(rows)]))
            metric[f'independent_reward@{k}'] = float(np.mean([np.mean(audit_rewards[offsets[i]:offsets[i] + min(k, len(row['top']))]) for i, row in enumerate(rows)]))
    metric['kl_vs_base'] = float(np.mean([row['kl'] for row in rows]))
    return metric


def summarize(holdouts):
    summary = {}
    for key in holdouts[0]:
        values = [row[key] for row in holdouts]
        summary[f'{key}_mean'] = float(np.mean(values))
        summary[f'{key}_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return summary


def holdout_overlap(store):
    sets = []
    for index in range(store.info['holdout_counts']['test']):
        frame = store.holdout('test', index)
        sets.append(set(zip(frame.user_id, frame.item_id, frame.timestamp)))
    overlaps = [len(left & right) / len(left | right) for i, left in enumerate(sets) for right in sets[i + 1:]]
    return {'shared_users': len(set.intersection(*[set(store.holdout('test', index).user_id) for index in range(len(sets))])), 'mean_query_jaccard': float(np.mean(overlaps)) if overlaps else None}


def evaluate_variant(store, policy, base, reward_model, audit_model, output, device, rerank_n=0, batch_size=16, reward_function='ratio', like_weight=0.3, task=None, include_test=True, rerank_alpha=None):
    results = {'catalog_items': len(store.catalog.items), 'original_train_catalog_items': store.info['train_catalog_items'], 'reward_function': reward_function, 'rerank_n': rerank_n, 'rerank_alpha': rerank_alpha, 'repeat_allowed': True, 'test_holdout_overlap': holdout_overlap(store) if include_test else None}
    for split in (('validation', 'test') if include_test else ('validation',)):
        holdouts = []
        for index in range(store.info['holdout_counts'][split]):
            value = score_holdout(store, policy, base, reward_model, audit_model, split, index, device, rerank_n, batch_size, reward_function, like_weight, rerank_alpha)
            holdouts.append(value)
            if task:
                for name, number in value.items():
                    task.get_logger().report_scalar(f'{split}/{name}', 'holdout', number, index)
        results[split] = {'protocol': store.info['holdout_protocols'][split], 'holdouts': holdouts, 'summary': summarize(holdouts), 'holdouts_independent': False if split == 'test' else None}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    return results


def rollout_five(store, policy, audit_model, output, device, split='validation', holdout=0, max_queries=100, reward_function='ratio', like_weight=0.3):
    queries = store.holdout(split, holdout).iloc[:max_queries]
    sequences = []
    for query in queries.itertuples(index=False):
        generated = []
        seen = store.seen_tokens(query.user_id, query.timestamp)
        steps = []
        for step in range(5):
            prompt = store.prompt(query.user_id, query.timestamp, generated)
            logits = score_prompts(policy, [prompt], store.max_length, device, 1)[0]
            available = candidate_mask(store)
            token = int(best_tokens(logits, available, 1)[0])
            components = predict_components(audit_model, [prompt], np.array([token]), store.max_length, device)
            item = int(store.catalog.token_to_item[token])
            steps.append({'step': step + 1, 'item_id': item, 'predicted_reward': float(reward_value(components, reward_function, like_weight)[0]), 'popular': item in store.catalog.popular_items, 'repeated_real_history': token in seen, 'repeated_generated': token in generated})
            generated.append(token)
        sequences.append({'user_id': int(query.user_id), 'timestamp': int(query.timestamp), 'steps': steps})
    distinct_items = len({step['item_id'] for row in sequences for step in row['steps']})
    summary = {'queries': len(sequences), 'step1_reward': float(np.mean([row['steps'][0]['predicted_reward'] for row in sequences])), 'mean_five_step_reward': float(np.mean([step['predicted_reward'] for row in sequences for step in row['steps']])), 'reward_by_step': [float(np.mean([row['steps'][step]['predicted_reward'] for row in sequences])) for step in range(5)], 'popular_fraction': float(np.mean([step['popular'] for row in sequences for step in row['steps']])), 'repeat_fraction': float(np.mean([step['repeated_real_history'] for row in sequences for step in row['steps']])), 'generated_repeat_fraction': float(np.mean([step['repeated_generated'] for row in sequences for step in row['steps']])), 'distinct_items': distinct_items, 'diversity_fraction': distinct_items / (5 * len(sequences))}
    result = {'reward_function': reward_function, 'summary': summary, 'sequences': sequences, 'interpretation': 'Generated interactions are simulated context only; rewards are model predictions.'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    return result
