from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluate import best_tokens, candidate_mask
from gpt2 import score_prompts
from reward_model import predict_components, reward_value


def compare_reward_hacking(comparisons, split='test'):
    rows = []
    for name, result in comparisons.items():
        summary = result[split]['summary']
        trained = summary.get('predicted_reward@20_mean')
        audit = summary.get('independent_reward@20_mean')
        rows.append({'model': name, 'reward_function': result['reward_function'], 'trained_reward@20': trained, 'independent_reward@20': audit, 'reward_gap': None if trained is None or audit is None else trained - audit, 'recall@20': summary['recall@20_mean'], 'kl_vs_base': summary['kl_vs_base_mean']})
    return pd.DataFrame(rows)


def plot_comparison(comparisons, output, split='test'):
    names = list(comparisons)
    fields = [('recall@20', 'Recall@20'), ('independent_reward@20', 'Independent predicted ratio reward@20'), ('popular_fraction@20', 'Popular fraction@20'), ('kl_vs_base', 'KL versus base GPT-2')]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for axis, (field, title) in zip(axes.flat, fields):
        axis_names = [name for name in names if field != 'independent_reward@20' or comparisons[name]['reward_function'] == 'ratio']
        means = [comparisons[name][split]['summary'].get(f'{field}_mean', np.nan) for name in axis_names]
        standard = [comparisons[name][split]['summary'].get(f'{field}_std') or 0 for name in axis_names]
        axis.bar(axis_names, means, yerr=standard, capsize=3)
        axis.set_title(title)
        axis.tick_params(axis='x', labelrotation=55)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_rollouts(rollouts, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, function in zip(axes.flat[:3], ('ratio', 'duration', 'combined')):
        for name, result in rollouts.items():
            if result['reward_function'] == function:
                axis.plot(range(1, 6), result['summary']['reward_by_step'], marker='o', label=name)
        axis.set_title(function)
        axis.set_xlabel('Generated step')
        axis.set_ylabel('Independent predicted reward')
        axis.legend(fontsize=8)
    names = list(rollouts)
    axes[1, 1].bar(names, [rollouts[name]['summary']['diversity_fraction'] for name in names])
    axes[1, 1].set_ylabel('Distinct items / generated items')
    axes[1, 1].tick_params(axis='x', labelrotation=55)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_rerank_tradeoff(comparisons, output):
    fig, axis = plt.subplots(figsize=(8, 6))
    for name, result in comparisons.items():
        summary = result['validation']['summary']
        axis.scatter(summary['ndcg@20_mean'], summary['predicted_reward@20_mean'], label=name)
        axis.annotate(name, (summary['ndcg@20_mean'], summary['predicted_reward@20_mean']), fontsize=8)
    axis.set_xlabel('Validation NDCG@20')
    axis.set_ylabel('Validation predicted reward@20, seed 42')
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_training_diagnostics(history, output):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for field in ('reward', 'predicted_reward', 'val_predicted_reward@20', 'val_ndcg@20'):
        if field in frame:
            axes[0, 0].plot(frame.step, frame[field], label=field)
    axes[0, 0].legend(fontsize=8)
    for field in ('policy_loss', 'ce_loss', 'pairwise_loss', 'kl'):
        if field in frame:
            axes[0, 1].plot(frame.step, frame[field], label=field)
    axes[0, 1].legend(fontsize=8)
    for field in ('gradient_norm_policy', 'gradient_norm_ce', 'gradient_norm_pairwise', 'gradient_norm_kl'):
        if field in frame:
            axes[1, 0].plot(frame.step, frame[field], label=field)
    axes[1, 0].legend(fontsize=8)
    if 'val_ndcg@20' in frame:
        axes[1, 1].plot(frame.step, frame['val_ndcg@20'], label='NDCG@20')
    if 'val_independent_reward@20' in frame:
        axes[1, 1].plot(frame.step, frame['val_independent_reward@20'], label='Independent reward@20')
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def stable_spearman(left, right):
    if len(left) < 2 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return None
    return float(spearmanr(left, right).statistic)


def reward_bias_diagnostics(store, base, reward_model, audit_model, cache_path, device, query_count=100, candidates_per_source=100, seed=42):
    queries = store.holdout('validation', 0).iloc[:query_count]
    prompts = [store.prompt(row.user_id, row.timestamp) for row in queries.itertuples(index=False)]
    logits = score_prompts(base, prompts, store.max_length, device)
    mask = candidate_mask(store)
    rng = np.random.default_rng(seed)
    candidate_groups, source_groups, repeated_groups = [], [], []
    for index, row in enumerate(queries.itertuples(index=False)):
        top = best_tokens(logits[index], mask, candidates_per_source)
        random_pool = np.setdiff1d(np.arange(2, store.catalog.vocab_size), top)
        random = rng.choice(random_pool, size=candidates_per_source, replace=False)
        candidates = np.concatenate((top, random))
        candidate_groups.append(candidates)
        source_groups.extend(['gpt_top'] * len(top) + ['uniform'] * len(random))
        seen = store.seen_tokens(row.user_id, row.timestamp)
        repeated_groups.extend([token in seen for token in candidates])
    candidates = np.concatenate(candidate_groups)
    expanded_prompts = [prompt for prompt, group in zip(prompts, candidate_groups) for _ in group]
    trained = reward_value(predict_components(reward_model, expanded_prompts, candidates, store.max_length, device), 'ratio')
    audit = reward_value(predict_components(audit_model, expanded_prompts, candidates, store.max_length, device), 'ratio')
    cache = pd.read_parquet(cache_path, columns=['item_id', 'track_length_seconds']).iloc[:store.info['split_sizes']['train']]
    lengths = cache.groupby('item_id').track_length_seconds.median()
    item_ids = store.catalog.token_to_item[candidates]
    length = lengths.reindex(item_ids).to_numpy(np.float64)
    popularity = np.log1p(store.catalog.counts[candidates - 2])
    repeated = np.asarray(repeated_groups)
    source = np.asarray(source_groups)
    result = {'queries': len(queries), 'candidates_per_source': candidates_per_source}
    for label in ('gpt_top', 'uniform'):
        selected = source == label
        valid_length = selected & np.isfinite(length)
        result[label] = {'predicted_reward_mean': float(trained[selected].mean()),
                         'audit_reward_mean': float(audit[selected].mean()),
                         'seed_gap_mean': float((trained[selected] - audit[selected]).mean()),
                         'repeat_fraction': float(repeated[selected].mean()),
                         'reward_repeat_gap': float(trained[selected & repeated].mean() - trained[selected & ~repeated].mean()) if (selected & repeated).any() and (selected & ~repeated).any() else None,
                         'reward_popularity_spearman': stable_spearman(trained[selected], popularity[selected]),
                         'reward_length_spearman': stable_spearman(trained[valid_length], length[valid_length])}
    reward_model.cpu()
    audit_model.cpu()
    return result
