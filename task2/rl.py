import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import BOS, PAD, LMWindows, prompt_batch, set_seed
from evaluate import score_holdout
from gpt2 import last_logits, load_gpt
from reward_model import reward_value


@dataclass
class RLSettings:
    steps: int = 500
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    temperature: float = 1.0
    kl_coefficient: float = 0.02
    ce_coefficient: float = 0.0
    uncertainty_penalty: float = 0.0
    reward_weight: float = 1.0
    relevance_weight: float = 0.0
    reward_function: str = 'ratio'
    like_weight: float = 0.3
    baseline_decay: float = 0.9
    reward_std_floor: float = 0.05
    group_size: int = 1
    pairwise_weight: float = 0.0
    reward_batch_size: int = 256
    eval_every: int = 50
    gradient_log_every: int = 50
    max_ndcg_drop: float = 0.005
    seed: int = 42


def training_prompts(store, limit, seed):
    train = store.splits['train']
    train = train[train.item_id.isin(store.catalog.item_to_token)]
    selected = train.sample(n=min(len(train), limit), random_state=seed)
    return [store.prompt(row.user_id, row.timestamp) for row in selected.itertuples(index=False)]


def masked_logits(logits, temperature):
    values = logits.float() / temperature
    values = values.clone()
    values[:, PAD] = -1e9
    values[:, BOS] = -1e9
    return values


def ensemble_reward(models, ids, attention, actions, settings, device):
    batch, group = actions.shape
    history = ids.detach().cpu().repeat_interleave(group, dim=0)
    masks = attention.detach().cpu().repeat_interleave(group, dim=0)
    candidates = actions.detach().reshape(-1).cpu()
    estimates = []

    with torch.inference_mode():
        for model in models:
            model.eval()
            predictions = []
            for start in range(0, len(candidates), settings.reward_batch_size):
                end = start + settings.reward_batch_size
                components = model(history[start:end], masks[start:end], candidates[start:end])
                predictions.append(reward_value(components.float().numpy(), settings.reward_function, settings.like_weight))
            estimates.append(np.concatenate(predictions).reshape(batch, group))

    values = np.stack(estimates)
    mean = values.mean(axis=0)
    deviation = values.std(axis=0)
    reward = mean - settings.uncertainty_penalty * deviation
    return (
        torch.as_tensor(reward, dtype=torch.float32, device=device),
        float(mean.mean()),
        float(deviation.mean()),
    )


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return float(torch.sqrt(sum(gradient.detach().float().square().sum() for gradient in gradients if gradient is not None)).cpu())


def reinforce_step(policy, reference, models, optimizer, prompts, settings, max_length, device, baseline, ce_batch=None, log_gradients=False):
    ids, attention = prompt_batch(prompts, max_length, device)

    policy.eval()
    logits = masked_logits(last_logits(policy, ids, attention), settings.temperature)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    actions = torch.multinomial(probs.detach(), settings.group_size, replacement=False)

    rewards, predicted_mean, uncertainty = ensemble_reward(
        models, ids, attention, actions, settings, device
    )
    with torch.no_grad():
        reference_logits = masked_logits(last_logits(reference, ids, attention), settings.temperature)
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)
        if settings.relevance_weight:
            reference_mean = reference_logits[:, 2:].mean(dim=1, keepdim=True)
            reference_std = reference_logits[:, 2:].std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
            relevance = 0.5 + 0.5 * torch.tanh((reference_logits.gather(1, actions) - reference_mean) / (2 * reference_std))
        else:
            relevance = torch.zeros_like(rewards)
    combined = settings.reward_weight * rewards + settings.relevance_weight * relevance
    if settings.group_size > 1:
        group_mean = combined.mean(dim=1, keepdim=True)
        group_std = combined.std(dim=1, unbiased=False, keepdim=True)
        advantages = (combined - group_mean) / group_std.clamp_min(settings.reward_std_floor)
    else:
        center = combined.new_tensor(0.5 if baseline is None else baseline)
        advantages = (combined - center) / combined.std(unbiased=False).clamp_min(settings.reward_std_floor)
    remaining_logits = logits
    chosen_terms = []
    for column in range(settings.group_size):
        chosen = actions[:, column:column + 1]
        chosen_terms.append(F.log_softmax(remaining_logits, dim=-1).gather(1, chosen))
        remaining_logits = remaining_logits.scatter(1, chosen, -1e9)
    chosen_log_probs = torch.cat(chosen_terms, dim=1)
    pg_loss = -(advantages.detach() * chosen_log_probs).mean()

    if settings.pairwise_weight:
        best_index = combined.argmax(dim=1, keepdim=True)
        worst_index = combined.argmin(dim=1, keepdim=True)
        marginal_log_probs = log_probs.gather(1, actions)
        margin = marginal_log_probs.gather(1, best_index) - marginal_log_probs.gather(1, worst_index)
        reward_gap = (combined.gather(1, best_index) - combined.gather(1, worst_index)).detach()
        pairwise_loss = ((reward_gap / settings.reward_std_floor).clamp(max=2) * F.softplus(-margin)).mean()
    else:
        pairwise_loss = logits.new_zeros(())
    kl = (probs * (log_probs - reference_log_probs)).sum(dim=1).mean()
    if settings.ce_coefficient:
        ce_ids, ce_attention, ce_labels = (part.to(device) for part in ce_batch)
        ce_loss = policy(input_ids=ce_ids, attention_mask=ce_attention, labels=ce_labels).loss
    else:
        ce_loss = logits.new_zeros(())
    terms = {'policy': pg_loss,
             'pairwise': settings.pairwise_weight * pairwise_loss if settings.pairwise_weight else logits.new_zeros(()),
             'kl': settings.kl_coefficient * kl if settings.kl_coefficient else logits.new_zeros(()),
             'ce': settings.ce_coefficient * ce_loss if settings.ce_coefficient else logits.new_zeros(())}
    metrics = {}
    if log_gradients:
        parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        for name, term in terms.items():
            metrics[f'gradient_norm_{name}'] = gradient_norm(term, parameters) if term.requires_grad else 0.0
    loss = sum(terms.values())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    metrics['gradient_norm_total'] = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), settings.gradient_clip))
    optimizer.step()

    current_reward = float(combined.mean())
    baseline = current_reward if baseline is None else (
        settings.baseline_decay * baseline + (1 - settings.baseline_decay) * current_reward
    )
    metrics.update({
        'loss': float(loss.detach()),
        'policy_loss': float(pg_loss.detach()),
        'rl_loss': float(pg_loss.detach()),
        'pairwise_loss': float(pairwise_loss.detach()),
        'weighted_pairwise_loss': float((settings.pairwise_weight * pairwise_loss).detach()),
        'ce_loss': float(ce_loss.detach()),
        'weighted_ce_loss': float((settings.ce_coefficient * ce_loss).detach()),
        'reward': current_reward,
        'predicted_reward': predicted_mean,
        'relevance_reward': float(relevance.mean()),
        'uncertainty': uncertainty,
        'kl': float(kl.detach()),
        'kl_loss': float((settings.kl_coefficient * kl).detach()),
    })
    return baseline, metrics


def validation_metrics(store, policy, reference, model, audit_model, device, settings):
    results = score_holdout(
        store, policy, reference, model, audit_model, 'validation', 0, device,
        batch_size=settings.batch_size, reward_function=settings.reward_function,
        like_weight=settings.like_weight,
    )
    model.cpu()
    if audit_model is not None:
        audit_model.cpu()
    return results


def train_rl(store, base_checkpoint, reward_models, settings, checkpoint_dir, device, audit_model=None, task=None):
    if settings.steps <= 0 or settings.batch_size <= 0 or settings.group_size <= 0:
        raise ValueError('steps, batch_size and group_size must be positive')
    if settings.group_size > len(store.catalog.items) or (settings.pairwise_weight and settings.group_size < 2):
        raise ValueError('Invalid group_size for the active catalog or pairwise loss')
    if settings.temperature <= 0 or settings.eval_every <= 0 or settings.gradient_log_every <= 0 or settings.reward_std_floor <= 0:
        raise ValueError('temperature, eval_every, gradient_log_every and reward_std_floor must be positive')
    if min(settings.kl_coefficient, settings.ce_coefficient, settings.pairwise_weight,
           settings.reward_weight, settings.relevance_weight, settings.uncertainty_penalty) < 0:
        raise ValueError('Loss and reward coefficients must be nonnegative')
    if not reward_models:
        raise ValueError('At least one reward model is required')
    if audit_model is None:
        raise ValueError('An independent audit reward model is required for validation')
    if settings.uncertainty_penalty and len(reward_models) < 2:
        raise ValueError('Uncertainty penalty requires at least two reward models')

    set_seed(settings.seed)
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(f'Experiment directory already exists: {checkpoint_dir}')
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / 'settings.json').write_text(json.dumps(asdict(settings), indent=2))

    policy = load_gpt(base_checkpoint, device).eval()
    reference = load_gpt(base_checkpoint, device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    for model in [*reward_models, audit_model]:
        if model is None:
            continue
        model.eval().cpu()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    examples = training_prompts(store, max(10000, settings.steps * settings.batch_size), settings.seed)
    if not examples:
        raise ValueError('No RL training contexts found')
    ce_windows = LMWindows(store.splits['train'], store.catalog, store.max_length) if settings.ce_coefficient else None
    if ce_windows is not None and not len(ce_windows):
        raise ValueError('No supervised train windows found')

    initial = validation_metrics(store, policy, reference, reward_models[0], audit_model, device, settings)
    best_reward = initial['predicted_reward@20']
    minimum_ndcg = initial['ndcg@20'] - settings.max_ndcg_drop
    (checkpoint_dir / 'initial_validation.json').write_text(json.dumps(initial, indent=2))
    policy.save_pretrained(checkpoint_dir / 'best', safe_serialization=True)
    best_validation = initial
    best_step = 0
    last_validation = initial

    rng = np.random.default_rng(settings.seed)
    baseline, history = None, []
    for step in range(1, settings.steps + 1):
        positions = rng.integers(len(examples), size=settings.batch_size)
        prompts = [examples[index] for index in positions]
        ce_batch = None
        if ce_windows is not None:
            indices = rng.integers(len(ce_windows), size=settings.batch_size)
            ce_batch = tuple(torch.stack([ce_windows[index][part] for index in indices]) for part in range(3))
        baseline, metrics = reinforce_step(
            policy, reference, reward_models, optimizer, prompts,
            settings, store.max_length, device, baseline, ce_batch,
            log_gradients=step == 1 or step % settings.gradient_log_every == 0
        )
        metrics['step'] = step

        if step % settings.eval_every == 0 or step == settings.steps:
            val = validation_metrics(store, policy, reference, reward_models[0], audit_model, device, settings)
            last_validation = val
            metrics.update({f'val_{key}': value for key, value in val.items() if isinstance(value, (int, float))})
            metrics['val_parameter_delta_l2'] = float(torch.sqrt(sum(
                (current.detach() - original.detach()).float().square().sum()
                for current, original in zip(policy.parameters(), reference.parameters())
            )))
            if (np.isfinite(val['predicted_reward@20'])
                    and val['ndcg@20'] >= minimum_ndcg
                    and val['predicted_reward@20'] > best_reward):
                best_reward = val['predicted_reward@20']
                best_validation = val
                best_step = step
                policy.save_pretrained(checkpoint_dir / 'best', safe_serialization=True)

        history.append(metrics)
        if task is not None:
            for name, number in metrics.items():
                if name != 'step':
                    task.get_logger().report_scalar('rl', name, number, step)

    policy.save_pretrained(checkpoint_dir / 'last', safe_serialization=True)
    (checkpoint_dir / 'history.json').write_text(json.dumps(history, indent=2))
    (checkpoint_dir / 'validation_metrics.json').write_text(json.dumps(best_validation, indent=2))
    (checkpoint_dir / 'last_validation.json').write_text(json.dumps(last_validation, indent=2))
    (checkpoint_dir / 'selection.json').write_text(json.dumps({'criterion': 'maximum predicted_reward@20 subject to ndcg@20 >= initial - max_ndcg_drop', 'best_step': best_step, 'initial_ndcg@20': initial['ndcg@20'], 'minimum_ndcg@20': minimum_ndcg}, indent=2))
    return load_gpt(checkpoint_dir, device), history
