import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dataset import PAD, prompt_batch, set_seed


@dataclass
class RewardSettings:
    embedding_dim: int = 64
    hidden_dim: int = 128
    epochs: int = 3
    patience: int = 3
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_train_rows: int = 150000
    max_eval_rows: int = 20000
    duration_scale_seconds: float = 600.0
    like_weight: float = 0.3
    seed: int = 42


def reward_targets(frame, duration_scale_seconds):
    ratio = pd.to_numeric(frame.played_ratio_pct, errors='coerce').fillna(0).to_numpy(np.float32)
    ratio = np.clip(ratio, 0, 100) / 100.0
    length = pd.to_numeric(frame.track_length_seconds, errors='coerce').fillna(0).to_numpy(np.float32)
    duration = np.clip(ratio * np.maximum(length, 0) / duration_scale_seconds, 0, 1)
    event = frame.event_type.astype(str).to_numpy()
    listen = event == 'listen'
    explicit = np.isin(event, ['like', 'dislike', 'unlike', 'undislike'])
    like = (event == 'like').astype(np.float32)
    duration_mask = listen & (length > 0)
    return ratio, duration, like, listen, duration_mask, explicit


class RewardDataset(Dataset):
    def __init__(self, frame, store, settings, max_rows=None, seed=42):
        frame = frame[frame.item_id.isin(store.catalog.item_to_token)].copy()
        if max_rows and len(frame) > max_rows:
            explicit = frame[frame.event_type.isin(['like', 'dislike', 'unlike', 'undislike'])]
            listen = frame[frame.event_type.eq('listen')]
            keep_explicit = explicit.sample(n=min(len(explicit), max_rows // 3), random_state=seed)
            keep_listen = listen.sample(n=min(len(listen), max_rows - len(keep_explicit)), random_state=seed)
            frame = pd.concat([keep_explicit, keep_listen]).sort_values('timestamp', kind='stable')
        self.frame = frame.reset_index(drop=True)
        self.store = store
        self.settings = settings
        self.targets = reward_targets(self.frame, settings.duration_scale_seconds)
        self.tokens = self.frame.item_id.map(store.catalog.item_to_token).to_numpy(np.int64)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        prompt = self.store.prompt(row.user_id, row.timestamp)
        ids = np.zeros(self.store.max_length, dtype=np.int64)
        mask = np.zeros(self.store.max_length, dtype=np.int64)
        ids[:len(prompt)] = prompt
        mask[:len(prompt)] = 1
        ratio, duration, like, listen, duration_mask, explicit = self.targets
        labels = np.array([ratio[index], duration[index], like[index]], dtype=np.float32)
        labels_mask = np.array([listen[index], duration_mask[index], explicit[index]], dtype=np.float32)
        return (torch.from_numpy(ids), torch.from_numpy(mask), torch.tensor(self.tokens[index]), torch.from_numpy(labels), torch.from_numpy(labels_mask))


class RewardNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD)
        self.layers = nn.Sequential(nn.Linear(embedding_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 3))

    def forward(self, history, attention, candidate):
        embedded = self.embedding(history)
        active = (attention * (history != 1)).float()
        pooled = (embedded * active.unsqueeze(-1)).sum(dim=1) / active.sum(dim=1).clamp_min(1).unsqueeze(-1)
        item = self.embedding(candidate)
        return torch.sigmoid(self.layers(torch.cat((pooled, item, pooled * item), dim=-1)))


def loss_for_reward(prediction, targets, masks):
    regression = (prediction[:, :2] - targets[:, :2]).square()
    classification = nn.functional.binary_cross_entropy(prediction[:, 2].clamp(1e-6, 1 - 1e-6), targets[:, 2], reduction='none')
    terms = [(regression[:, column] * masks[:, column]).sum() / masks[:, column].sum().clamp_min(1) for column in range(2)]
    terms.append((classification * masks[:, 2]).sum() / masks[:, 2].sum().clamp_min(1))
    return sum(terms)



def train_reward(store, reward_splits, settings, checkpoint_dir, device, task=None):
    set_seed(settings.seed)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / 'settings.json').write_text(json.dumps(asdict(settings), indent=2))
    train = RewardDataset(reward_splits['train'], store, settings, settings.max_train_rows, settings.seed)
    if len(train) == 0:
        raise ValueError('No reward training events in active catalog')
    loader = DataLoader(train, batch_size=settings.batch_size, shuffle=True, generator=torch.Generator().manual_seed(settings.seed))
    model = RewardNetwork(store.catalog.vocab_size, settings.embedding_dim, settings.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    best, history = float('inf'), []
    bad_epochs = 0
    for epoch in range(settings.epochs):
        model.train()
        losses = []
        for ids, attention, item, targets, masks in loader:
            ids, attention, item = ids.to(device), attention.to(device), item.to(device)
            targets, masks = targets.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for_reward(model(ids, attention, item), targets, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate_reward(model, reward_splits['validation'], store, settings, device)
        row = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), **metrics}
        history.append(row)
        if task:
            task.get_logger().report_scalar('reward/loss', f'seed_{settings.seed}', row['train_loss'], epoch + 1)
            if 'ratio_rmse' in row:
                task.get_logger().report_scalar('reward/ratio_rmse', f'seed_{settings.seed}', row['ratio_rmse'], epoch + 1)
        criterion = metrics.get('ratio_rmse', row['train_loss'])
        if criterion < best:
            best = criterion
            bad_epochs = 0
            torch.save(model.cpu().state_dict(), checkpoint_dir / 'best.pt')
            model.to(device)
        else:
            bad_epochs += 1
            if bad_epochs >= settings.patience:
                print(f'Early stopping at epoch {epoch + 1}. Best validation ratio RMSE: {best:.4f}')
                break
    (checkpoint_dir / 'history.json').write_text(json.dumps(history, indent=2))
    model.load_state_dict(torch.load(checkpoint_dir / 'best.pt', map_location=device, weights_only=True))
    return model, history


def load_reward(checkpoint_dir, vocab_size, device):
    checkpoint_dir = Path(checkpoint_dir)
    settings = RewardSettings(**json.loads((checkpoint_dir / 'settings.json').read_text()))
    model = RewardNetwork(vocab_size, settings.embedding_dim, settings.hidden_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / 'best.pt', map_location=device, weights_only=True))
    model.eval()
    return model, settings


def predict_components(model, prompts, candidates, max_length, device, batch_size=512):
    model.to(device).eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(candidates), batch_size):
            ids, attention = prompt_batch(prompts[start:start + batch_size], max_length, device)
            items = torch.as_tensor(candidates[start:start + batch_size], dtype=torch.long, device=device)
            values.append(model(ids, attention, items).cpu().numpy())
    return np.concatenate(values) if values else np.empty((0, 3), dtype=np.float32)


def reward_value(components, function='ratio', like_weight=0.3):
    if function == 'ratio':
        return components[..., 0]
    if function == 'duration':
        return components[..., 1]
    if function == 'combined':
        return (1 - like_weight) * components[..., 0] + like_weight * components[..., 2]
    raise ValueError(f'Unknown reward function: {function}')


def evaluate_reward(model, frame, store, settings, device):
    dataset = RewardDataset(frame, store, settings, settings.max_eval_rows, settings.seed)
    loader = DataLoader(dataset, batch_size=settings.batch_size)
    predictions, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for ids, attention, item, labels, label_masks in loader:
            predictions.append(model(ids.to(device), attention.to(device), item.to(device)).cpu().numpy())
            targets.append(labels.numpy())
            masks.append(label_masks.numpy().astype(bool))
    if not predictions:
        return {'events': 0}
    predictions, targets, masks = np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
    metrics = {'events': len(predictions)}
    for index, name in enumerate(('ratio', 'duration')):
        selected = masks[:, index]
        if selected.any():
            truth, predicted = targets[selected, index], predictions[selected, index]
            metrics[f'{name}_mae'] = float(mean_absolute_error(truth, predicted))
            metrics[f'{name}_rmse'] = float(np.sqrt(mean_squared_error(truth, predicted)))
            if name == 'duration':
                metrics['duration_seconds_mae'] = metrics[f'{name}_mae'] * settings.duration_scale_seconds
                metrics['duration_seconds_rmse'] = metrics[f'{name}_rmse'] * settings.duration_scale_seconds
            if len(truth) > 1 and np.std(truth) > 0 and np.std(predicted) > 0:
                metrics[f'{name}_spearman'] = float(spearmanr(truth, predicted).statistic)
    selected = masks[:, 2]
    if selected.any() and len(np.unique(targets[selected, 2])) == 2:
        metrics['like_roc_auc'] = float(roc_auc_score(targets[selected, 2], predictions[selected, 2]))
        metrics['like_pr_auc'] = float(average_precision_score(targets[selected, 2], predictions[selected, 2]))
    metrics['like_examples'] = int(selected.sum())
    return metrics
