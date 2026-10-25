import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel

from dataset import LMWindows, PAD, prompt_batch, set_seed


@dataclass
class GPTSettings:
    max_items: int = 30000
    max_length: int = 128
    embedding_dim: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    max_windows: int | None = None
    validation_queries: int = 2000
    seed: int = 42


def build_model(catalog, settings):
    config = GPT2Config(vocab_size=catalog.vocab_size, n_positions=settings.max_length, n_ctx=settings.max_length, n_embd=settings.embedding_dim, n_layer=settings.layers, n_head=settings.heads, embd_pdrop=settings.dropout, attn_pdrop=settings.dropout, resid_pdrop=settings.dropout, pad_token_id=PAD, bos_token_id=1, eos_token_id=None, use_cache=False, loss_type='ForCausalLM')
    model = GPT2LMHeadModel(config)
    model.loss_type = 'ForCausalLM'
    return model


def last_logits(model, ids, attention):
    output = model(input_ids=ids, attention_mask=attention)
    lengths = attention.sum(dim=1) - 1
    return output.logits[torch.arange(len(ids), device=ids.device), lengths]


def score_prompts(model, prompts, max_length, device, batch_size=32):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            ids, attention = prompt_batch(prompts[start:start + batch_size], max_length, device)
            result.append(last_logits(model, ids, attention).float().cpu().numpy())
    return np.concatenate(result)


def validation_loss(model, store, max_queries, batch_size, device):
    frame = store.splits['validation']
    frame = frame[frame.item_id.isin(store.catalog.item_to_token)]
    if max_queries and len(frame) > max_queries:
        frame = frame.iloc[np.linspace(0, len(frame) - 1, max_queries, dtype=int)]
    targets = frame.item_id.map(store.catalog.item_to_token).to_numpy(np.int64)
    prompts = [store.prompt(row.user_id, row.timestamp) for row in frame.itertuples(index=False)]
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            ids, attention = prompt_batch(prompts[start:start + batch_size], store.max_length, device)
            logits = last_logits(model, ids, attention)
            labels = torch.tensor(targets[start:start + batch_size].copy(), device=device)
            total += torch.nn.functional.cross_entropy(logits, labels, reduction='sum').item()
            count += len(labels)
    return total / count if count else float('nan')


def train_gpt(store, settings, checkpoint_dir, device, task=None):
    set_seed(settings.seed)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    store.catalog.save(checkpoint_dir / 'vocab.json')
    (checkpoint_dir / 'settings.json').write_text(json.dumps(asdict(settings), indent=2))
    dataset = LMWindows(store.splits['train'], store.catalog, settings.max_length, settings.max_windows)
    if len(dataset) == 0:
        raise ValueError('No train windows in active catalog')
    generator = torch.Generator().manual_seed(settings.seed)
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True, generator=generator)
    model = build_model(store.catalog, settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    history, best = [], float('inf')
    for epoch in range(settings.epochs):
        model.train()
        total, batches = 0.0, 0
        for ids, attention, labels in loader:
            ids, attention, labels = ids.to(device), attention.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, attention_mask=attention, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        train_loss = total / batches
        val_loss = validation_loss(model, store, settings.validation_queries, settings.batch_size, device)
        row = {'epoch': epoch + 1, 'train_loss': train_loss, 'validation_loss': val_loss}
        history.append(row)
        if task:
            for key in ('train_loss', 'validation_loss'):
                task.get_logger().report_scalar('gpt2/loss', key, row[key], epoch + 1)
        if val_loss < best:
            best = val_loss
            model.save_pretrained(checkpoint_dir / 'best', safe_serialization=True)
    (checkpoint_dir / 'history.json').write_text(json.dumps(history, indent=2))
    return GPT2LMHeadModel.from_pretrained(checkpoint_dir / 'best').to(device), history


def load_gpt(checkpoint_dir, device):
    return GPT2LMHeadModel.from_pretrained(Path(checkpoint_dir) / 'best').to(device)
