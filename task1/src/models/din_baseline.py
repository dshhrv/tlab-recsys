import time
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.catboost_baseline import CatBoostBaseline


class Dice(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.BatchNorm1d(width, eps=1e-8)
        self.alpha = nn.Parameter(torch.zeros(width))

    def forward(self, values):
        shape = values.shape
        flat = values.reshape(-1, shape[-1])
        probability = torch.sigmoid(self.norm(flat))
        return (probability * flat + (1 - probability) * self.alpha * flat).reshape(shape)


class DINNetwork(nn.Module):
    def __init__(self, num_users, num_items, categories, item_width, category_width, user_width):
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, item_width, padding_idx=0)
        self.category_embedding = nn.Embedding(int(categories.max()) + 1, category_width, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users + 1, user_width, padding_idx=0)
        self.register_buffer("item_categories", torch.as_tensor(categories, dtype=torch.long))
        width = item_width + category_width
        self.activation = nn.Sequential(nn.Linear(width * 4, 64), Dice(64), nn.Linear(64, 32), Dice(32), nn.Linear(32, 1))
        self.output = nn.Sequential(nn.Linear(width * 2 + user_width + 1, 128), Dice(128), nn.Linear(128, 64), Dice(64), nn.Linear(64, 1))

    def forward(self, users, items, history):
        candidate = torch.cat((self.item_embedding(items), self.category_embedding(self.item_categories[items])), dim=-1)
        past = torch.cat((self.item_embedding(history), self.category_embedding(self.item_categories[history])), dim=-1)
        target = candidate.unsqueeze(1).expand_as(past)
        activation_input = torch.cat((target, past, target - past, target * past), dim=-1)
        weights = torch.sigmoid(self.activation(activation_input).squeeze(-1)) * history.ne(0)
        interest = (past * weights.unsqueeze(-1)).sum(dim=1)
        history_length = torch.log1p(history.ne(0).sum(dim=1).float()).unsqueeze(-1)
        features = torch.cat((candidate, interest, self.user_embedding(users), history_length), dim=-1)
        return self.output(features).squeeze(-1)


def train_sequences(folder, max_history):
    pairs = pd.read_parquet(folder / "train_pairs.parquet").sort_values(["timestamp", "query_id"], kind="stable").reset_index(drop=True)
    events = pd.read_csv(folder / "train.csv").sort_values("timestamp", kind="stable")
    event_users = events.user_id.to_numpy()
    event_items = events.item_id.to_numpy()
    event_times = events.timestamp.to_numpy()
    histories = defaultdict(lambda: deque(maxlen=max_history))
    sequences = np.zeros((len(pairs), max_history), dtype="int64")
    cursor = 0
    for index, pair in enumerate(pairs.itertuples(index=False)):
        while cursor < len(events) and event_times[cursor] < pair.timestamp:
            histories[int(event_users[cursor])].append(int(event_items[cursor]) + 1)
            cursor += 1
        recent = list(histories[pair.user_id])
        if recent:
            sequences[index, -len(recent):] = recent
    users = pairs.user_id.to_numpy(dtype="int64") + 1
    items = pairs.item_id.to_numpy(dtype="int64") + 1
    labels = pairs.label.to_numpy(dtype="float32")
    return users, items, sequences, labels


def sequence_batches(folder, split, holdout, max_history, batch_size):
    queries = pd.read_csv(folder / f"holdout_{split}_{holdout}.csv").sort_values("timestamp", kind="stable")
    files = ["train.csv", "validation.csv"] + (["test.csv"] if split == "test" else [])
    events = pd.concat([pd.read_csv(folder / file) for file in files], ignore_index=True).sort_values("timestamp", kind="stable")
    catalog = np.sort(pd.read_csv(folder / "train.csv", usecols=["item_id"]).item_id.unique())
    event_users = events.user_id.to_numpy()
    event_items = events.item_id.to_numpy()
    event_times = events.timestamp.to_numpy()
    histories = defaultdict(lambda: deque(maxlen=max_history))
    seen = defaultdict(set)
    cursor = 0
    for query in queries.itertuples(index=False):
        while cursor < len(events) and event_times[cursor] < query.timestamp:
            user = int(event_users[cursor])
            item = int(event_items[cursor])
            histories[user].append(item + 1)
            seen[user].add(item)
            cursor += 1
        candidates = np.concatenate(([query.item_id], catalog[(catalog != query.item_id) & ~np.isin(catalog, list(seen[query.user_id]))]))
        history = np.zeros(max_history, dtype="int64")
        recent = list(histories[query.user_id])
        if recent:
            history[-len(recent):] = recent
        for start in range(0, len(candidates), batch_size):
            end = min(start + batch_size, len(candidates))
            yield query.query_id, query.user_id + 1, candidates[start:end] + 1, history, start == 0, end == len(candidates), len(candidates)


class DINBaseline(CatBoostBaseline):
    def __init__(self, folder, seed=42, epochs=5, batch_size=512, learning_rate=0.001, max_history=50, item_width=32, category_width=16, user_width=16, eval_batch_size=1024):
        super().__init__(folder, seed)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.params = {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate, "max_history": max_history, "item_width": item_width, "category_width": category_width, "user_width": user_width, "eval_batch_size": eval_batch_size, "device": str(self.device)}

    def fit(self):
        torch.manual_seed(self.seed)
        items = pd.read_csv(self.folder / "items.csv").sort_values("item_id")
        users = pd.read_csv(self.folder / "users.csv")
        categories = pd.factorize(items.category.fillna("unknown").astype(str), sort=True)[0].astype("int64") + 1
        categories = np.concatenate(([0], categories))
        self.model = DINNetwork(len(users), len(items), categories, self.params["item_width"], self.params["category_width"], self.params["user_width"]).to(self.device)
        started = time.monotonic()
        train = train_sequences(self.folder, self.params["max_history"])
        tensors = [torch.as_tensor(values) for values in train]
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TensorDataset(*tensors), batch_size=self.params["batch_size"], shuffle=True, generator=generator, drop_last=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params["learning_rate"], weight_decay=1e-5)
        for epoch in range(self.params["epochs"]):
            self.model.train()
            for user_ids, item_ids, histories, labels in loader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                histories = histories.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(user_ids, item_ids, histories)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()
        self.training_seconds = time.monotonic() - started
        self.feature_columns = ["user_embedding", "candidate_item_embedding", "candidate_category_embedding", "history_item_embeddings", "history_category_embeddings", "local_activation", "history_length"]
        return self

    def predict(self, split, holdout=0):
        self.model.eval()
        for query_id, user, items, history, first, last, count in sequence_batches(self.folder, split, holdout, self.params["max_history"], self.params["eval_batch_size"]):
            users = torch.full((len(items),), int(user), dtype=torch.long, device=self.device)
            candidates = torch.as_tensor(items, dtype=torch.long, device=self.device)
            histories = torch.as_tensor(np.broadcast_to(history, (len(items), len(history))).copy(), dtype=torch.long, device=self.device)
            with torch.no_grad():
                scores = self.model(users, candidates, histories).cpu().numpy()
            yield query_id, items - 1, scores, first, last, count
