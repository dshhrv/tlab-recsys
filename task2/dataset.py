import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

PAD = 0
BOS = 1


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def device_for_run():
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


class Catalog:
    def __init__(self, train, max_items=None):
        counts = train.item_id.value_counts()
        if max_items is not None:
            counts = counts.iloc[:max_items]
        self.items = counts.index.to_numpy(np.int64)
        self.counts = counts.to_numpy(np.int64)
        self.item_to_token = {int(item): index + 2 for index, item in enumerate(self.items)}
        self.token_to_item = np.concatenate(([-1, -1], self.items))
        self.vocab_size = len(self.items) + 2
        self.popular_items = set(map(int, self.items[:max(1, int(np.ceil(len(self.items) * 0.01)))]))

    def token(self, item):
        return self.item_to_token.get(int(item), PAD)

    def save(self, path):
        Path(path).write_text(json.dumps({'items': self.items.tolist(), 'counts': self.counts.tolist()}))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        instance = cls.__new__(cls)
        instance.items = np.asarray(data['items'], dtype=np.int64)
        instance.counts = np.asarray(data['counts'], dtype=np.int64)
        instance.item_to_token = {int(item): i + 2 for i, item in enumerate(instance.items)}
        instance.token_to_item = np.concatenate(([-1, -1], instance.items))
        instance.vocab_size = len(instance.items) + 2
        instance.popular_items = set(map(int, instance.items[:max(1, int(np.ceil(len(instance.items) * 0.01)))]))
        return instance


class EventStore:
    def __init__(self, folder, catalog, max_length):
        self.folder = Path(folder)
        self.catalog = catalog
        self.max_length = max_length
        self.info = json.loads((self.folder / 'info.json').read_text())
        self.splits = {split: pd.read_csv(self.folder / f'{split}.csv') for split in ('train', 'validation', 'test')}
        self.events = pd.concat(self.splits.values(), ignore_index=True)
        self.events = self.events.sort_values('timestamp', kind='stable').reset_index(drop=True)
        self.by_user = {}
        for user, frame in self.events.groupby('user_id', sort=False):
            self.by_user[int(user)] = (frame.timestamp.to_numpy(np.int64), frame.item_id.to_numpy(np.int64))

    def history(self, user, timestamp):
        times, items = self.by_user.get(int(user), (np.empty(0, np.int64), np.empty(0, np.int64)))
        stop = np.searchsorted(times, int(timestamp), side='left')
        return items[:stop]

    def prompt(self, user, timestamp, generated=()):
        times, items = self.by_user.get(int(user), (np.empty(0, np.int64), np.empty(0, np.int64)))
        stop = np.searchsorted(times, int(timestamp), side='left')
        budget = max(0, self.max_length - 1 - len(generated))
        recent = []
        if budget:
            for item in items[:stop][::-1]:
                token = self.catalog.token(item)
                if token != PAD:
                    recent.append(token)
                    if len(recent) == budget:
                        break
        return ([BOS] + recent[::-1] + [int(token) for token in generated])[-self.max_length:]

    def seen_tokens(self, user, timestamp):
        return {self.catalog.token(item) for item in self.history(user, timestamp)} - {PAD}

    def holdout(self, split, index):
        return pd.read_csv(self.folder / f'holdout_{split}_{index}.csv').sort_values('timestamp', kind='stable').reset_index(drop=True)

    def observed(self, split):
        return self.splits[split]


def load_store(folder, max_items=30000, max_length=128, vocab_path=None):
    folder = Path(folder)
    if vocab_path is not None:
        catalog = Catalog.load(vocab_path)
    else:
        catalog = Catalog(pd.read_csv(folder / 'train.csv', usecols=['item_id']), max_items)
    return EventStore(folder, catalog, max_length)


class LMWindows(Dataset):
    def __init__(self, train, catalog, max_length=128, max_windows=None):
        self.windows = []
        for _, frame in train.groupby('user_id', sort=False):
            frame = frame.sort_values('timestamp', kind='stable')
            items = frame.item_id.to_numpy(np.int64)
            times = frame.timestamp.to_numpy(np.int64)
            active = np.array([item in catalog.item_to_token for item in items])
            tokens = np.array([catalog.token(item) for item in items[active]], dtype=np.int64)
            times = times[active]
            if len(tokens) == 0:
                continue
            sequence = np.concatenate(([BOS], tokens))
            sequence_times = np.concatenate(([-1], times))
            stride = max_length - 1
            for start in range(0, len(sequence) - 1, stride):
                window = sequence[start:start + max_length]
                window_times = sequence_times[start:start + max_length]
                if len(window) < 2:
                    continue
                labels = window.copy()
                labels[0] = -100
                labels[1:][window_times[1:] == window_times[:-1]] = -100
                self.windows.append((window, labels))
        if max_windows is not None and len(self.windows) > max_windows:
            positions = np.linspace(0, len(self.windows) - 1, max_windows, dtype=int)
            self.windows = [self.windows[i] for i in positions]
        self.max_length = max_length

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        tokens, labels = self.windows[index]
        inputs = np.zeros(self.max_length, dtype=np.int64)
        targets = np.full(self.max_length, -100, dtype=np.int64)
        attention = np.zeros(self.max_length, dtype=np.int64)
        inputs[:len(tokens)] = tokens
        targets[:len(labels)] = labels
        attention[:len(tokens)] = 1
        return tuple(torch.from_numpy(value) for value in (inputs, attention, targets))


def prompt_batch(prompts, max_length, device):
    width = min(max_length, max(len(prompt) for prompt in prompts))
    ids = np.zeros((len(prompts), width), dtype=np.int64)
    mask = np.zeros_like(ids)
    for index, prompt in enumerate(prompts):
        values = np.asarray(prompt[-width:], dtype=np.int64)
        ids[index, :len(values)] = values
        mask[index, :len(values)] = 1
    return torch.as_tensor(ids, device=device), torch.as_tensor(mask, device=device)


def prepare_reward_cache(folder, output):
    folder, output = Path(folder), Path(output)
    if output.exists():
        return output
    raw = pq.ParquetFile(folder / 'raw' / 'multi_event.parquet')
    users = set()
    for batch in raw.iter_batches(columns=['uid'], batch_size=1_000_000):
        users.update(np.unique(batch.column(0).to_numpy()).tolist())
    all_users = np.array(sorted(users), dtype=np.uint64)
    info = json.loads((folder / 'info.json').read_text())
    selected = all_users[np.argsort(all_users * np.uint64(11400714819323198485))[:info['selected_user_count']]].astype(np.uint32)
    selected_array = pa.array(selected)
    columns = ['uid', 'item_id', 'timestamp', 'event_type', 'played_ratio_pct', 'track_length_seconds', 'is_organic']
    frames = []
    for batch in raw.iter_batches(columns=columns, batch_size=1_000_000):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table['uid'], value_set=selected_array))
        if table.num_rows:
            frames.append(table.to_pandas())
    events = pd.concat(frames, ignore_index=True)
    users = pd.Index(sorted(events.uid.unique()))
    items = pd.Index(sorted(events.item_id.unique()))
    events['user_id'] = users.get_indexer(events.uid)
    events['item_id'] = items.get_indexer(events.item_id)
    events['timestamp'] = events.timestamp.astype('int64') * 5
    events['event_type'] = events.event_type.astype(str)
    events = events.sort_values('timestamp', kind='stable').reset_index(drop=True)
    prepared = pd.concat([pd.read_csv(folder / f'{split}.csv') for split in ('train', 'validation', 'test')], ignore_index=True)
    keys = ['user_id', 'item_id', 'timestamp', 'event_type']
    if len(events) != len(prepared) or not events[keys].equals(prepared[keys]):
        raise ValueError('Raw reward events do not align with the existing task1 splits')
    output.parent.mkdir(parents=True, exist_ok=True)
    events[['user_id', 'item_id', 'timestamp', 'event_type', 'played_ratio_pct', 'track_length_seconds', 'is_organic']].to_parquet(output, index=False)
    return output


def reward_splits(cache, info):
    events = pd.read_parquet(cache)
    train_end = info['split_sizes']['train']
    validation_end = train_end + info['split_sizes']['validation']
    return {'train': events.iloc[:train_end], 'validation': events.iloc[train_end:validation_end], 'test': events.iloc[validation_end:]}
