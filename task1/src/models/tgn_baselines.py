import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier

from src.models.catboost_baseline import CATEGORICAL, KEYS, build_features, full_catalog_batches
from src.models.ranking_metrics import TOP_K, TopKItems, coverage_values
from src.models.tgn import TGNModel
from src.models.tgn_support import set_training_task
from tgn.model.tgn import TGN
from tgn.utils.data import Data
from tgn.utils.utils import NeighborFinder, get_neighbor_finder


class TrainData:
    def __init__(self, frame, info, catalog):
        self._df = frame
        self.name = info["name"]
        self.n_users = info["num_users"]
        self.n_items = info["num_items"]
        self.candidate_items = catalog


def read_events(folder, parts):
    return pd.concat([pd.read_csv(folder / f"{part}.csv", usecols=["user_id", "item_id", "timestamp"]) for part in parts], ignore_index=True).sort_values("timestamp", kind="stable").reset_index(drop=True)


def make_data(events, n_users):
    sources = events.user_id.to_numpy(np.int64) + 1
    destinations = events.item_id.to_numpy(np.int64) + n_users + 1
    timestamps = events.timestamp.to_numpy(np.float64)
    return Data(sources, destinations, timestamps, np.zeros(len(events), dtype=np.int64), np.ones(len(events), dtype=np.float32))


def make_tgn(frame, info, catalog, params, epochs, task=None, phase="tgn/full"):
    settings = dict(params)
    settings["n_epochs"] = epochs
    settings["num_candidates"] = 2
    settings["loss_type"] = "bpr"
    model = TGNModel(**settings)
    set_training_task(task, phase)
    model.fit(TrainData(frame, info, catalog))
    set_training_task(None)
    model.model.eval()
    return model


def make_cold_tgn(info, params):
    torch.manual_seed(params["seed"])
    nodes = info["num_users"] + info["num_items"] + 1
    dimension = params["memory_dim"]
    model = TGN(neighbor_finder=NeighborFinder([[] for _ in range(nodes)]), node_features=np.zeros((nodes, dimension), dtype=np.float32), edge_features=np.zeros((1, dimension), dtype=np.float32), device=torch.device(params["device"]), n_layers=1, n_heads=2, dropout=0.1, use_memory=True, memory_update_at_start=True, message_dimension=100, memory_dimension=dimension, embedding_module_type="graph_attention", message_function="identity", mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0, std_time_shift_dst=1, n_neighbors=params["n_degree"], aggregator_type="attention", memory_updater_type="gru", use_edge_features=False, trainable_node_features=True).to(params["device"])
    model.eval()
    return model


class Replay:
    def __init__(self, model, events, start, n_users, neighbors):
        self.model = model
        self.events = events
        self.cursor = start
        self.n_users = n_users
        self.neighbors = neighbors
        self.snapshot = None
        self.model.set_neighbor_finder(get_neighbor_finder(make_data(events, n_users), uniform=False, max_node_idx=model.n_nodes - 1))
        self.model.eval()

    def advance(self, timestamp):
        times = self.events.timestamp.to_numpy()
        while self.cursor < len(self.events) and times[self.cursor] < timestamp:
            end = self.cursor + 1
            while end < len(self.events) and times[end] == times[self.cursor]:
                end += 1
            group = self.events.iloc[self.cursor:end]
            with torch.no_grad():
                self.model.process_real_interaction(group.user_id.to_numpy(np.int64) + 1, group.item_id.to_numpy(np.int64) + self.n_users + 1, group.timestamp.to_numpy(np.float64), np.zeros(len(group), dtype=np.int64), self.neighbors)
            self.cursor = end
            self.snapshot = None

    def embeddings(self, user, items, timestamp):
        sources = np.array([user + 1], dtype=np.int64)
        candidates = np.asarray(items, dtype=np.int64) + self.n_users + 1
        times = np.array([timestamp], dtype=np.float64)
        with torch.no_grad():
            if self.snapshot is None and self.model.use_memory:
                self.snapshot = self.model.get_updated_memory(list(range(self.model.n_nodes)), self.model.memory.messages)
            user_embedding, item_embeddings = self.model.compute_temporal_embeddings_read_only(sources, candidates, times, self.neighbors, self.snapshot)
        return user_embedding[0].detach().cpu().numpy(), item_embeddings.detach().cpu().numpy()


def embedding_frame(user_embedding, item_embeddings):
    users = np.broadcast_to(user_embedding, item_embeddings.shape)
    products = users * item_embeddings
    dimension = len(user_embedding)
    values = {f"tgn_user_{index}": users[:, index] for index in range(dimension)}
    values.update({f"tgn_item_{index}": item_embeddings[:, index] for index in range(dimension)})
    values.update({f"tgn_product_{index}": products[:, index] for index in range(dimension)})
    values["tgn_dot"] = products.sum(axis=1)
    return pd.DataFrame(values)


def fold_boundaries(events, folds):
    times = np.unique(events.timestamp.to_numpy())
    return np.array_split(times, folds)


def build_oof(folder, info, catalog, params, folds, task):
    events = read_events(folder, ["train"])
    pairs = pd.read_parquet(folder / "train_pairs.parquet").sort_values(["timestamp", "query_id"], kind="stable").reset_index(drop=True)
    rows = []
    for index, fold_times in enumerate(fold_boundaries(events, folds)):
        start_time = fold_times[0]
        end_time = fold_times[-1]
        prefix = events[events.timestamp < start_time]
        scoring_pairs = pairs[(pairs.timestamp >= start_time) & (pairs.timestamp <= end_time)]
        if scoring_pairs.empty:
            continue
        if index == 0:
            model = make_cold_tgn(info, params)
            start = 0
        else:
            model = make_tgn(prefix, info, catalog, params, params["n_epochs"], task, f"tgn/oof_{index}").model
            start = len(prefix)
        replay = Replay(model, events, start, info["num_users"], params["n_degree"])
        for (_, _, timestamp), group in scoring_pairs.groupby(["query_id", "user_id", "timestamp"], sort=False):
            replay.advance(timestamp)
            user = int(group.user_id.iloc[0])
            user_embedding, item_embeddings = replay.embeddings(user, group.item_id.to_numpy(), timestamp)
            block = pd.concat([group[KEYS].reset_index(drop=True), embedding_frame(user_embedding, item_embeddings)], axis=1)
            rows.append(block)
    result = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "query_id"], kind="stable").reset_index(drop=True)
    assert len(result) == len(pairs)
    assert not result[KEYS].duplicated().any()
    return result


def artifact_paths(folder, seed):
    root = folder / f"tgn_artifacts_seed_{seed}"
    return root, root / "full_tgn.pt", root / "train_oof.parquet", root / "manifest.json"


def ensure_tgn(folder, params, folds=5, need_oof=False, force=False, task=None):
    folder = Path(folder)
    info = json.loads((folder / "info.json").read_text())
    catalog = np.sort(pd.read_csv(folder / "train.csv", usecols=["item_id"]).item_id.unique())
    root, full_path, oof_path, manifest_path = artifact_paths(folder, params["seed"])
    config = {"params": params, "folds": folds, "train_rows": info["split_sizes"]["train"], "catalog_items": len(catalog), "oof_protocol": "expanding time folds; first fold uses an untrained causal TGN"}
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if force or existing != config:
        full_path.unlink(missing_ok=True)
        oof_path.unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    if need_oof and not oof_path.exists():
        build_oof(folder, info, catalog, params, folds, task).to_parquet(oof_path, index=False)
    if not full_path.exists():
        train = read_events(folder, ["train"])
        trained = make_tgn(train, info, catalog, params, params["n_epochs"], task)
        torch.save(trained.model, full_path)
    manifest_path.write_text(json.dumps(config, indent=2))
    model = torch.load(full_path, map_location=params["device"], weights_only=False)
    model.eval()
    return model, oof_path


def metric_summary(values):
    summary = {}
    for key in values[0]:
        samples = [row[key] for row in values]
        summary[f"{key}_mean"] = float(np.mean(samples))
        summary[f"{key}_std"] = float(np.std(samples, ddof=1)) if len(samples) > 1 else None
    return summary


def rank_metrics(ranks, counts, cold, recommended, catalog_size):
    result = {"queries": len(ranks), "cold_targets_skipped": cold, "candidates_min": int(min(counts)) if counts else 0, "candidates_max": int(max(counts)) if counts else 0}
    values = np.asarray(ranks)
    for k in TOP_K:
        result[f"recall@{k}"] = float(np.mean(values <= k)) if len(values) else 0.0
        result[f"ndcg@{k}"] = float(np.mean(np.where(values <= k, 1 / np.log2(values + 1), 0))) if len(values) else 0.0
    result.update(coverage_values(recommended, catalog_size))
    return result


class TGNBaseline:
    def __init__(self, folder, seed=42, device=None, n_epochs=10, memory_dim=64, n_degree=10, batch_size=256, memory_batch_size=32, item_batch_size=2048, folds=5):
        self.folder = Path(folder)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.params = {"seed": seed, "device": device, "n_epochs": n_epochs, "memory_dim": memory_dim, "n_degree": n_degree, "batch_size": batch_size, "memory_batch_size": memory_batch_size}
        self.seed = seed
        self.item_batch_size = item_batch_size
        self.folds = folds
        self.info = json.loads((self.folder / "info.json").read_text())
        self.catalog = np.sort(pd.read_csv(self.folder / "train.csv", usecols=["item_id"]).item_id.unique())
        self.catalog_set = set(self.catalog)
        self.model = None
        self.task = None

    def fit(self, force=False):
        started = time.monotonic()
        self.model, _ = ensure_tgn(self.folder, self.params, self.folds, False, force, self.task)
        if self.task:
            self.task.get_logger().report_single_value("training_seconds", time.monotonic() - started)
        return self

    def replay(self, split):
        parts = ["train", "validation"] + (["test"] if split == "test" else [])
        events = read_events(self.folder, parts)
        return Replay(self.model, events, self.info["split_sizes"]["train"], self.info["num_users"], self.params["n_degree"]), events

    def score(self, replay, user, items, timestamp):
        user_embedding, item_embeddings = replay.embeddings(user, items, timestamp)
        return item_embeddings @ user_embedding

    def candidate_batches(self, split, holdout=0):
        replay, events = self.replay(split)
        queries = pd.read_csv(self.folder / f"holdout_{split}_{holdout}.csv").sort_values("timestamp", kind="stable")
        seen = {user: set(group.item_id) for user, group in events.iloc[:self.info["split_sizes"]["train"]].groupby("user_id")}
        cursor = self.info["split_sizes"]["train"]
        for query in queries.itertuples(index=False):
            replay.advance(query.timestamp)
            while cursor < replay.cursor:
                seen.setdefault(int(events.user_id.iloc[cursor]), set()).add(int(events.item_id.iloc[cursor]))
                cursor += 1
            if query.item_id not in self.catalog_set:
                yield replay, query, None, True, True, 0
                continue
            available = np.concatenate(([query.item_id], self.catalog[(self.catalog != query.item_id) & ~np.isin(self.catalog, list(seen.get(query.user_id, ())))]))
            first = True
            for start in range(0, len(available), self.item_batch_size):
                batch = available[start:start + self.item_batch_size]
                yield replay, query, batch, first, start + len(batch) == len(available), len(available)
                first = False

    def predict(self, split, holdout=0):
        for replay, query, batch, first, last, count in self.candidate_batches(split, holdout):
            scores = None if batch is None else self.score(replay, query.user_id, batch, query.timestamp)
            yield query.query_id, batch, scores, first, last, count

    def evaluate(self, split, holdout=0):
        memory = self.model.memory.backup_memory()
        ranks, counts, cold = [], [], 0
        recommended = {k: set() for k in TOP_K}
        positive, higher = None, 0
        for _, items, scores, first, last, count in self.predict(split, holdout):
            if scores is None:
                cold += 1
                continue
            if first:
                positive = scores[0]
                higher = int(np.count_nonzero(scores[1:] > positive))
                top_items = TopKItems()
            else:
                higher += int(np.count_nonzero(scores > positive))
            top_items.add(items, scores)
            if last:
                ranks.append(higher + 1)
                counts.append(count)
                for k in TOP_K:
                    recommended[k].update(top_items.items[:k])
        self.model.memory.restore_memory(memory)
        return rank_metrics(ranks, counts, cold, recommended, len(self.catalog))

    def run(self, dataset, log_clearml=True, run_test=True, force=False):
        if log_clearml:
            from clearml import Task
            if not (Path.home() / "clearml.conf").exists() and not (Path.home() / ".clearml.conf").exists() and not os.getenv("CLEARML_API_ACCESS_KEY") and not os.getenv("CLEARML_CONFIG_FILE"):
                os.environ["CLEARML_CACHE_DIR"] = str(self.folder / "clearml_cache")
                Task.set_offline(True)
            self.task = Task.init(project_name="CatBoost RecSys Baselines", task_name=f"{dataset} {self.__class__.__name__} seed {self.seed}", reuse_last_task_id=False)
            self.task.connect({"dataset": dataset, "model": self.__class__.__name__, "params": self.params, "catalog_items": len(self.catalog), "folds": self.folds}, name="General", ignore_remote_overrides=True)
        self.fit(force)
        results = self.results(run_test)
        self.finish(results)
        return results

    def results(self, run_test):
        results = {}
        for split in ["validation", "test"] if run_test else ["validation"]:
            values = [self.evaluate(split, holdout) for holdout in range(self.info["holdout_counts"][split])]
            results[split] = {"protocol": self.info["holdout_protocols"][split], "catalog_scope": self.info["evaluation_catalog_scope"], "train_catalog_items": len(self.catalog), "holdouts": values, "summary": metric_summary(values)}
        return results

    def finish(self, results):
        if self.task:
            for split, section in results.items():
                for index, metrics in enumerate(section["holdouts"]):
                    for name, value in metrics.items():
                        self.task.get_logger().report_scalar(title=f"{split}/{name}", series="holdout", value=value, iteration=index)
                for name, value in section["summary"].items():
                    if value is not None:
                        self.task.get_logger().report_single_value(f"{split}/{name}", value)
            self.task.close()
            self.task = None
        name = getattr(self, "output_name", self.__class__.__name__)
        path = self.folder / f"{name}_seed_{self.seed}_full_catalog_metrics.json"
        path.write_text(json.dumps(results, indent=2))


class CatBoostTGNBaseline(TGNBaseline):
    def __init__(self, folder, manual_features=True, iterations=600, depth=7, learning_rate=0.05, **kwargs):
        super().__init__(folder, **kwargs)
        self.manual_features = manual_features
        self.catboost_params = {"iterations": iterations, "depth": depth, "learning_rate": learning_rate, "loss_function": "Logloss", "random_seed": self.seed, "task_type": "CPU", "thread_count": 2, "verbose": False, "allow_writing_files": False}
        self.catboost = None
        self.feature_columns = None
        self.output_name = "CatBoostTGNManual" if manual_features else "CatBoostTGNEmbeddings"

    def fit(self, force=False):
        started = time.monotonic()
        self.model, oof_path = ensure_tgn(self.folder, self.params, self.folds, True, force, self.task)
        oof = pd.read_parquet(oof_path)
        if self.manual_features:
            manual = build_features(self.folder, "train")
            train = manual.merge(oof.drop(columns=["label"]), on=["query_id", "user_id", "item_id", "timestamp"], validate="one_to_one")
        else:
            train = oof
        self.feature_columns = [name for name in train.columns if name not in KEYS]
        self.catboost = CatBoostClassifier(**self.catboost_params)
        self.catboost.fit(train[self.feature_columns], train.label, cat_features=[name for name in CATEGORICAL if name in self.feature_columns])
        if self.task:
            self.task.get_logger().report_single_value("training_seconds", time.monotonic() - started)
            self.task.get_logger().report_single_value("feature_count", len(self.feature_columns))
        return self

    def score(self, replay, user, items, timestamp, manual=None):
        user_embedding, item_embeddings = replay.embeddings(user, items, timestamp)
        features = embedding_frame(user_embedding, item_embeddings)
        if self.manual_features:
            features = pd.concat([manual.reset_index(drop=True), features], axis=1)
        return self.catboost.predict_proba(features[self.feature_columns])[:, 1]

    def predict(self, split, holdout=0):
        if not self.manual_features:
            for replay, query, batch, first, last, count in self.candidate_batches(split, holdout):
                scores = None if batch is None else self.score(replay, query.user_id, batch, query.timestamp)
                yield query.query_id, batch, scores, first, last, count
            return
        replay, _ = self.replay(split)
        queries = pd.read_csv(self.folder / f"holdout_{split}_{holdout}.csv").set_index("query_id")
        for query_id, frame, first, last, count in full_catalog_batches(self.folder, split, holdout, self.item_batch_size):
            query = queries.loc[query_id]
            if query.item_id not in self.catalog_set:
                if first:
                    yield query_id, None, None, True, True, 0
                continue
            replay.advance(query.timestamp)
            items = frame.item_key.to_numpy(dtype=np.int64)
            yield query_id, items, self.score(replay, int(query.user_id), items, int(query.timestamp), frame), first, last, count

    def run(self, dataset, log_clearml=True, run_test=True, force=False):
        if log_clearml:
            from clearml import Task
            if not (Path.home() / "clearml.conf").exists() and not (Path.home() / ".clearml.conf").exists() and not os.getenv("CLEARML_API_ACCESS_KEY") and not os.getenv("CLEARML_CONFIG_FILE"):
                os.environ["CLEARML_CACHE_DIR"] = str(self.folder / "clearml_cache")
                Task.set_offline(True)
            self.task = Task.init(project_name="CatBoost RecSys Baselines", task_name=f"{dataset} {self.__class__.__name__} manual={self.manual_features} seed {self.seed}", reuse_last_task_id=False)
            self.task.connect({"dataset": dataset, "model": self.__class__.__name__, "params": self.params, "catboost_params": self.catboost_params, "manual_features": self.manual_features, "catalog_items": len(self.catalog), "folds": self.folds}, name="General", ignore_remote_overrides=True)
        self.fit(force)
        results = self.results(run_test)
        self.finish(results)
        return results

class CatBoostTGNEmbeddingsBaseline(CatBoostTGNBaseline):
    def __init__(self, folder, **kwargs):
        super().__init__(folder, manual_features=False, **kwargs)
