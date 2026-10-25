import json
import os
import time
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.models.ranking_metrics import TOP_K, TopKItems, coverage_values

WINDOWS = [1, 3, 7, 14, 30]
CATEGORICAL = ["user_key", "item_key", "category", "content_type", "user_age", "user_income", "user_sex"]
KEYS = ["query_id", "user_id", "item_id", "timestamp", "label"]


def count_window(values, timestamp, days):
    return len(values) - bisect_left(values, timestamp - days * 86400)


def age_seconds(values, timestamp):
    return timestamp - values[-1] if values else -1


def build_features(folder, split, holdout=0):
    folder = Path(folder)
    if split != "train":
        raise ValueError("Only training pairs are materialized; evaluation uses the full catalog stream")
    cache = folder / "train_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pairs = pd.read_parquet(folder / "train_pairs.parquet")
    events = pd.read_csv(folder / "train.csv")
    events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    pairs = pairs.sort_values(["timestamp", "query_id"], kind="stable").reset_index(drop=True)
    items = pd.read_csv(folder / "items.csv").set_index("item_id")
    users = pd.read_csv(folder / "users.csv").set_index("user_id")
    category = items["category"].fillna("unknown").astype(str).to_dict()
    content_type = items["content_type"].fillna("unknown").astype(str).to_dict()
    year = items["release_year"].fillna(0).to_dict()
    user_age = users["user_age"].fillna("unknown").astype(str).to_dict() if "user_age" in users else {}
    user_income = users["user_income"].fillna("unknown").astype(str).to_dict() if "user_income" in users else {}
    user_sex = users["user_sex"].fillna("unknown").astype(str).to_dict() if "user_sex" in users else {}
    user_kids = users["user_kids"].fillna(0).to_dict() if "user_kids" in users else {}
    histories = {name: defaultdict(list) for name in ["user", "item", "pair", "category", "user_category"]}
    event_user = events.user_id.to_numpy()
    event_item = events.item_id.to_numpy()
    event_time = events.timestamp.to_numpy()
    rows = []
    cursor = 0
    for pair in pairs.itertuples(index=False):
        while cursor < len(events) and event_time[cursor] < pair.timestamp:
            user = int(event_user[cursor])
            item = int(event_item[cursor])
            timestamp = int(event_time[cursor])
            cat = category.get(item, "unknown")
            histories["user"][user].append(timestamp)
            histories["item"][item].append(timestamp)
            histories["pair"][(user, item)].append(timestamp)
            histories["category"][cat].append(timestamp)
            histories["user_category"][(user, cat)].append(timestamp)
            cursor += 1
        cat = category.get(pair.item_id, "unknown")
        values = {
            "user": histories["user"][pair.user_id],
            "item": histories["item"][pair.item_id],
            "pair": histories["pair"][(pair.user_id, pair.item_id)],
            "category": histories["category"][cat],
            "user_category": histories["user_category"][(pair.user_id, cat)],
        }
        feature = {
            "user_key": str(pair.user_id),
            "item_key": str(pair.item_id),
            "category": cat,
            "content_type": content_type.get(pair.item_id, "unknown"),
            "release_year": float(year.get(pair.item_id, 0)),
            "user_age": user_age.get(pair.user_id, "unknown"),
            "user_income": user_income.get(pair.user_id, "unknown"),
            "user_sex": user_sex.get(pair.user_id, "unknown"),
            "user_kids": float(user_kids.get(pair.user_id, 0)),
            "time_days": float(pair.timestamp - event_time[0]) / 86400,
        }
        for name, timestamps in values.items():
            feature[f"{name}_count"] = len(timestamps)
            feature[f"{name}_age"] = age_seconds(timestamps, pair.timestamp)
            for window in WINDOWS:
                feature[f"{name}_{window}d"] = count_window(timestamps, pair.timestamp, window)
        feature["category_affinity"] = len(values["user_category"]) / (len(values["user"]) + 1)
        feature["item_category_share"] = len(values["item"]) / (len(values["category"]) + 1)
        rows.append(feature)
    result = pd.concat([pairs[KEYS], pd.DataFrame(rows)], axis=1)
    result.to_parquet(cache, index=False)
    return result


def full_catalog_batches(folder, split, holdout, batch_size=4096):
    folder = Path(folder)
    queries = pd.read_csv(folder / f"holdout_{split}_{holdout}.csv").sort_values("timestamp", kind="stable")
    files = ["train.csv", "validation.csv"] + (["test.csv"] if split == "test" else [])
    events = pd.concat([pd.read_csv(folder / file) for file in files], ignore_index=True)
    events = events.sort_values("timestamp", kind="stable").reset_index(drop=True)
    catalog = np.sort(pd.read_csv(folder / "train.csv", usecols=["item_id"]).item_id.unique())
    items = pd.read_csv(folder / "items.csv").set_index("item_id")
    users = pd.read_csv(folder / "users.csv").set_index("user_id")
    category = items["category"].fillna("unknown").astype(str).to_dict()
    content_type = items["content_type"].fillna("unknown").astype(str).to_dict()
    year = items["release_year"].fillna(0).to_dict()
    user_age = users["user_age"].fillna("unknown").astype(str).to_dict() if "user_age" in users else {}
    user_income = users["user_income"].fillna("unknown").astype(str).to_dict() if "user_income" in users else {}
    user_sex = users["user_sex"].fillna("unknown").astype(str).to_dict() if "user_sex" in users else {}
    user_kids = users["user_kids"].fillna(0).to_dict() if "user_kids" in users else {}
    histories = {name: defaultdict(list) for name in ["user", "item", "pair", "category", "user_category"]}
    seen = defaultdict(set)
    event_user = events.user_id.to_numpy()
    event_item = events.item_id.to_numpy()
    event_time = events.timestamp.to_numpy()
    cursor = 0
    for query in queries.itertuples(index=False):
        while cursor < len(events) and event_time[cursor] < query.timestamp:
            user = int(event_user[cursor])
            item = int(event_item[cursor])
            timestamp = int(event_time[cursor])
            cat = category.get(item, "unknown")
            histories["user"][user].append(timestamp)
            histories["item"][item].append(timestamp)
            histories["pair"][(user, item)].append(timestamp)
            histories["category"][cat].append(timestamp)
            histories["user_category"][(user, cat)].append(timestamp)
            seen[user].add(item)
            cursor += 1
        available = np.concatenate(([query.item_id], catalog[(catalog != query.item_id) & ~np.isin(catalog, list(seen[query.user_id]))]))
        user_times = histories["user"][query.user_id]
        rows = []
        for position, item in enumerate(available):
            item = int(item)
            cat = category.get(item, "unknown")
            values = {
                "user": user_times,
                "item": histories["item"].get(item, ()),
                "pair": histories["pair"].get((query.user_id, item), ()),
                "category": histories["category"].get(cat, ()),
                "user_category": histories["user_category"].get((query.user_id, cat), ()),
            }
            feature = {
                "user_key": str(query.user_id),
                "item_key": str(item),
                "category": cat,
                "content_type": content_type.get(item, "unknown"),
                "release_year": float(year.get(item, 0)),
                "user_age": user_age.get(query.user_id, "unknown"),
                "user_income": user_income.get(query.user_id, "unknown"),
                "user_sex": user_sex.get(query.user_id, "unknown"),
                "user_kids": float(user_kids.get(query.user_id, 0)),
                "time_days": float(query.timestamp - event_time[0]) / 86400,
            }
            for name, timestamps in values.items():
                feature[f"{name}_count"] = len(timestamps)
                feature[f"{name}_age"] = age_seconds(timestamps, query.timestamp)
                for window in WINDOWS:
                    feature[f"{name}_{window}d"] = count_window(timestamps, query.timestamp, window)
            feature["category_affinity"] = len(values["user_category"]) / (len(user_times) + 1)
            feature["item_category_share"] = len(values["item"]) / (len(values["category"]) + 1)
            rows.append(feature)
            if len(rows) == batch_size or position == len(available) - 1:
                yield query.query_id, pd.DataFrame(rows), position + 1 == len(rows), position == len(available) - 1, len(available)
                rows = []


class CatBoostBaseline:
    def __init__(self, folder, seed=42, iterations=600, depth=7, learning_rate=0.05):
        self.folder = Path(folder)
        self.seed = seed
        self.params = {"iterations": iterations, "depth": depth, "learning_rate": learning_rate, "loss_function": "Logloss", "random_seed": seed, "task_type": "CPU", "thread_count": 2, "verbose": False, "allow_writing_files": False}
        self.model = None
        self.feature_columns = None
        self.training_seconds = 0

    def prepare(self, train, other=None):
        columns = [column for column in train.columns if column not in KEYS]
        return train[columns], other[columns] if other is not None else None

    def fit(self):
        train = build_features(self.folder, "train")
        self.feature_columns = [column for column in train.columns if column not in KEYS]
        features, _ = self.prepare(train)
        self.model = CatBoostClassifier(**self.params)
        started = time.monotonic()
        self.model.fit(features, train.label, cat_features=[name for name in CATEGORICAL if name in features])
        self.training_seconds = time.monotonic() - started
        return self

    def predict(self, split, holdout=0):
        for query_id, frame, first, last, count in full_catalog_batches(self.folder, split, holdout):
            features, _ = self.prepare(frame)
            yield query_id, frame.item_key.to_numpy(dtype=np.int64), self.model.predict_proba(features)[:, 1], first, last, count

    def evaluate(self, split, holdout=0):
        ranks = []
        candidate_counts = []
        recommended = {k: set() for k in TOP_K}
        positive_score = None
        higher = 0
        for query_id, items, scores, first, last, count in self.predict(split, holdout):
            if first:
                positive_score = scores[0]
                higher = int(np.count_nonzero(scores[1:] > positive_score))
                top_items = TopKItems()
            else:
                higher += int(np.count_nonzero(scores > positive_score))
            top_items.add(items, scores)
            if last:
                ranks.append(higher + 1)
                candidate_counts.append(count)
                for k in TOP_K:
                    recommended[k].update(top_items.items[:k])
        ranks = np.asarray(ranks)
        metrics = {"queries": len(ranks), "candidates_min": int(min(candidate_counts)), "candidates_max": int(max(candidate_counts))}
        for k in TOP_K:
            metrics[f"recall@{k}"] = float(np.mean(ranks <= k))
            metrics[f"ndcg@{k}"] = float(np.mean(np.where(ranks <= k, 1 / np.log2(ranks + 1), 0)))
        catalog_size = json.loads((self.folder / "info.json").read_text())["train_catalog_items"]
        metrics.update(coverage_values(recommended, catalog_size))
        return metrics

    def run(self, dataset, log_clearml=True, run_test=True):
        info = json.loads((self.folder / "info.json").read_text())
        task = None
        if log_clearml:
            from clearml import Task

            if not (Path.home() / "clearml.conf").exists() and not (Path.home() / ".clearml.conf").exists() and not os.getenv("CLEARML_API_ACCESS_KEY") and not os.getenv("CLEARML_CONFIG_FILE"):
                os.environ["CLEARML_CACHE_DIR"] = str(self.folder / "clearml_cache")
                Task.set_offline(True)
            task = Task.init(project_name="CatBoost RecSys Baselines", task_name=f"{dataset} {self.__class__.__name__} GTS Last/Random seed {self.seed}", reuse_last_task_id=False)
            task.connect({"dataset": dataset, "model": self.__class__.__name__, "seed": self.seed, "parameters": self.params, "search_parameters": getattr(self, "search_params", {}), "holdout_protocols": info["holdout_protocols"], "holdout_counts": info["holdout_counts"], "test_holdout_seeds": info["test_holdout_seeds"], "evaluation_catalog": info["evaluation_catalog"], "evaluation_catalog_scope": info["evaluation_catalog_scope"], "train_catalog_items": info["train_catalog_items"], "selected_users_observed_items": info["selected_users_observed_items"], "original_metadata_items": info["original_metadata_items"], "openfe_rows": getattr(self, "openfe_rows", 0), "openfe_features": getattr(self, "openfe_features", 0)}, name="General", ignore_remote_overrides=True)
        self.fit()
        if task:
            if hasattr(self, "selected_crosses"):
                task.connect({"search_fields": self.search_fields, "selected_crosses": [list(fields) for fields in self.selected_crosses]}, name="AutoCross", ignore_remote_overrides=True)
            task.get_logger().report_single_value("training_seconds", self.training_seconds)
            task.get_logger().report_single_value("feature_count", len(self.feature_columns))
        results = {}
        for split in ["validation", "test"] if run_test else ["validation"]:
            holdout_count = info["holdout_counts"][split]
            values = [self.evaluate(split, index) for index in range(holdout_count)]
            summary = {}
            for key in values[0]:
                measurements = [row[key] for row in values]
                summary[f"{key}_mean"] = float(np.mean(measurements))
                summary[f"{key}_std"] = float(np.std(measurements, ddof=1)) if holdout_count > 1 else None
            results[split] = {"protocol": info["holdout_protocols"][split], "catalog": info["evaluation_catalog"], "catalog_scope": info["evaluation_catalog_scope"], "train_catalog_items": info["train_catalog_items"], "holdouts": values, "summary": summary}
            if task:
                for index, metrics in enumerate(values):
                    for key, value in metrics.items():
                        task.get_logger().report_scalar(title=f"{split}/{key}", series="holdout", value=value, iteration=index)
                for key, value in summary.items():
                    if value is not None:
                        task.get_logger().report_single_value(f"{split}/{key}", value)
        if task:
            task.close()
        output = self.folder / f"{self.__class__.__name__}_seed_{self.seed}_full_catalog_metrics.json"
        output.write_text(json.dumps(results, indent=2))
        return results


class CatBoostOpenFEBaseline(CatBoostBaseline):
    def __init__(self, folder, seed=42, iterations=600, depth=7, learning_rate=0.05, openfe_rows=20000, openfe_features=20):
        super().__init__(folder, seed, iterations, depth, learning_rate)
        self.openfe_rows = openfe_rows
        self.openfe_features = openfe_features
        self.generated = None
        self.numeric_columns = None
        self.openfe = None

    def prepare(self, train, other=None):
        columns = [column for column in train.columns if column not in KEYS]
        base = train[columns].reset_index(drop=True)
        if other is None:
            augmented, _ = self.openfe.transform(base[self.numeric_columns], base.iloc[:1][self.numeric_columns], self.generated, n_jobs=1)
        else:
            target = other[columns].reset_index(drop=True)
            augmented, target_augmented = self.openfe.transform(base[self.numeric_columns], target[self.numeric_columns], self.generated, n_jobs=1)
        for column in augmented.columns:
            if column not in base:
                base[column] = pd.to_numeric(augmented[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if other is None:
            return base, None
        for column in target_augmented.columns:
            if column not in target:
                target[column] = pd.to_numeric(target_augmented[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return base, target[base.columns]

    def fit(self):
        import multiprocessing
        from openfe import get_candidate_features

        from .openfe_safe import FiniteCorrelationOpenFE

        multiprocessing.set_start_method("fork", force=True)
        train = build_features(self.folder, "train")
        discovery = ["user_count", "item_count", "pair_count", "category_count", "user_category_count", "user_age", "item_age", "user_7d", "item_7d", "pair_30d", "user_category_30d", "category_affinity", "item_category_share"]
        self.numeric_columns = [column for column in discovery if column in train and column not in CATEGORICAL and train[column].nunique() > 1]
        per_class = min(self.openfe_rows // 2, int(train.label.value_counts().min()))
        chosen = train.groupby("label", group_keys=False).sample(n=per_class, random_state=self.seed).sort_index().reset_index(drop=True)
        started = time.monotonic()
        candidates = [feature for feature in get_candidate_features(numerical_features=self.numeric_columns, order=1) if feature.name != "freq"]
        self.openfe = FiniteCorrelationOpenFE()
        generated = self.openfe.fit(data=chosen[self.numeric_columns].astype("float32"), label=chosen[["label"]], task="classification", candidate_features_list=candidates, n_jobs=1, seed=self.seed, min_candidate_features=100, n_data_blocks=4, stage1_metric="corr", stage2_params={"n_estimators": 200, "importance_type": "gain", "num_leaves": 16, "seed": self.seed, "n_jobs": 1, "verbosity": -1}, verbose=False, tmp_save_path=str(self.folder / "openfe_tmp_data.feather"))
        self.generated = generated[:self.openfe_features]
        self.feature_columns = [column for column in train.columns if column not in KEYS] + [str(feature) for feature in self.generated]
        features, _ = self.prepare(train)
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(features, train.label, cat_features=[name for name in CATEGORICAL if name in features])
        self.training_seconds = time.monotonic() - started
        self.feature_columns = list(features.columns)
        return self
