import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.models.catboost_baseline import CATEGORICAL, CatBoostBaseline, build_features


def logistic(values):
    return 1 / (1 + np.exp(-np.clip(values, -20, 20)))


def log_loss(values, labels):
    probabilities = logistic(values)
    return float(-np.mean(labels * np.log(probabilities + 1e-8) + (1 - labels) * np.log(1 - probabilities + 1e-8)))


def cross_hash(frame, fields):
    return pd.util.hash_pandas_object(frame[list(fields)].astype(str), index=False).to_numpy(dtype="uint64")


def fit_field(codes, offsets, labels, weights, start, stop, batch_size, learning_rate):
    bins = len(weights)
    for left in range(start, stop, batch_size):
        right = min(left + batch_size, stop)
        keys = codes[left:right]
        errors = logistic(offsets[left:right] + weights[keys]) - labels[left:right]
        counts = np.bincount(keys, minlength=bins)
        gradients = np.bincount(keys, weights=errors, minlength=bins) / (counts + 10)
        weights -= learning_rate * gradients.astype("float32")


def fit_original_fields(train, validation, labels, fields, bins, batch_size, learning_rate):
    train_codes = np.stack([(cross_hash(train, (field,)) % bins).astype("int32") for field in fields])
    validation_codes = np.stack([(cross_hash(validation, (field,)) % bins).astype("int32") for field in fields])
    weights = np.zeros((len(fields), bins), dtype="float32")
    prevalence = np.clip(labels.mean(), 1e-4, 1 - 1e-4)
    bias = float(np.log(prevalence / (1 - prevalence)))
    for epoch in range(3):
        for left in range(0, len(train), batch_size):
            right = min(left + batch_size, len(train))
            scores = np.full(right - left, bias, dtype="float32")
            for index in range(len(fields)):
                scores += weights[index, train_codes[index, left:right]]
            errors = logistic(scores) - labels[left:right]
            for index in range(len(fields)):
                keys = train_codes[index, left:right]
                counts = np.bincount(keys, minlength=bins)
                gradients = np.bincount(keys, weights=errors, minlength=bins) / (counts + 10)
                weights[index] -= learning_rate * gradients.astype("float32")
    train_offsets = np.full(len(train), bias, dtype="float32")
    validation_offsets = np.full(len(validation), bias, dtype="float32")
    for index in range(len(fields)):
        train_offsets += weights[index, train_codes[index]]
        validation_offsets += weights[index, validation_codes[index]]
    return train_offsets, validation_offsets


def search_crosses(frame, max_crosses, max_order, search_rows, bins, batch_size, learning_rate):
    fields = [field for field in CATEGORICAL if field in frame and frame[field].nunique() > 1]
    if len(fields) < 2 or len(frame) < 10:
        return [], fields
    dependencies = set()
    for left in fields:
        for right in fields:
            if left != right and frame.groupby(left, sort=False)[right].nunique(dropna=False).max() == 1:
                dependencies.add((left, right))
    if len(frame) > search_rows:
        frame = frame.iloc[np.linspace(0, len(frame) - 1, search_rows, dtype=int)].reset_index(drop=True)
    boundary = max(2, min(len(frame) - 2, int(len(frame) * 0.8)))
    train = frame.iloc[:boundary]
    validation = frame.iloc[boundary:]
    labels = train.label.to_numpy(dtype="float32")
    validation_labels = validation.label.to_numpy(dtype="float32")
    train_offsets, validation_offsets = fit_original_fields(train, validation, labels, fields, bins, batch_size, learning_rate)
    selected = []
    current_loss = log_loss(validation_offsets, validation_labels)
    for iteration in range(max_crosses):
        active = [(field,) for field in fields] + selected
        children = set()
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                combined = tuple(field for field in fields if field in left or field in right)
                redundant = any((left, right) in dependencies for left in combined for right in combined if left != right)
                if 2 <= len(combined) <= max_order and combined not in selected and not redundant:
                    children.add(combined)
        if not children:
            break
        codes = {}
        for child in sorted(children):
            codes[child] = ((cross_hash(train, child) % bins).astype("int32"), (cross_hash(validation, child) % bins).astype("int32"))
        weights = {child: np.zeros(bins, dtype="float32") for child in children}
        survivors = sorted(children)
        trained = 0
        budget = min(len(train), max(256, batch_size))
        while len(survivors) > 1:
            losses = []
            for child in survivors:
                train_codes, validation_codes = codes[child]
                fit_field(train_codes, train_offsets, labels, weights[child], trained, budget, batch_size, learning_rate)
                losses.append((log_loss(validation_offsets + weights[child][validation_codes], validation_labels), child))
            survivors = [child for _, child in sorted(losses)[:max(1, len(losses) // 2)]]
            trained = budget
            budget = min(len(train), budget * 2)
        best = survivors[0]
        train_codes, validation_codes = codes[best]
        fit_field(train_codes, train_offsets, labels, weights[best], trained, len(train), batch_size, learning_rate)
        new_loss = log_loss(validation_offsets + weights[best][validation_codes], validation_labels)
        if new_loss >= current_loss:
            break
        train_offsets += weights[best][train_codes]
        validation_offsets += weights[best][validation_codes]
        current_loss = new_loss
        selected.append(best)
    return selected, fields


class CatBoostAutoCrossBaseline(CatBoostBaseline):
    def __init__(self, folder, seed=42, iterations=600, depth=7, learning_rate=0.05, max_crosses=6, max_order=3, search_rows=30000, search_bins=65536):
        super().__init__(folder, seed, iterations, depth, learning_rate)
        self.search_params = {"max_crosses": max_crosses, "max_order": max_order, "search_rows": search_rows, "search_bins": search_bins}
        self.selected_crosses = []
        self.search_fields = []

    def prepare(self, train, other=None):
        base, target = super().prepare(train, other)
        base = base.copy()
        for fields in self.selected_crosses:
            base["cross_" + "__".join(fields)] = cross_hash(train, fields).astype(str)
        if target is None:
            return base, None
        target = target.copy()
        for fields in self.selected_crosses:
            target["cross_" + "__".join(fields)] = cross_hash(other, fields).astype(str)
        return base, target

    def fit(self):
        train = build_features(self.folder, "train")
        started = time.monotonic()
        self.selected_crosses, self.search_fields = search_crosses(train, self.search_params["max_crosses"], self.search_params["max_order"], self.search_params["search_rows"], self.search_params["search_bins"], 512, 0.5)
        features, _ = self.prepare(train)
        self.model = CatBoostClassifier(**self.params)
        categorical = [name for name in CATEGORICAL if name in features] + ["cross_" + "__".join(fields) for fields in self.selected_crosses]
        self.model.fit(features, train.label, cat_features=categorical)
        self.training_seconds = time.monotonic() - started
        self.feature_columns = list(features.columns)
        output = self.folder / f"{self.__class__.__name__}_seed_{self.seed}_crosses.json"
        output.write_text(json.dumps({"fields": self.search_fields, "selected_crosses": self.selected_crosses, "search_params": self.search_params}, indent=2))
        return self
