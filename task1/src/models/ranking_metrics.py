"""Helpers for streaming top-k recommendation metrics."""

import numpy as np


TOP_K = (5, 10, 20)


class TopKItems:
    def __init__(self, k=max(TOP_K)):
        self.k = k
        self.items = np.empty(0, dtype=np.int64)
        self.scores = np.empty(0, dtype=np.float64)
        self.positions = np.empty(0, dtype=np.int64)
        self.next_position = 0

    def add(self, items, scores):
        items = np.asarray(items, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float64)
        if len(items) != len(scores):
            raise ValueError("Item IDs and scores must have the same length")
        positions = np.arange(self.next_position, self.next_position + len(items))
        self.next_position += len(items)
        all_items = np.concatenate((self.items, items))
        all_scores = np.concatenate((self.scores, np.nan_to_num(scores, nan=-np.inf)))
        all_positions = np.concatenate((self.positions, positions))
        best = np.lexsort((all_positions, -all_scores))[: self.k]
        self.items = all_items[best]
        self.scores = all_scores[best]
        self.positions = all_positions[best]


def coverage_values(recommended, catalog_size):
    if catalog_size <= 0:
        raise ValueError("Catalog size must be positive")
    return {f"coverage@{k}": len(recommended[k]) / catalog_size for k in TOP_K}
