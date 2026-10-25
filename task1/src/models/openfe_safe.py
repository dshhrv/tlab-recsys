"""OpenFE with a defined correlation score for constant candidate features."""

import numpy as np
import pandas as pd
from openfe import OpenFE


class FiniteCorrelationOpenFE(OpenFE):
    def _evaluate(self, candidate_feature, train_y, val_y, train_init, val_init, init_metric):
        if self.stage1_metric != "corr":
            return super()._evaluate(candidate_feature, train_y, val_y, train_init, val_init, init_metric)

        values = pd.concat(
            [candidate_feature.data.loc[train_y.index], candidate_feature.data.loc[val_y.index]]
        ).to_numpy(dtype=np.float64).ravel()
        labels = pd.concat([train_y, val_y]).to_numpy(dtype=np.float64).ravel()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        labels = np.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
        values -= values.mean()
        labels -= labels.mean()
        denominator = np.linalg.norm(values) * np.linalg.norm(labels)
        if denominator == 0 or not np.isfinite(denominator):
            return 0.0
        score = abs(float(np.dot(values, labels) / denominator))
        return score if np.isfinite(score) else 0.0
