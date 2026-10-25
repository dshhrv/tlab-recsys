import optuna
import numpy as np
from pathlib import Path
import os
import torch

from abc import abstractmethod


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        temporary_path,
    )
    os.replace(temporary_path, path)


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu"))


class BaseModel:
    """
    Base class for all models.
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

    def __str__(self):
        return self.name

    @abstractmethod
    def fit(self, train_dataset, val_dataset):
        """
        Fit the model to the dataset.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def predict(self, dataset, top_n: int) -> np.ndarray:
        """
        Make predictions on the given data.

        Inputs:
            dataset: The dataset to make predictions on.
            top_n (int): The number of top items to recommend.
        Returns:
            np.ndarray: The predictions (shape: [n_users, top_n]).
            Note: here n_users is the number of users in the whole dataset,
            not only in the test split.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """
        Save the model checkpoint to the specified path.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def load_checkpoint(self, path: str):
        """
        Load the model checkpoint from the specified path.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def suggest_additional_params(self) -> dict:
        """
        Suggest additional hyperparameters for the model after train with validation.
        Returns:
            dict: A dictionary of additional hyperparameters.
        """
        return {}


class BaseMetric:
    """
    Base class for all metrics.
    """
    def __init__(self, name: str, *args, **kwargs):
        self.name = name

    def __str__(self):
        return self.name

    @abstractmethod
    def __call__(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the metric given predictions and targets.

        Inputs:
            predictions (np.ndarray): The model predictions.
            targets (np.ndarray): The ground truth targets
        Returns:
            The computed metric value.
        """
        raise NotImplementedError("Subclasses should implement this method.")
