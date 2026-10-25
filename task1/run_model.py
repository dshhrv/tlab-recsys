import json
import inspect
from pathlib import Path
import re

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from scipy import stats
import torch

from src.metrics import (
    Summarizer,
)
from src.utils import (
    create_logger,
    fix_random_seed,
    save_metrics,
)

logger = create_logger(name=__name__)


@hydra.main(version_base=None, 
            config_path="configs", 
            config_name="default")
def main(cfg):
    if cfg.get("test_only", False):
        test_metrics = run_train(cfg, test_mode=True)
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        save_metrics(test_metrics, output_dir)
        return

    val_metrics = run_train(cfg, test_mode=False)

    if "additional_cfg_params" in val_metrics:
        for key, value in val_metrics["additional_cfg_params"].items():
            cfg.model[key] = value

    test_metrics = run_train(cfg, test_mode=True)
    metrics = {**val_metrics, **test_metrics}

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    save_metrics(metrics, output_dir)


def run_train(cfg, verbose: bool = True, test_mode: bool = False):
    fix_random_seed(cfg.get("seed", 42))
    num_seeds = 10

    config = OmegaConf.to_container(cfg, resolve=True)
    if verbose:
        logger.info(f"Training config:\n{OmegaConf.to_yaml(config)}\n")

    model = instantiate(cfg.model)
    if verbose:
        logger.info(f"Model init: {str(model)}\n")

    train_dataset = instantiate(cfg.dataset, 
                                split="train", 
                                merge_train_val=test_mode
                                )
    if not test_mode:
        eval_dataset = instantiate(
            cfg.dataset,
            split="val",
            merge_train_val=test_mode,
            holdout_filename="holdout_validation.csv",
        )
    else:
        eval_datasets = [
            instantiate(
                cfg.dataset,
                split="test",
                merge_train_val=test_mode,
                holdout_filename=f"holdout_test_{holdout_id}.csv",
            )
            for holdout_id in range(num_seeds)
        ]

    metrics = Summarizer([
        instantiate(metric_cfg, n_items=train_dataset.n_items) for metric_cfg in cfg.metrics
    ])
    
    if verbose:
        logger.info(f"Start training...\n")

    if not test_mode:
        fit_parameters = inspect.signature(model.fit).parameters
        fit_kwargs = {}
        if "metrics" in fit_parameters:
            fit_kwargs["metrics"] = metrics
        if "max_top_n" in fit_parameters:
            fit_kwargs["max_top_n"] = cfg.max_top_n
        checkpoint_path = None
        if "checkpoint_path" in fit_parameters and "resume_epoch" in fit_parameters:
            checkpoint_name = re.sub(r"\s+", "", str(cfg.get("checkpoint_name", "weights")))
            checkpoint_path = Path(HydraConfig.get().runtime.output_dir) / f"{checkpoint_name}.pt"
            resume_epoch = max(0, int(cfg.get("num", 0)))
            fit_kwargs["checkpoint_path"] = str(checkpoint_path)
            fit_kwargs["resume_epoch"] = resume_epoch
        model.fit(train_dataset, eval_dataset, **fit_kwargs)
    else:
        fit_parameters = inspect.signature(model.fit).parameters
        fit_kwargs = {}
        if "checkpoint_path" in fit_parameters and "resume_epoch" in fit_parameters:
            load_path = cfg.get("load_checkpoint_path", None)
            if load_path:
                fit_kwargs["checkpoint_path"] = str(load_path)
                fit_kwargs["resume_epoch"] = 0
        model.fit(train_dataset, None, **fit_kwargs)

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    fit_parameters = inspect.signature(model.fit).parameters
    if test_mode or "checkpoint_path" not in fit_parameters:
        torch.save(model.model.state_dict(), output_dir / f"{cfg.model.name}_weights.pt")
    
    if verbose:
        logger.info(f"Training completed.\n")

    if not test_mode:
        predictions = model.predict(eval_dataset, top_n=cfg.max_top_n)
        holdout_users = eval_dataset.get_holdout_users()
        eval_metrics = {
            f"{eval_dataset._split}/{k}": v
            for k, v in metrics(
                predictions[holdout_users, :],
                eval_dataset.get_holdout_array()[holdout_users],
            ).items()
        }
    else:
        if hasattr(model, "predict_many_test_holdouts"):
            predictions = model.predict_many_test_holdouts(eval_datasets, top_n=cfg.max_top_n)
        else:
            memory = getattr(getattr(model, "model", None), "memory", None)
            base_memory = memory.backup_memory() if memory is not None else None
            predictions = []
            for eval_dataset in eval_datasets:
                if base_memory is not None:
                    memory.restore_memory(base_memory)
                predictions.append(model.predict(eval_dataset, top_n=cfg.max_top_n))

        if len(eval_datasets) == 0:
            raise ValueError("No evaluation datasets provided for test mode.")

        holdout_metric_log = {}
        per_holdout_metrics = []
        for holdout_id, (eval_dataset, preds) in enumerate(zip(eval_datasets, predictions)):
            holdout_users = eval_dataset.get_holdout_users()
            metric_values = metrics(
                preds[holdout_users, :],
                eval_dataset.get_holdout_array()[holdout_users],
            )
            per_holdout_metrics.append(metric_values)
            holdout_metric_log[f"holdout_{holdout_id}"] = metric_values

        n_holdouts = len(per_holdout_metrics)
        aggregated_metrics = {}
        for metric_name in per_holdout_metrics[0].keys():
            values = np.array([metric_dict[metric_name] for metric_dict in per_holdout_metrics], dtype=float)
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1)) if n_holdouts > 1 else 0.0
            if n_holdouts > 1:
                ci_low, ci_high = stats.t.interval(
                    0.95,
                    df=n_holdouts - 1,
                    loc=mean_value,
                    scale=std_value / np.sqrt(n_holdouts),
                )
                ci_low = float(ci_low)
                ci_high = float(ci_high)
            else:
                ci_low = float(mean_value)
                ci_high = float(mean_value)

            aggregated_metrics[f"{metric_name}"] = mean_value
            aggregated_metrics[f"{metric_name}_std"] = std_value
            aggregated_metrics[f"{metric_name}_ci_low"] = ci_low
            aggregated_metrics[f"{metric_name}_ci_high"] = ci_high

        eval_metrics = {f"test/{k}": v for k, v in aggregated_metrics.items()}
        if verbose:
            logger.info(f"Holdout metrics: {json.dumps(holdout_metric_log, indent=2)}\n")

    additional_cfg_params = {}
    if not test_mode and hasattr(model, "suggest_additional_params"):
        suggested = model.suggest_additional_params()
        if isinstance(suggested, dict):
            additional_cfg_params = suggested

    if additional_cfg_params:
        eval_metrics["additional_cfg_params"] = additional_cfg_params

    if verbose:
        logger.info(f"Final metrics: {json.dumps(eval_metrics, indent=2)}\n")

    return eval_metrics

if __name__ == "__main__":
    main()
