from pathlib import Path

import torch
import torch.nn.functional as F


class BaseModel:
    def __init__(self, name):
        self.name = name

    def save_checkpoint(self, path):
        save_weights(self.model, path)

    def load_checkpoint(self, path):
        load_weights(self.model, path)


class TGNBPR(torch.nn.Module):
    def forward(self, positive_scores, negative_scores):
        return -F.logsigmoid(positive_scores.unsqueeze(1) - negative_scores).mean()


class TGNCSE(torch.nn.Module):
    def forward(self, positive_scores, negative_scores):
        logits = torch.cat((positive_scores.unsqueeze(1), negative_scores), dim=1)
        return F.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=logits.device))


class TGNFCE(torch.nn.Module):
    def forward(self, catalog_scores, positive_item_ids):
        return F.cross_entropy(catalog_scores, positive_item_ids.long())


def save_weights(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_weights(model, path):
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))


class TrainingRun:
    def __init__(self):
        self.task = None
        self.phase = "tgn/full"

    def define_metric(self, name, step_metric=None):
        pass

    def log(self, values):
        if self.task is not None:
            iteration = int(values.get("epoch", 0))
            for name, value in values.items():
                if name != "epoch" and isinstance(value, (int, float)):
                    self.task.get_logger().report_scalar(title=f"{self.phase}/{name}", series="epoch", value=float(value), iteration=iteration)

    def finish(self):
        pass


_training_run = TrainingRun()


def set_training_task(task, phase="tgn/full"):
    _training_run.task = task
    _training_run.phase = phase


def current_run():
    return _training_run


def init_run(project, name, config=None):
    return _training_run
