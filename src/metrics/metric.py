from torchmetrics import Metric
from torch.nn import ModuleList
import torch


class MetricsList(Metric):
    def __init__(self, dist_sync_on_step=False, compute_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=compute_on_step)
        self.metrics = ModuleList()

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for metric_idx in range(len(self.metrics)):
            # TODO: optimize metrics device passing
            self.metrics[metric_idx].to(preds.device)
            self.metrics[metric_idx].update(preds, target)

    def compute(self):
        result = {}
        for metric_idx in range(len(self.metrics)):
            result_ = self.metrics[metric_idx].compute()
            result.update(result_)
        return result

    def reset(self):
        for metric_idx in range(len(self.metrics)):
            self.metrics[metric_idx].reset()

    def add(self, metric: Metric):
        self.metrics.append(module=metric)
