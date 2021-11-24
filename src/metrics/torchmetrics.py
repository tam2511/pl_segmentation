from torchmetrics import Metric
import torch

available_names = [
    'PixelLevelAccuracy', 'PixelLevelRecall', 'PixelLevelPrecision', 'PixelLevelFBeta', 'PixelLevelF1'
]


class PixelLevelBase(Metric):
    def __init__(self, average='macro', num_classes=0, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.average = average
        if num_classes is None or num_classes < 1:
            raise ValueError('You must passing number of classes')
        if average not in ['macro', 'none']:
            raise ValueError('Available average values: macro, none')
        self.add_state('value', default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state('num', default=torch.tensor(0.0), dist_reduce_fx="sum")

    def __stats(self, preds: torch.Tensor, targets: torch.Tensor):
        tp = (preds * targets).sum(dim=2)
        tn = (torch.logical_not(preds) * torch.logical_not(targets)).sum(dim=2)
        fp = (preds * torch.logical_not(targets)).sum(dim=2)
        fn = (torch.logical_not(preds) * targets).sum(dim=2)
        return tp, tn, fp, fn

    def update(self, pred_masks: torch.Tensor, target_masks: torch.Tensor):
        batch_size, num_classes = pred_masks.size(0), pred_masks.size(1)
        preds, targets = pred_masks.view(batch_size, num_classes, -1), target_masks.view(batch_size, num_classes, -1)
        preds = preds > self.threshold
        self.value += self.reduce(*self.__stats(preds, targets)).sum(0)
        self.num += batch_size

    def compute(self):
        value = self.value / self.num
        if self.average == 'macro':
            return value.mean()
        return value

    def reset(self):
        self.value *= 0
        self.num *= 0

    def reduce(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor):
        raise NotImplementedError('Method "reduce" must be overwrite')


class PixelLevelAccuracy(PixelLevelBase):
    def __init__(self, average='macro', threshold=0.5, num_classes=None):
        super().__init__(average=average, threshold=threshold, num_classes=num_classes)

    def reduce(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor):
        return (tp + tn) / (tp + tn + fp + fn)


class PixelLevelRecall(PixelLevelBase):
    def __init__(self, average='macro', threshold=0.5, num_classes=None, epsilon=1e-8):
        super().__init__(average=average, threshold=threshold, num_classes=num_classes)
        self.epsilon = epsilon

    def reduce(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor):
        return tp / (tp + fn + self.epsilon)


class PixelLevelPrecision(PixelLevelBase):
    def __init__(self, average='macro', threshold=0.5, num_classes=None, epsilon=1e-8):
        super().__init__(average=average, threshold=threshold, num_classes=num_classes)
        self.epsilon = epsilon

    def reduce(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor):
        return tp / (tp + fp + self.epsilon)


class PixelLevelFBeta(PixelLevelBase):
    def __init__(self, average='macro', threshold=0.5, num_classes=None, beta=1.0, epsilon=1e-8):
        super().__init__(average=average, threshold=threshold, num_classes=num_classes)
        self.beta = beta
        self.epsilon = epsilon

    def reduce(self, tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor):
        return (1 + self.beta ** 2) * tp / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.epsilon)


class PixelLevelF1(PixelLevelFBeta):
    def __init__(self, average='macro', threshold=0.5, num_classes=None):
        super().__init__(average=average, threshold=threshold, num_classes=num_classes, beta=1.0)


class TorchMetric(Metric):
    def __init__(self, metric_info, class_names=None):
        super().__init__()
        if metric_info['name'] not in available_names:
            raise NotImplementedError('{} is not implemented'.format(metric_info['name']))
        self.metric = eval(metric_info['name'])(**{_: metric_info[_] for _ in metric_info if _ != 'name'})
        self.class_names = class_names
        self.metric_info = metric_info

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.metric.update(preds, target)

    def compute(self) -> dict:
        result = self.metric.compute()
        if result.size() == torch.Size([len(self.class_names)]):
            return {'{}_{}'.format(self.metric_info['name'], self.class_names[idx]): result[idx] for idx in
                    range(len(self.class_names))}
        return {'{}_{}'.format(self.metric_info['name'], self.metric.average): result}

    def reset(self):
        self.metric.reset()
