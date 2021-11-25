from torch.optim.lr_scheduler import *
from torch.optim import Optimizer

available_names = [
    'LambdaLR', 'MultiplicativeLR', 'StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
    'CyclicLR', 'OneCycleLR', 'CosineAnnealingWarmRestarts'
]


def create_lr_scheduler(lr_scheduler_name: str, optimizer: Optimizer, kwargs: dict):
    '''
    Return lr_scheduler object
    :param lr_scheduler_name: name of scheduler
    :param optimizer: torch.optim.Optimizer object
    :param kwargs: dict of params for lr_scheduler
    :return: lr_scheduler
    '''
    if lr_scheduler_name in available_names:
        return eval(lr_scheduler_name)(optimizer=optimizer, **kwargs)
    else:
        raise NotImplementedError('{} not implemented'.format(lr_scheduler_name))
