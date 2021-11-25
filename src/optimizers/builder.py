from torch.optim import *
import torch.nn as nn

available_names = [
    'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 'ASGD', 'LBFGS', 'RMSprop', 'Rprop', 'SGD'
]


def create_optimizer(optimizer_name: str, model: nn.Module, kwargs: dict) -> Optimizer:
    '''
    Create Optimizer object from name
    :param optimizer_name: name of optimizer
    :param model: model
    :param kwargs: dict of params of optimizer
    :return: Optimizer object
    '''
    if optimizer_name in available_names:
        return eval(optimizer_name)(params=filter(lambda p: p.requires_grad, model.parameters()), **kwargs)
    else:
        raise NotImplementedError('{} not implemented'.format(optimizer_name))
