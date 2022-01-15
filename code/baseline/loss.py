
import torch
import torch.nn as nn
import torch.nn.functional as F


_criterion_entropoints = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss()
}

def criterion_entrypoint(criterion_name):
    return _criterion_entropoints[criterion_name]

def is_criterion(criterion_name):
    return criterion_name in _criterion_entropoints

def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn
        # criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion