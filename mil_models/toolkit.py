"""
Time: 2023.12.12
@author: MA JIABO
"""

import torch
import torch.nn.functional as F


def get_task_adapter(task):
    if task == 'survival':
        return survial_task_logits_adapter
    elif task == 'subtyping':
        return subtyping_task_logits_adapter
    else:
        raise NotImplementedError(f'{task} is not implemented yet, please implement it in mil_models/toolkit.py ...')


def survial_task_logits_adapter(logits):
    """Convert the logits to survial task to calculate loss

    Args:
        logits (_type_): _description_

    Returns:
        _type_: _description_
    """
    Y_hat = torch.topk(logits, 1, dim = 1)[1]
    hazards = torch.sigmoid(logits)
    S = torch.cumprod(1 - hazards, dim=1)
    return hazards, S, Y_hat


def subtyping_task_logits_adapter(logits):
    """Subtyping task adapter

    Args:
        logits (_type_): _description_

    Returns:
        _type_: _description_
    """
    Y_prob = F.softmax(logits, dim=1)
    Y_hat = torch.topk(logits, 1, dim=1)[1]
    return logits, Y_prob, Y_hat

    