from torch import nn
from .toolkit import get_task_adapter


class BaseMILModel(nn.Module):
    def __init__(self, task) -> None:
        super().__init__()
        self.task_adapter = get_task_adapter(task)

