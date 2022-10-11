from torch.utils.data import DataLoader
from typing import Union, List


class DataModuleBase(object):
    def __init__(self, batch_size, num_workers) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_loader(self) -> DataLoader:
        raise NotImplementedError

    def valid_loader(self) -> DataLoader:
        raise NotImplementedError

    def test_loader(self) -> DataLoader:
        raise NotImplementedError

    @classmethod
    def atom_attr_sizes(cls) -> Union[List, None]:
        raise NotImplementedError

    @classmethod
    def bond_attr_sizes(cls) -> Union[List, None]:
        raise NotImplementedError

    @property
    def num_tasks(self) -> int:
        raise NotImplementedError

    @property
    def eval_metric(self) -> str:
        return NotImplementedError

    def evaluate(self, preds, targets) -> float:
        raise NotImplementedError
