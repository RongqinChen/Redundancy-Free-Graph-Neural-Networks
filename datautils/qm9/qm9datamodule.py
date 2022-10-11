import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from trainutils import DataModuleBase

from .qm9dataset import QM9Dataset
from ..transform import collate_fn_dict, transform_fn_dict


class QM9_DataModule(DataModuleBase):
    def __init__(self, batch_size: int, num_workers: int, target_idx: int,
                 save_transform, transform_type: str, transform_fn_kwargs: dict) -> None:
        super().__init__(batch_size, num_workers)
        assert save_transform in {'raw'}
        assert 0 <= target_idx < 19
        self.target_idx = target_idx
        self._pre_transform = transform_fn_kwargs['pre_transform']
        del transform_fn_kwargs['pre_transform']
        self._collate_fn = collate_fn_dict[transform_type]
        self._transform_fn = transform_fn_dict[transform_type]
        self._transform_fn_kwargs = transform_fn_kwargs
        self._qm9_dataset = QM9Dataset(self._do_transform, norm_fn=self._norm_fn)
        sample_idx_list = list(range(len(self._qm9_dataset)))
        random.shuffle(sample_idx_list)
        tenpercent = int(len(sample_idx_list)*0.1)
        self._train_idx_list = sample_idx_list[2*tenpercent:]
        self._compute_train_set_target_mean_and_std()
        self._valid_idx_list = sample_idx_list[tenpercent:2*tenpercent]
        self._test_idx_list = sample_idx_list[:tenpercent]
        self._valid_dataset = Subset(self._qm9_dataset, self._valid_idx_list)
        self._test_dataset = Subset(self._qm9_dataset, self._test_idx_list)

    def _compute_train_set_target_mean_and_std(self):
        y_list = [self._qm9_dataset._target[idx][self.target_idx].item()
                  for idx in self._train_idx_list]
        self._y_mean = np.mean(y_list)
        self._y_std = np.std(y_list)

    @property
    def train_set_y_mean(self,):
        return self._y_mean

    @property
    def train_set_y_std(self,):
        return self._y_std

    def __repr__(self) -> str:
        return "QM9"

    def train_loader(self):
        random.shuffle(self._train_idx_list)
        train_dataset = Subset(self._qm9_dataset, self._train_idx_list)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, collate_fn=self._collate_fn)
        return loader

    def valid_loader(self):
        loader = DataLoader(self._valid_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, collate_fn=self._collate_fn)
        return loader

    def test_loader(self):
        loader = DataLoader(self._test_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, collate_fn=self._collate_fn)
        return loader

    def _do_transform(self, np_dict):
        np_dict['atom_num'] = np_dict['atom_attr'].shape[0]
        np_dict['bond_num'] = np_dict['edge_dst'].shape[0]
        # np_dict['y'] = (np_dict['y'][self.target_idx].reshape((-1,)) - self.train_set_y_mean) / self.train_set_y_std
        np_dict['y'] = np_dict['y'][self.target_idx].reshape((-1,))
        bond_attr = np_dict['bond_attr']
        bond_dist = np_dict['bond_dist']
        if 'norm_dist' in self._transform_fn_kwargs:
            if self._transform_fn_kwargs['norm_dist']:
                bond_dist = bond_dist / bond_dist.max()
            del self._transform_fn_kwargs['norm_dist']

        np_dict['atom_attr'] = torch.from_numpy(np_dict['atom_attr'])
        np_dict['bond_attr'] = torch.from_numpy(np.hstack([bond_attr, bond_dist]))
        return self._transform_fn(np_dict, **self._transform_fn_kwargs), torch.from_numpy(np_dict['y'])

    def _norm_fn(self, sample_data):
        dglhg, label = sample_data
        label = (label-self.train_set_y_mean) / self.train_set_y_std
        return dglhg, label

    @property
    def atom_attr_dim(self):
        return self._qm9_dataset.inputs_size()['node_feat_size']

    @property
    def bond_attr_dim(self):
        return self._qm9_dataset.inputs_size()['edge_feat_size'] + 1

    @property
    def atom_attr_sizes(self):
        return None

    @property
    def bond_attr_sizes(self):
        return None

    @property
    def num_tasks(self):
        return 1

    @property
    def eval_metric(self):
        return 'mae'

    def evaluate(self, preds, targets):
        mae = np.mean(np.abs((preds - targets) * self.train_set_y_std))
        return mae
