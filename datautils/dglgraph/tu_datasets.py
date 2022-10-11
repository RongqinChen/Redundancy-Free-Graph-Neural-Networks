import random
import os.path as osp
import numpy as np
from dgl.data import TUDataset
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from trainutils import DataModuleBase

from .lmdb_dataset import LMDB_Dataset
from .individual_pkl_dataset import Individual_PKL_Dataset
from ..transform import collate_fn_dict, transform_fn_dict


class TU_DataModule(DataModuleBase):
    def __init__(self, dataset_name: str, chunk_size, fold_idx: int, batch_size: int, num_workers: int,
                 save_transform, transform_type: str, transform_fn_kwargs: dict, seed: int) -> None:
        super().__init__(batch_size, num_workers)
        self._dataset_name = dataset_name
        self._transform_type = transform_type
        self._dataset_util = TU_DatasetsUtil(dataset_name, chunk_size, fold_idx, save_transform, transform_type,
                                             transform_fn_kwargs, seed=seed)
        self._collate_fn = collate_fn_dict[transform_type]
        self._train_dataset = None
        self._valid_loader = None

    def __repr__(self) -> str:
        return f"TUDataset/{self._dataset_name}"

    def train_loader(self):
        if self._train_dataset is None:
            self._train_dataset = self._dataset_util.train_dataset()

        self._train_dataset.shuffle()
        loader = DataLoader(self._train_dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_workers, collate_fn=self._collate_fn)
        return loader

    def valid_loader(self):
        if self._valid_loader is None:
            dataset = self._dataset_util.valid_dataset()
            self._valid_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                                            num_workers=self.num_workers, collate_fn=self._collate_fn)
        return self._valid_loader

    @property
    def num_tasks(self):
        num_tasks_dict = {
            'DD': 2,
            'BZR': 2,
            'COX2': 2,
            'DHFR': 2,
            'ENZYMES': 6,
        }
        return num_tasks_dict[self._dataset_name]

    @property
    def atom_attr_sizes(self):
        return None

    @property
    def atom_attr_dim(self):
        atom_attr_dim_dict = {
            'DD': 1,
            'BZR': 3,
            'COX2': 3,
            'DHFR': 3,
            'ENZYMES': 18,
        }
        return atom_attr_dim_dict[self._dataset_name]

    @property
    def bond_attr_sizes(self):
        return None

    @property
    def bond_attr_dim(self):
        return None

    @property
    def ring_attr_sizes(self):
        return None

    @property
    def ring_attr_dim(self):
        return None

    @property
    def eval_metric(self):
        return 'acc'

    def evaluate(self, preds, targets):
        preds_a = np.argmax(preds, 1)
        correct = np.equal(preds_a, targets).sum()
        accuracy = correct / preds.shape[0]
        return accuracy


class TU_DatasetsUtil:
    '''Adapted from `ogb.graphproppred.GraphPropPredDataset`'''

    def __init__(self, dataset_name: str, chunk_size, fold_idx: int, save_transform: str, transform_type: str,
                 transform_fn_kwargs: dict, root='datasets', seed=0):
        '''
            - dataset_name (str): name of dataset
            - save_transform (str): which method to save transformed data, options are `no`, `lmdb` and `pkl`.
            - transform_type (str): which hetero graph transformed into
            - root (str): root directory to store the dataset folder
        '''
        super(TU_DatasetsUtil, self).__init__()
        assert 1 <= fold_idx <= 10, "fold_idx must be from 1 to 10."
        self._dataset_name = dataset_name
        self._chunk_size = chunk_size
        self._fold_idx = fold_idx
        self._seed = seed
        self._raw_root = root + '/TUDataset'
        self._save_transform = save_transform
        self._transform_type = transform_type
        self._transform_fn_kwargs = transform_fn_kwargs
        self._transform_fn = transform_fn_dict[transform_type]
        postfix = f'{self._transform_type}'
        for key, val in transform_fn_kwargs.items():
            postfix += (f".{key}_{val}")

        self._data_path = f'{self._raw_root}/{self._dataset_name}/{save_transform}.{postfix}'
        self._dglgraph_list = []
        self._label_list = []
        self._train_idx_list = None
        self._valid_idx_list = None
        self._load_dataset()
        self._pre_process()
        self._separate_data()
        self._train_dataset = None
        self._valid_loader = None

    def _load_dataset(self):
        self._dgl_dataset = TUDataset(self._dataset_name, raw_dir=self._raw_root, verbose=False)
        for dglgraph, label in self._dgl_dataset:
            self._dglgraph_list.append(dglgraph)
            self._label_list.append(label[0])

    def _pre_process(self):
        if self._save_transform == 'raw':
            return
        if osp.exists(self._data_path):
            return

        if self._save_transform == 'lmdb':
            lmdb_d = LMDB_Dataset(self._data_path, self._transform_fn, self._transform_fn_kwargs, self._chunk_size)
            lmdb_d.process_and_save(self._dglgraph_list, self._label_list)
        elif self._save_transform == 'indi_pkl':
            indi_pkl_d = Individual_PKL_Dataset(self._data_path, self._transform_fn, self._transform_fn_kwargs)
            indi_pkl_d.process_and_save(self._dglgraph_list, self._label_list)

    def _separate_data(self):
        fold_idx = self._fold_idx  # ranging from 1 to 10
        seed = self._seed
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        idx_list = []
        for idx in skf.split(np.zeros(len(self._label_list)), self._label_list):
            idx_list.append(idx)

        self._train_idx_list, self._valid_idx_list = idx_list[fold_idx - 1]

    def train_dataset(self):
        train_ds = _Simple_Dataset(self._save_transform, self._transform_fn, self._transform_fn_kwargs,
                                   self._data_path, self._dataset_name, self._chunk_size, self._raw_root, self._train_idx_list)
        return train_ds

    def valid_dataset(self):
        valid_ds = _Simple_Dataset(self._save_transform, self._transform_fn, self._transform_fn_kwargs,
                                   self._data_path, self._dataset_name, self._chunk_size, self._raw_root, self._valid_idx_list)
        return valid_ds


class _Simple_Dataset(Dataset):
    def __init__(self, save_transform, transform_fn, transform_fn_kwargs, data_path, dataset_name, chunk_size, raw_root, idx_list) -> None:
        super(_Simple_Dataset, self).__init__()
        self._save_transform = save_transform
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._data_path = data_path
        self._dataset_name = dataset_name
        self._chunk_size = chunk_size
        self._raw_root = raw_root
        self._idx_list = idx_list
        self._load_dataset()

    def _load_dataset(self):
        if 'raw' == self._save_transform:
            assert self._dataset_name is not None
            assert self._raw_root is not None
            self._dgl_dataset = TUDataset(self._dataset_name, self_loop=False, raw_dir=self._raw_root, verbose=False)
            assert max(self._idx_list) < len(self._dgl_dataset)
        elif 'lmdb' == self._save_transform:
            assert self._data_path is not None
            self._lmdb_d = LMDB_Dataset(self._data_path, self._transform_fn, self._transform_fn_kwargs, self._chunk_size)
            self._lmdb_d.init_read_mode()
        elif 'indi_pkl' == self._save_transform:
            assert self._data_path is not None
            self._indi_pkl_d = Individual_PKL_Dataset(self._data_path, self._transform_fn, self._transform_fn_kwargs)
        else:
            raise ValueError(f'Invalid save_transform option `{self._save_transform}`.')

    def __getitem__(self, index):
        if index > len(self):
            raise IndexError

        oidx = self._idx_list[index]
        if 'lmdb' == self._save_transform:
            (sample_dglhg, label) = self._lmdb_d[oidx]
        elif 'indi_pkl' == self._save_transform:
            (sample_dglhg, label) = self._indi_pkl_d[oidx]
        elif 'raw' == self._save_transform:
            (graph_dglg, label) = self._dgl_dataset[oidx]
            sample_dglhg = self._transform_fn(graph_dglg, **self._transform_fn_kwargs)
        else:
            raise ValueError(f'Invalid save_transform option `{self._save_transform}`.')

        return sample_dglhg, label

    def __len__(self):
        return len(self._idx_list)

    def shuffle(self):
        random.shuffle(self._idx_list)
