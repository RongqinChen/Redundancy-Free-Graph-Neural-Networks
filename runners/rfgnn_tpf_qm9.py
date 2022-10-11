import os
import json
from trainutils import DataModuleBase, HyperParameters, TrainerBase
from tqdm import tqdm
from datautils import QM9_DataModule
from torch.functional import Tensor
import numpy as np
import sys
from datetime import datetime
from typing import Tuple
import torch
from setting import Disable_Tqdm
import resource
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


class Trainer(TrainerBase):
    def __init__(self, hyper_p: HyperParameters) -> None:
        super().__init__(hyper_p)

    def train_batch(self, batch) -> float:
        graph_batch, labels = batch
        graph_batch = graph_batch.to(self._device)
        labels = labels.to(self._device).to(torch.float32).reshape((labels.shape[0], -1))
        self._optimizer.zero_grad()
        preds = self._nn_model(graph_batch)
        loss: Tensor = self._loss_module(preds, labels)
        loss.backward()
        self._optimizer.step()
        return loss.detach().cpu().item()

    def evaluate(self, loader, data_split) -> Tuple[float, float]:
        self._nn_model.eval()
        targets_list = []
        preds_list = []
        loss_list = []

        with torch.no_grad():
            with tqdm(loader, total=len(loader), dynamic_ncols=True, disable=Disable_Tqdm) as batches_tqdm:
                for idx, batch in enumerate(batches_tqdm):
                    desc = f'Evaluating on {data_split} set'
                    desc = f'{desc:30s} | Iteration #{idx+1}/{len(loader)}'
                    batches_tqdm.set_description(desc)
                    graph_batch, labels = batch
                    targets_list.append(labels.cpu().numpy())
                    labels = labels.to(self._device).to(torch.float32).reshape((labels.shape[0], -1))
                    graph_batch = graph_batch.to(self._device)
                    preds: Tensor = self._nn_model(graph_batch)
                    preds_list.append(preds.detach().cpu().numpy())
                    loss = self._loss_module(preds, labels)
                    batches_tqdm.set_postfix(loss=loss.item())
                    loss_list.append(loss.item())

            targets = np.vstack(targets_list)
            preds = np.vstack(preds_list)
            score = self._evaluate(preds, targets)

        loss = sum(loss_list) / len(loss_list)
        return loss, score

    def _make_lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        patience = self._hyper_p.patience
        lr_decay_factor = self._hyper_p.lr_decay_factor
        if patience is not None and lr_decay_factor is not None and 0. < lr_decay_factor < 1.:
            try:
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self._optimizer, factor=lr_decay_factor, patience=patience, min_lr=0.00001)
            except Exception as e:
                raise ValueError(f'{e}\nInitialize scheduler, {patience, lr_decay_factor}')
        else:
            lr_scheduler = None

        return lr_scheduler

    def lr_scheduler_step(self):
        if self._lr_scheduler is not None:
            self._lr_scheduler.step(self._valid_loss)


def make_hyper_p(datamodule: DataModuleBase,
                 nn_model_name, loss_module_name,
                 loss_module_args_dict, higher_better, target_idx,
                 **para_dict):

    hyper_p = HyperParameters()
    hyper_p.datamodule_name = f'datautils.{datamodule.__name__}'
    hyper_p.datamodule_args_dict = {
        'batch_size': 64,
        'num_workers': 8,
        'transform_type': 'TPF',
        'save_transform': 'raw',
        'transform_fn_kwargs': {'level': para_dict['tree_level'], 'norm_dist': True, 'pre_transform': True},
        'target_idx': target_idx,
    }

    hyper_p.lr = 1e-3
    hyper_p.min_lr = 0.00001
    hyper_p.optimizer_name = 'Adam'
    hyper_p.optimizer_args = {'weight_decay': 0}
    hyper_p.loss_module_name = loss_module_name
    hyper_p.loss_module_args_dict = loss_module_args_dict
    hyper_p.eval_higher_is_better = higher_better
    hyper_p.eval_train_step = 10
    hyper_p.max_num_epochs = 200
    hyper_p.early_stoping_patience = 200
    hyper_p.patience = 5
    hyper_p.lr_decay_factor = 0.7
    hyper_p.nn_model_name = nn_model_name
    hyper_p.eval_test_loss_only_when_improved = True
    hyper_p.nn_model_args_dict = {
        'num_tasks': 'datamodule.num_tasks',  # int
        'atom_attr_dim': 'datamodule.atom_attr_dim',  # List[int]
        'atom_attr_sizes': 'datamodule.atom_attr_sizes',  # List[int]
        'bond_attr_dim': 'datamodule.bond_attr_dim',  # Optinal[List[int]]
        'bond_attr_sizes': 'datamodule.bond_attr_sizes',  # Optinal[List[int]]
    }
    hyper_p.nn_model_args_dict.update(para_dict)
    label = ''
    for key, val in para_dict.items():
        label += str(val) + ','

    label = label[:-1] + f'/target_{target_idx}'
    hyper_p.label = label
    return hyper_p


def hyper_p_search(datamodule, nn_model_name, target_idx,
                   loss_module_name, loss_module_args_dict, higher_better,
                   num_repeats, config_dict: list, hyper_p_range: list):

    for hyper_p_id in hyper_p_range:
        para_dict = config_dict[hyper_p_id]
        for seed in range(1, num_repeats + 1):
            hyper_p = make_hyper_p(datamodule, nn_model_name,
                                   loss_module_name, loss_module_args_dict, higher_better, target_idx,
                                   **para_dict)
            hyper_p.seed = seed
            yield hyper_p


def run(device=0, num_repeats=10, target_idx=0):
    nn_model_name = 'models.RFGNN_TPF_Predict'
    dataset_name_list = ['qm9']
    higher_better_list = [False]

    config_path = 'config/rfgnn_qm9.json'
    with open(config_path, 'r') as rfile:
        config_dict = json.load(rfile)

    hyper_p_range = list(map(str, range(len(config_dict))))
    for dataset_idx in range(len(dataset_name_list)):
        task_dict = {
            'datamodule': QM9_DataModule,
            # 'dataset_name': dataset_name_list[dataset_idx],
            'nn_model_name': nn_model_name,
            'target_idx': target_idx,
            'loss_module_name': 'torch.nn.L1Loss',
            'loss_module_args_dict': {},
            'higher_better': higher_better_list[dataset_idx]
        }
        for hyper_p in hyper_p_search(**task_dict, num_repeats=num_repeats,
                                      config_dict=config_dict, hyper_p_range=hyper_p_range):
            hyper_p.device = device
            # hyper_p.seed = 0
            try:
                trainer = Trainer(hyper_p)
                trainer.run()
            except Exception as e:
                with open('run_error.log', 'a') as afile:
                    now = datetime.now()
                    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                    print(date_time, file=afile)
                    print(e, file=afile)
                raise e


if __name__ == '__main__':
    device = 0
    num_repeats = 10
    target_idx_begin, target_idx_end = 0, 12
    if len(sys.argv) > 1:
        device = int(sys.argv[1])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = 0
    if len(sys.argv) > 2:
        num_repeats = int(sys.argv[2])
    if len(sys.argv) > 3:
        target_idx_begin = int(sys.argv[3])
    if len(sys.argv) > 4:
        target_idx_end = int(sys.argv[4])

    for target_idx in range(target_idx_begin, target_idx_end):
        run(device, num_repeats, target_idx)
