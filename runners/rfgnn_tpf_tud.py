import json
from trainutils import DataModuleBase, HyperParameters, TrainerBase
from tqdm import tqdm
from datautils import TU_DataModule
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

    def test_epoch(self, epoch_idx) -> None:
        self._test_loss = None
        self._test_score = None
        self._best_test_loss = None
        self._best_test_score = None

    def train_batch(self, batch) -> float:
        graph_batch, labels = batch
        graph_batch = graph_batch.to(self._device)
        labels = labels.to(self._device)
        self._optimizer.zero_grad()
        preds = self._nn_model(graph_batch)
        loss: Tensor = self._loss_module(preds, labels.squeeze())
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
                    targets_list.append(labels.squeeze().cpu().numpy())
                    labels = labels.to(self._device).squeeze()
                    graph_batch = graph_batch.to(self._device)
                    preds: Tensor = self._nn_model(graph_batch)
                    preds_list.append(preds.detach().cpu().numpy())
                    loss = self._loss_module(preds, labels)
                    loss_list.append(loss.item())

            targets = np.hstack(targets_list)
            preds = np.vstack(preds_list)
            score = self._evaluate(preds, targets)

        loss = sum(loss_list) / len(loss_list)
        return loss, score


def make_hyper_p(datamodule: DataModuleBase, dataset_name,
                 nn_model_name, loss_module_name,
                 loss_module_args_dict, higher_better, fold_idx,
                 **para_dict):

    hyper_p = HyperParameters()
    hyper_p.datamodule_name = f'datautils.{datamodule.__name__}'
    hyper_p.datamodule_args_dict = {
        'batch_size': 128,
        'num_workers': 4,
        'transform_type': 'TPF',
        'save_transform': 'lmdb',
        'dataset_name': dataset_name,
        'chunk_size': 128,
        'transform_fn_kwargs': {'level': para_dict['tree_level']},
        'seed': 0,
        'fold_idx': fold_idx,
    }

    hyper_p.lr = 1e-2
    hyper_p.min_lr = 0.
    hyper_p.optimizer_name = 'Adam'
    hyper_p.optimizer_args = {'weight_decay': 0}
    hyper_p.loss_module_name = loss_module_name
    hyper_p.loss_module_args_dict = loss_module_args_dict
    hyper_p.eval_higher_is_better = higher_better
    hyper_p.max_num_epochs = 350
    hyper_p.early_stoping_patience = 350
    hyper_p.schedule_step = 50
    hyper_p.lr_decay_factor = 0.5
    hyper_p.nn_model_name = nn_model_name
    hyper_p.nn_model_args_dict = {
        'num_tasks': 'datamodule.num_tasks',  # int
        'atom_attr_dim': 'datamodule.atom_attr_dim',  # List[int]
        'bond_attr_sizes': 'datamodule.bond_attr_sizes',  # Optinal[List[int]]
    }
    hyper_p.nn_model_args_dict.update(para_dict)
    label = ''
    for key, val in para_dict.items():
        label += str(val) + ','

    label = label[:-1] + f'/fold_{fold_idx}'
    hyper_p.label = label
    return hyper_p


def hyper_p_search(datamodule, dataset_name, nn_model_name,
                   loss_module_name, loss_module_args_dict, higher_better,
                   num_repeats, config_dict: list, hyper_p_range: list):

    for hyper_p_id in hyper_p_range:
        para_dict = config_dict[hyper_p_id]
        for seed in range(1, num_repeats + 1):
            hyper_p = make_hyper_p(datamodule, dataset_name, nn_model_name,
                                   loss_module_name, loss_module_args_dict, higher_better, seed,
                                   **para_dict)
            yield hyper_p


def run(device=0, num_repeats=10):
    nn_model_name = 'models.RFGNN_TPF_Predict'
    dataset_name_list = ['ENZYMES', 'BZR', 'COX2', 'DHFR', ]
    # dataset_name_list = ['COX2']
    higher_better_list = [True] * len(dataset_name_list)

    config_path = 'config/rfgnn_tud.json'
    with open(config_path, 'r') as rfile:
        config_dict = json.load(rfile)

    hyper_p_range = list(map(str, range(len(config_dict))))
    for dataset_idx in range(len(dataset_name_list)):
        task_dict = {
            'datamodule': TU_DataModule,
            'dataset_name': dataset_name_list[dataset_idx],
            'nn_model_name': nn_model_name,
            'loss_module_name': 'torch.nn.CrossEntropyLoss',
            'loss_module_args_dict': {},
            'higher_better': higher_better_list[dataset_idx]
        }
        for hyper_p in hyper_p_search(**task_dict, num_repeats=num_repeats,
                                      config_dict=config_dict, hyper_p_range=hyper_p_range):
            hyper_p.device = device
            hyper_p.seed = 0
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
    if len(sys.argv) > 1:
        device = int(sys.argv[1])
        num_repeats = 10
        if len(sys.argv) > 2:
            num_repeats = int(sys.argv[2])

    run(device, num_repeats)
