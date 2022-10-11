import importlib
import json
import os
import os.path as osp
from datetime import datetime
from decimal import Decimal
from typing import Tuple
import logging
import numpy as np
import random
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import dgl
from setting import Disable_Tqdm

from .datamodule import DataModuleBase
from .hyper_parameters import HyperParameters


class TrainerBase(object):
    def __init__(self, hyper_p: HyperParameters) -> None:
        super().__init__()
        self._hyper_p = hyper_p
        # seed all
        seed = hyper_p.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        dgl.seed(seed)

        if hyper_p.early_stoping_patience is None or hyper_p.early_stoping_patience < 1:
            hyper_p.early_stoping_patience = hyper_p.max_num_epochs

        self._log_root_dir = 'logs'
        if torch.cuda.is_available() and hyper_p.device > -1:
            self._device = torch.device('cuda:%d' % hyper_p.device)
        else:
            self._device = torch.device('cpu')

        print(self._device)
        self._data_module = self._make_data_module()
        self._nn_model = self._make_nn_model()
        self._n_parameters = hyper_p.n_parameters = self._count_parameters()
        self._loss_module = self._make_loss_module()
        self._optimizer = self._make_optimizer()
        self._lr_scheduler = self._make_lr_scheduler()
        self._evaluate = self._data_module.evaluate
        self._set_logging()

        # results
        self._improved = None
        self._best_epoch = 0
        self._train_loss = None
        self._train_score = None
        self._best_valid_loss = None
        self._best_valid_score = -np.inf if hyper_p.eval_higher_is_better else np.inf
        self._valid_loss = None
        self._valid_score = None
        self._test_loss = None
        self._test_score = None
        self._best_test_loss = None
        self._best_test_score = None

    def _make_data_module(self) -> DataModuleBase:
        hyper_p = self._hyper_p
        module_name, datamodule_name = hyper_p.datamodule_name.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            module_cls = getattr(module, datamodule_name)
        except Exception as e:
            raise ValueError(f'{e}\nInvalid data_module, {hyper_p.datamodule_name}')

        try:
            data_module = module_cls(**hyper_p.datamodule_args_dict)
        except Exception as e:
            raise ValueError(f'{e}\nInvalid data_module_paras, {hyper_p.datamodule_args_dict}')

        assert isinstance(data_module, DataModuleBase)
        return data_module

    def _make_nn_model(self) -> nn.Module:
        hyper_p = self._hyper_p
        module_name, nn_model_name = hyper_p.nn_model_name.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            model_module = getattr(module, nn_model_name)
        except Exception as e:
            raise ValueError(f'{e}\nInvalid nn_model, {hyper_p.nn_model_name}')

        nn_model_args_dict = hyper_p.nn_model_args_dict
        for key, val in nn_model_args_dict.items():
            if isinstance(val, str) and val.startswith('datamodule.'):
                prop = val[11:]
                prop_val = getattr(self._data_module, prop)
                nn_model_args_dict[key] = prop_val
        try:
            model = model_module(**nn_model_args_dict)
            assert isinstance(model, nn.Module)
        except Exception as e:
            raise ValueError(f'{e}\nInvalid nn_model_args_dict, {nn_model_args_dict}')

        model.to(self._device)
        return model

    def _make_optimizer(self) -> optim.Optimizer:
        try:
            optimizer_type = getattr(optim, self._hyper_p.optimizer_name)
            optimizer = optimizer_type(self._nn_model.parameters(), lr=self._hyper_p.lr, **self._hyper_p.optimizer_args)
        except Exception as e:
            raise ValueError(f'{e}\nInvalid optimizer, {self._hyper_p.optimizer_name}')

        return optimizer

    def _make_lr_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        schedule_step = self._hyper_p.schedule_step
        lr_decay_factor = self._hyper_p.lr_decay_factor
        if schedule_step is not None and lr_decay_factor is not None and 0. < lr_decay_factor < 1.:
            try:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, schedule_step, gamma=lr_decay_factor, )
            except Exception as e:
                raise ValueError(f'{e}\nInitialize scheduler, {schedule_step, lr_decay_factor}')
        else:
            lr_scheduler = None

        return lr_scheduler

    def _make_loss_module(self) -> nn.Module:
        hyper_p = self._hyper_p
        module_name, fn_name = hyper_p.loss_module_name.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            raise ValueError(f'{e}\nInvalid loss_module, {hyper_p.loss_module_name}')

        try:
            loss_module = getattr(module, fn_name)(**hyper_p.loss_module_args_dict)
            assert isinstance(loss_module, nn.Module)
        except Exception as e:
            raise ValueError(f'{e}\nInvalid loss_module_args_dict, {hyper_p.loss_module_args_dict}')

        return loss_module

    def _test_improve(self, epoch_idx):
        eval_higher_is_better = self._hyper_p.eval_higher_is_better
        if (eval_higher_is_better and self._valid_score > self._best_valid_score) or \
           (not eval_higher_is_better and self._valid_score < self._best_valid_score):
            self._best_epoch = epoch_idx
            self._best_valid_loss = self._valid_loss
            self._best_valid_score = self._valid_score
            self._improved = True
        else:
            self._improved = False

    def _set_logging(self) -> None:
        hyper_p = self._hyper_p
        timestamp = datetime.now().strftime('%m%d-%H%M%S')
        datamodule_name = str(self._data_module)
        nn_model_name = hyper_p.nn_model_name.rsplit('.', 1)[1]
        self._data_model_log_dir = osp.join(self._log_root_dir, datamodule_name, nn_model_name)
        exper_log_dir = osp.join(self._data_model_log_dir, hyper_p.label, timestamp)
        self._exper_log_dir = exper_log_dir
        os.makedirs(exper_log_dir)
        hyper_p_dict = hyper_p.todict()
        with open(f'{exper_log_dir}/hyper_parameters.json', 'w') as wfile:
            json.dump(hyper_p_dict, wfile, indent=4)

        self._summary_writer = SummaryWriter(log_dir=exper_log_dir)
        self._logger = logging.getLogger()
        self._logger.handlers.clear()
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'{exper_log_dir}/train.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)
        self._logger.debug(self._nn_model)
        self._logger.debug(self._optimizer)

    def _count_parameters(self) -> int:
        model_parameters = filter(lambda p: p.requires_grad, self._nn_model.parameters())
        n_parameters = sum([np.prod(p.size()) for p in model_parameters])
        return int(n_parameters)

    def _update_statues(self, epoch_idx) -> None:
        lr_list = [f"{Decimal(param_group['lr']):.1E}" for param_group in self._optimizer.param_groups]
        train_loss_str = f'{self._train_loss:.5f}' if self._train_loss > 1e-4 else f'{Decimal(self._train_loss):.2E}'
        valid_loss_str = f'{self._valid_loss:.5f}' if self._valid_loss > 1e-4 else f'{Decimal(self._valid_loss):.2E}'

        if self._test_loss is None:
            test_loss_str = ' ' * 7
        elif self._test_loss > 1e-4:
            test_loss_str = f'{self._test_loss:.5f}'
        else:
            test_loss_str = f'{Decimal(self._test_loss):.2E}'

        if self._test_score is None:
            test_score_str = ' ' * 7
        elif self._test_score > 1e-4:
            test_score_str = f'{self._test_score:.5f}'
        else:
            test_score_str = f'{Decimal(self._test_score):.2E}'

        if self._improved:
            statue_dict = {
                'score': f'{self._train_score:.5f}/{self._valid_score:.5f}*,' + ' ' * 7 + '/' +
                f'{test_score_str}*,' + ' ' * 7,
                'loss': f'{train_loss_str}/{valid_loss_str}*,' + ' ' * 7 + f'/{test_loss_str}*,' + ' ' * 7,
            }
        else:
            if self._best_valid_loss > 1e-4:
                best_valid_loss_str = f'{self._best_valid_loss:.5f}'
            else:
                best_valid_loss_str = f'{Decimal(self._best_valid_loss):.2E}'

            if self._best_test_loss is None:
                best_test_loss_str = ' ' * 7
            elif self._best_test_loss > 1e-4:
                best_test_loss_str = f'{self._best_test_loss:.5f}'
            else:
                best_test_loss_str = f'{Decimal(self._best_test_loss):.2E}'

            if self._best_test_score is None:
                best_test_score_str = ' ' * 7
            elif self._best_test_score > 1e-4:
                best_test_score_str = f'{self._best_test_score:.5f}'
            else:
                best_test_score_str = f'{Decimal(self._best_test_score):.2E}'

            statue_dict = {
                'score':
                f'{self._train_score:.5f}/{self._best_valid_score:.5f}*,{self._valid_score:.5f}/' +
                f'{best_test_score_str}*,{test_score_str}',
                'loss': f'{train_loss_str}/{best_valid_loss_str}*,{valid_loss_str}/{best_test_loss_str}*,{test_loss_str}',
            }

        statue_list = [f'{key}: {val}' for key, val in statue_dict.items()]
        statue = '  '.join(statue_list)
        epoch_lr = f'Epoch {epoch_idx+1:03d}/{self._best_epoch+1:03d} LR: ' + ','.join(lr_list)
        print(f'{epoch_lr:30s} | {statue}')
        self._logger.debug(epoch_lr + ' ' + statue)

    def _save_result(self) -> None:
        eval_metric = self._data_module.eval_metric
        result_dict = {'n_parameters': self._n_parameters,
                       'best_epoch': self._best_epoch,
                       'train_loss': self._train_loss,
                       f'train_{eval_metric}': self._train_score,
                       'valid_loss*': self._best_valid_loss,
                       f'valid_{eval_metric}*': self._best_valid_score,
                       'test_loss': self._best_test_loss,
                       f'test_{eval_metric}': self._best_test_score, }

        fname = osp.join(self._data_model_log_dir, 'results.json')
        if os.path.isfile(fname):
            with open(fname, 'r') as rfile:
                whole_dict = json.load(rfile)
        else:
            whole_dict = dict()

        timestamp = datetime.now().strftime('%m%d-%H%M%S')
        whole_dict[self._hyper_p.label + '.' + timestamp] = {
            'log_dir': self._summary_writer.log_dir,
            'result': result_dict
        }
        with open(fname, 'w') as wfile:
            json.dump(whole_dict, wfile, indent=4)

    def _epoch_log(self, epoch_idx) -> None:
        eval_metric = self._data_module.eval_metric
        self._summary_writer.add_scalar('train/loss', self._train_loss, epoch_idx)
        self._summary_writer.add_scalar(f'train/{eval_metric}', self._train_score, epoch_idx)
        self._summary_writer.add_scalar('valid/loss', self._valid_loss, epoch_idx)
        self._summary_writer.add_scalar(f'valid/{eval_metric}', self._valid_score, epoch_idx)
        if self._test_loss is not None:
            self._summary_writer.add_scalar('test/loss', self._test_loss, epoch_idx)
            self._summary_writer.add_scalar(f'test/{eval_metric}', self._test_score, epoch_idx)

    def run(self,) -> None:
        self._logger.info(f'{self._hyper_p.label}\nn_parameters: {self._n_parameters}')
        self._logger.info(f'seed: {self._hyper_p.seed}')
        save_model_step = self._hyper_p.save_model_step
        for epoch_idx in range(self._hyper_p.max_num_epochs):
            print(self._exper_log_dir.split(os.path.sep, 1)[1])
            self.train_epoch(epoch_idx)
            self.valid_epoch(epoch_idx)
            self._test_improve(epoch_idx)
            if self._hyper_p.eval_test_loss_only_when_improved:
                self.test_epoch(epoch_idx)

            self._update_statues(epoch_idx)
            self._epoch_log(epoch_idx)
            self._improved = False
            if epoch_idx - self._best_epoch > self._hyper_p.early_stoping_patience:
                self._logger.info('\nTraining stopped after'
                                  f'{self._hyper_p.early_stoping_patience} epochs with no improvement!')
                break

            self.lr_scheduler_step()

            if save_model_step is not None and (epoch_idx+1) % save_model_step == 0:
                saving_path = os.path.join(self._exper_log_dir, 'checkpoint{}.pth'.format(epoch_idx))
                torch.save(
                    {'nn_model': self._nn_model.state_dict(),
                     "optimizer": self._optimizer.state_dict()},
                    saving_path)

        self._save_result()

    def lr_scheduler_step(self):
        if self._lr_scheduler is not None:
            if self._lr_scheduler.get_last_lr()[0] > self._hyper_p.min_lr:
                self._lr_scheduler.step()

    def train_epoch(self, epoch_idx) -> None:
        loader = self._data_module.train_loader()
        self._nn_model.train()
        loss_accsum = 0.

        with tqdm(loader, total=len(loader), dynamic_ncols=True, disable=Disable_Tqdm) as batches_tqdm:
            for idx, batched_data in enumerate(batches_tqdm):
                desc = 'Training'
                desc = f'{desc:30s} | Iteration #{idx+1}/{len(loader)}'
                batches_tqdm.set_description(desc)
                loss = self.train_batch(batched_data)
                loss_accsum += loss
                batches_tqdm.set_postfix(loss=loss)

            loss_mean = loss_accsum / (idx + 1)
            batches_tqdm.set_postfix(loss=loss_mean)

        if epoch_idx % self._hyper_p.eval_train_step == 0:
            loss, score = self.evaluate(loader, 'training')
            self._train_loss, self._train_score = loss, score

    def valid_epoch(self, epoch_idx) -> None:
        loader = self._data_module.valid_loader()
        loss, score = self.evaluate(loader, 'validation')
        self._valid_loss, self._valid_score = loss, score

    def test_epoch(self, epoch_idx) -> None:
        loader = self._data_module.test_loader()
        loss, score = 0., 0.
        if loader is not None:
            loss, score = self.evaluate(loader, 'test')
        if self._improved:
            self._best_test_loss, self._best_test_score = loss, score

        self._test_loss, self._test_score = loss, score

    def train_batch(self, batch) -> float:
        batched_graph, label = batch
        batched_graph.to(self._device)
        self._optimizer.zero_grad()
        preds = self._nn_model(batch)
        loss = self._loss_module(preds, label.to(self._device).to(torch.float32))
        loss.backward()
        self._optimizer.step()
        return loss.detach().cpu().item()

    def evaluate(self, loader, data_split) -> Tuple[float, float]:
        targets_list = []
        preds_list = []
        self._nn_model.eval()
        with torch.no_grad():
            with tqdm(loader, total=len(loader), dynamic_ncols=True, disable=Disable_Tqdm) as batches_tqdm:
                for idx, batch in enumerate(batches_tqdm):
                    desc = f'Evaluating on {data_split} set'
                    desc = f'{desc:30s} | Iteration #{idx+1}/{len(loader)}'
                    batches_tqdm.set_description(desc)
                    batched_graph, label = batch
                    targets_list.append(label)
                    batched_graph.to(self._device)
                    preds = self._nn_model(batched_graph)
                    preds_list.append(preds.detach().cpu())

        targets = torch.vstack(targets_list)
        preds = torch.vstack(preds_list)
        loss = self._loss_module(preds, targets.view(preds.shape).to(torch.float32)).item()
        score = self._evaluate(preds, targets).item()
        return loss, score

    def inference(self, loader):
        preds_list = []
        self._nn_model.eval()
        with torch.no_grad():
            with tqdm(loader, total=len(loader), dynamic_ncols=True, disable=Disable_Tqdm) as batches_tqdm:
                for idx, batch in enumerate(batches_tqdm):
                    batched_graph, _ = batch
                    batches_tqdm.set_description(f'Inference Iteration #{idx+1}/{len(loader)}')
                    batched_graph.to(self._device)
                    preds = self._nn_model(batched_graph)
                    preds_list.append(preds.detach().cpu())

        preds = torch.vstack(preds_list)
        return preds
