from typing import Any, Union


class HyperParameters(object):
    def __init__(self) -> None:
        super().__init__()
        self.label: str = ''
        self.device: int = 0
        self.nn_model_name: str = ''
        self.nn_model_args_dict: dict = {}
        self.datamodule_name: str = ''
        self.datamodule_args_dict: dict = {}
        self.lr: float = 1e-4
        self.min_lr: float = 1e-5
        self.optimizer_name: str = 'Adam'
        self.optimizer_args: dict = {}
        self.schedule_step = None
        self.lr_decay_factor = None
        self.loss_module_name: str = ''
        self.loss_module_args_dict: dict = {}
        self.eval_higher_is_better: bool = True
        self.max_num_epochs = 100
        self.save_model_step = None
        self.early_stoping_patience: Union[None, int] = 10
        self.n_parameters = 0
        self.eval_train_step = 1
        self.eval_test_loss_only_when_improved = False
        self.seed = 0
        # self.__extra_attr = {}

    def todict(self):
        return self.__dict__

    def __setattr__(self, __name: str, __value: Any) -> None:
        self.__dict__[__name] = __value

    def __getattr__(self, __name: str) -> Any:
        return self.__dict__.get(__name, None)
