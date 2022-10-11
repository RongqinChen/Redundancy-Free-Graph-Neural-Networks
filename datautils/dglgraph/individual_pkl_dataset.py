import os
import pickle as pkl
from setting import Disable_Tqdm
from tqdm.std import tqdm
from .dglg_2_np_dict import convert_dglg_into_np_dict


class Individual_PKL_Dataset():
    def __init__(self, db_path, transform_fn, transform_fn_kwargs):
        self._db_path = db_path
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs

    def _do_transform(self, np_dict):
        return self._transform_fn(np_dict, **self._transform_fn_kwargs)

    def process_and_save(self, dglgraph_list, label_list):
        os.mkdir(self._db_path)
        assert len(dglgraph_list) == len(label_list)
        global_tqdm_iter = tqdm(range(len(label_list)),
                                dynamic_ncols=True, position=0, disable=Disable_Tqdm)

        np_dict_list = convert_dglg_into_np_dict(dglgraph_list)
        for idx in global_tqdm_iter:
            global_tqdm_iter.set_description('Processing')
            dglhg = self._do_transform(np_dict_list[idx])
            label = label_list[idx]
            path = f'{self._db_path}/{idx}.pkl'
            with open(path, 'wb') as wbf:
                pkl.dump((dglhg, label), wbf)

    def __getitem__(self, idx: int):
        path = f'{self._db_path}/{idx}.pkl'
        with open(path, 'rb') as rbf:
            dglhg, label = pkl.load(rbf)

        return dglhg, label
