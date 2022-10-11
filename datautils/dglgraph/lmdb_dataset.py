
import os
import pickle as pkl
import lmdb
from setting import Disable_Tqdm
# from tqdm.contrib.concurrent import process_map
from tqdm.std import tqdm
from .dglg_2_np_dict import convert_dglg_into_np_dict


class LMDB_Dataset():
    def __init__(self, db_path, transform_fn, transform_fn_kwargs, chunk_size):
        self._db_path = db_path
        self._transform_fn = transform_fn
        self._transform_fn_kwargs = transform_fn_kwargs
        self._chunk_size = chunk_size

    def _do_transform(self, np_dict):
        return self._transform_fn(np_dict, **self._transform_fn_kwargs)

    def process_and_save(self, dglgraph_list, label_list):
        assert len(dglgraph_list) == len(label_list)
        env = lmdb.open(self._db_path, map_size=2**43)
        txn = env.begin(write=True)
        global_tqdm_iter = tqdm(range(0, len(label_list), self._chunk_size),
                                dynamic_ncols=True, position=0, disable=Disable_Tqdm)

        for idx_onset in global_tqdm_iter:
            global_tqdm_iter.set_description(f"Processing, # {idx_onset}/{len(label_list)}")
            idx_offset = min(idx_onset + self._chunk_size, len(label_list))
            dglgraph_chunk = dglgraph_list[idx_onset: idx_offset]
            np_dict_chunk = convert_dglg_into_np_dict(dglgraph_chunk)
            label_chunk = label_list[idx_onset: idx_offset]
            idx_chunk = list(range(idx_onset, idx_offset))
            dglhg_chunk = map(self._do_transform, tqdm(np_dict_chunk, desc='    chunk processing',
                                                       dynamic_ncols=True, position=1,
                                                       leave=False, disable=Disable_Tqdm))
            # dglhg_chunk = process_map(self._do_transform, np_dict_chunk, desc=''*16+'chunk processing',
            #                           chunksize=self._chunk_size // 8, position=1, leave=False, disable=Disable_Tqdm)
            for oidx, dglhg, label in zip(idx_chunk, dglhg_chunk, label_chunk):
                key_byte = f'{oidx}'.encode('ascii')
                tr_data_byte = pkl.dumps((dglhg, label))
                txn.put(key_byte, tr_data_byte)

            txn.commit()
            txn = env.begin(write=True)
        print()

    def init_read_mode(self):
        assert os.path.exists(self._db_path)
        self.env = lmdb.open(self._db_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx: int):
        key_byte = f'{idx}'.encode('ascii')
        data_byte = self.txn.get(key_byte)
        dglhg, label = pkl.loads(data_byte)
        return dglhg, label
