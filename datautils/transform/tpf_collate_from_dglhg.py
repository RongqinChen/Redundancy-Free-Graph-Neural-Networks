from dgl.heterograph import DGLHeteroGraph
import torch as th
import dgl

import numpy as np
from typing import List, Tuple


def collate_TPF_dglhg(samples: List[Tuple[DGLHeteroGraph, np.ndarray]]):
    tpf_list = [sample[0] for sample in samples]
    label_list = [sample[1] for sample in samples]
    ntypes = tpf_list[0].ntypes
    level = max([int(ntype.split('_')[1]) for ntype in ntypes if '_' in ntype])
    ntype__num_nodes_list__dict = {
        ntype: [tpf.num_nodes(ntype) for tpf in tpf_list]
        for ntype in ntypes
    }
    ntype__onset_arr__dict = {
        node_type: np.cumsum([0] + num_nodes_list[:-1])
        for node_type, num_nodes_list in ntype__num_nodes_list__dict.items()
    }
    for idx, tpf in enumerate(tpf_list):
        for k in range(1, level + 1):
            bond_k_oidx = tpf.edges[f'bond_{k}'].data['oidx']
            tpf.edges[f'bond_{k}'].data['oidx'] = bond_k_oidx + ntype__onset_arr__dict['bond'][idx]
            atom_k_oidx = tpf.nodes[f'atom_{k}'].data['oidx']
            tpf.nodes[f'atom_{k}'].data['oidx'] = atom_k_oidx + ntype__onset_arr__dict['atom'][idx]

    tpf_batch = dgl.batch(tpf_list)
    labels = th.stack(label_list)
    return tpf_batch, labels
