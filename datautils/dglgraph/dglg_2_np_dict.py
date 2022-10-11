from typing import List
from dgl import DGLGraph
from tqdm import tqdm
import torch
from torch import Tensor
from setting import Disable_Tqdm


def convert_dglg_into_np_dict(dglg_list: List[DGLGraph]):
    np_dict_list = []
    for dglg in tqdm(dglg_list, desc='convert_into_np_dict', dynamic_ncols=True, position=1,
                     leave=False, disable=Disable_Tqdm):

        if 'attr' in dglg.ndata:
            atom_attr = dglg.ndata['attr']
        elif 'node_attr' in dglg.ndata:
            atom_attr = dglg.ndata['node_attr']
        else:
            atom_attr = torch.ones((dglg.num_nodes(), 1), dtype=torch.float32)

        if torch.is_floating_point(atom_attr):
            atom_attr: Tensor = atom_attr.to(torch.float32)

        edges_index = dglg.edges()
        edge_src, edge_dst = edges_index[0].tolist(), edges_index[1].tolist()
        np_dict = {
            'atom_num': dglg.num_nodes(),
            'atom_attr': atom_attr,
            'edge_src': edge_src,
            'edge_dst': edge_dst,
        }
        np_dict_list.append(np_dict)

    return np_dict_list
