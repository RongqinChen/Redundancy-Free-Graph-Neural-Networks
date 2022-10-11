from typing import Dict
import torch as th
import dgl
from dgl import DGLHeteroGraph


def make_TPF_DGLHG(relate_dict: Dict, source_dict: Dict, attr_dict: Dict, num_nodes_dict: Dict,
                   dtype=th.long):

    data_dict = {}
    level = attr_dict['level']

    atom2bond_src = th.tensor(relate_dict['atom2bond_src'], dtype=dtype)
    atom2bond_dst = th.tensor(relate_dict['atom2bond_dst'], dtype=dtype)
    data_dict[('atom', 'atom2bond', 'bond')] = (atom2bond_src, atom2bond_dst)

    for k in range(1, level + 1):
        bond_k_dst = th.tensor(relate_dict[f'bond_{k}_dst'], dtype=dtype)
        bond_k_src = th.arange(bond_k_dst.size(0), dtype=dtype)
        data_dict[(f'atom_{k}', f'bond_{k}', f'atom_{k-1}')] = (bond_k_src, bond_k_dst)

    TPF: DGLHeteroGraph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    # set node attributes
    TPF.nodes['atom'].data['attr'] = attr_dict['atom_attr']
    if 'bond_attr' in attr_dict:
        TPF.nodes['bond'].data['attr'] = attr_dict['bond_attr']

    for k in range(1, level + 1):
        bond_k_oidx = th.tensor(source_dict[f'bond_{k}'], dtype=dtype)
        TPF.edges[f'bond_{k}'].data['oidx'] = bond_k_oidx
        atom_k_oidx = th.tensor(source_dict[f'atom_{k}'], dtype=dtype)
        TPF.nodes[f'atom_{k}'].data['oidx'] = atom_k_oidx

    return TPF
