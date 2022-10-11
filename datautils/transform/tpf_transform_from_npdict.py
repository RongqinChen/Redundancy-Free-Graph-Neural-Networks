from typing import List, Tuple
from tqdm import tqdm
import random

from .tpf_make_dglhg import make_TPF_DGLHG


__all__ = ['transform_into_TPF_DGLHG_from_npdict']


def transform_into_TPF_DGLHG_from_npdict(np_dict: dict, level, edge_ratio=1.0):
    relate_dict, source_dict, attr_dict, num_nodes_dict = extract_graph_data(np_dict)
    relate_dict, source_dict, attr_dict, num_nodes_dict = extract_TPF_data(relate_dict, source_dict, attr_dict,
                                                                           num_nodes_dict, level, edge_ratio)
    TPF_dglhg = make_TPF_DGLHG(relate_dict, source_dict, attr_dict, num_nodes_dict)
    return TPF_dglhg


def extract_graph_data(np_dict: dict):
    # dicts to return
    relate_dict = {}
    attr_dict = {}
    num_nodes_dict = {}
    source_dict = {}

    num_nodes_dict['atom'] = np_dict['atom_num']
    attr_dict['atom_attr'] = np_dict['atom_attr']
    if 'bond_attr' in np_dict:
        attr_dict['bond_attr'] = np_dict['bond_attr']
    edge_src, edge_dst = np_dict['edge_src'], np_dict['edge_dst']
    relate_dict['atom2atom_src'] = relate_dict['atom2bond_src'] = edge_src
    relate_dict['atom2atom_dst'] = edge_dst
    edge_list = list(map(tuple, zip(edge_src, edge_dst)))
    edge_list_filtered = [nodetuple for nodetuple in edge_list if nodetuple[0] <= nodetuple[1]]
    atomtuple2bond_dict = {}
    for bond_idx, nodetuple in enumerate(edge_list_filtered):
        atomtuple2bond_dict[(nodetuple[0], nodetuple[1])] = bond_idx
        atomtuple2bond_dict[(nodetuple[1], nodetuple[0])] = bond_idx

    atom2atom_bond = [atomtuple2bond_dict[atomtuple] for atomtuple in edge_list]
    relate_dict['atom2bond_dst'] = source_dict['atom2atom_bond'] = atom2atom_bond
    num_nodes_dict['bond'] = max(atom2atom_bond) + 1
    return relate_dict, source_dict, attr_dict, num_nodes_dict


def extract_TPF_data(relate_dict, source_dict, attr_dict, num_nodes_dict, level, edge_ratio):
    attr_dict['level'] = level
    num_atoms = num_nodes_dict['atom']
    atom2atom_src = relate_dict['atom2atom_src']
    atom2atom_dst = relate_dict['atom2atom_dst']
    atom2atom_bond = source_dict['atom2atom_bond']
    v2v_tuple_list = zip(atom2atom_src, atom2atom_dst)
    path_atoms_map = list(v2v_tuple_list)

    num_nodes_dict['atom_0'] = num_atoms

    # ****************** level 1 *****************************
    # bond_1
    num_nodes_dict['atom_1'] = len(atom2atom_src)
    relate_dict['bond_1_dst'] = atom2atom_src  # ordered
    source_dict['atom_1'] = atom2atom_dst
    source_dict['bond_1'] = atom2atom_bond

    # high levels
    a2a_ptrs = make_a2a_ptrs(num_atoms, atom2atom_src)
    path_bonds_map = [[bond] for bond in atom2atom_bond]
    for k in range(2, level + 1):
        path_bonds_map, path_atoms_map, pre_path_idx_list, bond_oidx_list, atom_oidx_list = \
            expand_tree_leaves(path_bonds_map, path_atoms_map, atom2atom_bond, atom2atom_dst, a2a_ptrs, edge_ratio)

        # ** bond_k
        num_nodes_dict[f'atom_{k}'] = len(path_atoms_map)
        relate_dict[f'bond_{k}_dst'] = pre_path_idx_list  # ordered
        source_dict[f'bond_{k}'] = bond_oidx_list
        source_dict[f'atom_{k}'] = atom_oidx_list

    return relate_dict, source_dict, attr_dict, num_nodes_dict


def make_a2a_ptrs(num_atom: int, a2a_src: list):
    num_a2a = len(a2a_src)
    a2a_ptrs = [0] * (num_atom + 1)
    a2a_ptrs[num_atom] = num_a2a
    next_atom = 1
    for idx, src in enumerate(a2a_src):
        while src >= next_atom:
            a2a_ptrs[next_atom] = idx
            next_atom += 1

    return a2a_ptrs


def expand_tree_leaves(pre_path_bonds_map: List[Tuple[int, ...]],
                       pre_path_atoms_map: List[Tuple[int, ...]],
                       a2a_bonds: List[int], a2a_dsts: List[int], a2a_ptrs: List[int], edge_ratio): \

    # no repeated bonds
    path_bonds_map: List[Tuple[int, ...]] = []
    path_atoms_map: List[Tuple[int, ...]] = []
    pre_path_idx_list: List[int] = []  # prefix path (i.e. upper-level-atom)
    bond_oidx_list: List[int] = []  # source bond index
    atom_oidx_list: List[int] = []  # source atom index

    for pre_path_idx in tqdm(range(len(pre_path_atoms_map)), disable=True, position=0):
        pre_bonds = pre_path_bonds_map[pre_path_idx]
        pre_atoms = pre_path_atoms_map[pre_path_idx]
        relay_atom = pre_atoms[-1]

        if relay_atom == pre_atoms[0]:  # Back to the original node
            continue
        ptr_start, ptr_stop = a2a_ptrs[relay_atom], a2a_ptrs[relay_atom + 1]
        a2a_idx_list = list(range(ptr_start, ptr_stop))
        if edge_ratio < 1.:
            edge_ratio_list = [edge_ratio] * len(a2a_idx_list)
            a2a_idx_list = random.choices(a2a_idx_list, edge_ratio_list)
        for a2a_idx in tqdm(a2a_idx_list, disable=True, position=1, leave=False):
            a2a_bond = a2a_bonds[a2a_idx]
            if a2a_bond not in pre_bonds:  # Don't visit bonds that have been visited before
                pre_path_idx_list.append(pre_path_idx)
                bond_oidx_list.append(a2a_bond)
                atom = a2a_dsts[a2a_idx]
                atom_oidx_list.append(atom)
                path_bonds_map.append((*pre_bonds, a2a_bond))
                path_atoms_map.append((*pre_atoms, atom))

    return path_bonds_map, path_atoms_map, pre_path_idx_list, bond_oidx_list, atom_oidx_list
