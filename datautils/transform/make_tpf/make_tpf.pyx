# distutils: language=c++
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map


cpdef __make_TPF(int tree_level, int num_atoms, vector[int] a2a_src_vec,
                 vector[int] a2a_dst_vec, vector[int] a2a_bond_vec):

    # dicts to return
    cdef map[string, vector[int]] relation_dict = map[string, vector[int]]()
    cdef map[string, int] num_nodes_dict = map[string, int]()
    cdef map[string, vector[int]] source_dict = map[string, vector[int]]()

    # make a2a_ptrs
    cdef vector[int] a2a_ptrs = make_a2a_ptrs(num_atoms, a2a_src_vec)

    # level 1
    num_nodes_dict[b'atom_0'] = num_atoms
    relation_dict[b'bond_1_dst'] = a2a_src_vec  # ordered
    source_dict[b'bond_1'] = a2a_bond_vec
    source_dict[b'atom_1'] = a2a_dst_vec
    num_nodes_dict[b'atom_1'] = a2a_src_vec.size()

    # high levels
    cdef vector[int] pre_path_idx_vec  # prefix path (i.e. upper-level-atom)
    cdef vector[int] bond_oidx_vec  # source bond index
    cdef vector[int] atom_oidx_vec  # source atom index
    cdef vector[vector[int]] path_atoms_vec = make_path_atoms_vec(a2a_src_vec, a2a_dst_vec)

    for k in range(2, tree_level + 1):
        path_atoms_vec, pre_path_idx_vec, bond_oidx_vec, atom_oidx_vec = \
            expand_tree_level(path_atoms_vec, a2a_bond_vec, a2a_dst_vec, a2a_ptrs)

        relation_dict[f'bond_{k}_dst'.encode('ascii')] = pre_path_idx_vec  # ordered
        source_dict[f'bond_{k}'.encode('ascii')] = bond_oidx_vec
        source_dict[f'atom_{k}'.encode('ascii')] = atom_oidx_vec
        num_nodes_dict[f'atom_{k}'.encode('ascii')] = atom_oidx_vec.size()

    return relation_dict, source_dict, num_nodes_dict


cdef vector[int] make_a2a_ptrs(int num_atoms, vector[int] a2a_src_vec):
    cdef int num_a2a = a2a_src_vec.size()
    cdef vector[int] a2a_ptrs = vector[int]()
    cdef int next_atom = 0

    for idx, src in enumerate(a2a_src_vec):
        while src >= next_atom:
            a2a_ptrs.push_back(idx)
            next_atom += 1

    while int(a2a_ptrs.size()) < num_atoms + 1:
        a2a_ptrs.push_back(num_a2a)
    return a2a_ptrs


cdef make_path_atoms_vec(vector[int] a2a_src_vec, vector[int] a2a_dst_vec):
    cdef vector[vector[int]] path_atoms_vec = vector[vector[int]]()
    cdef vector[int] atoms
    for idx in range(a2a_src_vec.size()):
        atoms = vector[int]()
        atoms.push_back(a2a_src_vec[idx])
        atoms.push_back(a2a_dst_vec[idx])
        path_atoms_vec.push_back(atoms)
    
    return path_atoms_vec


cdef expand_tree_level(vector[vector[int]] pre_path_atoms_vec,
                       vector[int] a2a_bond_vec, vector[int] a2a_dst_vec,
                       vector[int] a2a_ptrs):
    
    cdef vector[int] atoms
    cdef vector[vector[int]] path_atoms_vec = vector[vector[int]]()
    cdef vector[int] pre_path_idx_vec = vector[int]() # prefix path (i.e. upper-level-atom)
    cdef vector[int] bond_oidx_vec = vector[int]() # bonds' source index
    cdef vector[int] atom_oidx_vec = vector[int]() # atoms' source index
    cdef int ptr_start, ptr_stop
    cdef int a_oidx, b_oidx

    for pre_path_idx in range(pre_path_atoms_vec.size()):
        pre_atoms = pre_path_atoms_vec[pre_path_idx]
        relay_atom = pre_atoms.back()
        if relay_atom == pre_atoms.front():  # formed a ring
            continue

        ptr_start = a2a_ptrs[relay_atom]
        ptr_stop = a2a_ptrs[relay_atom + 1]
        for ptr in range(ptr_start, ptr_stop):
            a_oidx = a2a_dst_vec[ptr]
            if can_form_epath(pre_atoms, a_oidx) == 1:
                atoms = vector[int](pre_atoms)
                atoms.push_back(a_oidx)
                path_atoms_vec.push_back(atoms)
                pre_path_idx_vec.push_back(pre_path_idx)
                b_oidx = a2a_bond_vec[ptr]
                bond_oidx_vec.push_back(b_oidx)
                atom_oidx_vec.push_back(a_oidx)

    return path_atoms_vec, pre_path_idx_vec, bond_oidx_vec, atom_oidx_vec


cdef int can_form_epath(vector[int] vec, int val):
    if vec.size() == 2:
        for ele in vec:
            if ele == val:
                return -1
    else:
        for idx in range(1, vec.size()):
            if vec[idx] == val:
                return -1
    return 1
