from typing import List, Optional, Union

from dgl import DGLHeteroGraph
from torch import Tensor, nn
from torch.nn import Embedding, Module, ModuleList, init


class TPF_INIT(nn.Module):
    def __init__(self, tree_level: int,
                 atom_attr_sizes: Optional[List[int]], bond_attr_sizes: Optional[List[int]],
                 atom_attr_dim: Optional[int], bond_attr_dim: Optional[int],
                 basic_dim: int, tree_dim: int, lower_tree_dim: Optional[int],
                 nest_bottom: bool, transfrom_lower_tree: Union[bool, str],
                 norm: str, nonlinear: Optional[str],
                 where_assign_lower_tree: Optional[str]) -> None:
        """TPF_INIT Module, initializing TPF

        Args:
            tree_level (int): level of tree
            atom_attr_sizes (Optional[List[int]]): sizes of atom attributes
            bond_attr_sizes (Optional[List[int]]): sizes of bond attributes
            atom_attr_dim (Optional[int]): dimension of atom feature
            bond_attr_dim (Optional[int]): dimension of bond feature
            basic_dim (int): dimension of basic feature, feature dimension of any bond and the atoms on the bottom layer
            tree_dim (int): feature dimension of any tree
            lower_tree_dim (Optional[int]): feature dimension of trees of lower layer
            nest_bottom (bool): is bottom layer?
            transfrom_lower_tree (Union[bool, str]): do transform tree feature of lower layer?
            norm (str): which normlizaion method to apply?
            nonlinear (Optional[str]): which non-linear method to apply?
            where_assign_lower_tree (Optional[str]): which level assign lower layer tree feature to?

        Args of `forward`:
            TPF (DGLHeteroGraph): Truncated DFS Tree
            tree_x (Optional[Tensor]): feature of lower layer tree
        """
        super().__init__()
        assert tree_level > 0
        assert norm in {'BatchNorm1d', 'LayerNorm', 'Identity'}
        assert nonlinear in {'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Identity'}, f'Got {nonlinear}.'
        assert (atom_attr_sizes is None) != (atom_attr_dim is None)
        assert not (bond_attr_sizes is not None and bond_attr_dim is not None)
        NormModule = getattr(nn, norm)
        NonLinearModule = getattr(nn, nonlinear)
        self.module_dict = nn.ModuleDict()
        self.tree_level = tree_level
        self.atom_attr_sizes = atom_attr_sizes
        self.bond_attr_sizes = bond_attr_sizes
        self.atom_attr_dim = atom_attr_dim
        self.bond_attr_dim = bond_attr_dim
        self.basic_dim = basic_dim
        self.tree_dim = tree_dim
        self.lower_tree_dim = lower_tree_dim
        self.nest_bottom = nest_bottom
        self.transfrom_lower_tree = transfrom_lower_tree
        self.norm = norm
        self.nonlinear = nonlinear
        self.where_assign_lower_tree = where_assign_lower_tree

        # do bond initialization no matter which the nested layer is
        if bond_attr_sizes is not None:
            for k in range(1, tree_level + 1):
                self.module_dict[f'bond_{k}'] = AttrEmbedLinearReLU(bond_attr_sizes, basic_dim)

        if bond_attr_dim is not None:
            for k in range(1, tree_level + 1):
                self.module_dict[f'bond_{k}'] = nn.Sequential(nn.Linear(bond_attr_dim, basic_dim),
                                                              NormModule(basic_dim), NonLinearModule())

        # atom initialization
        if nest_bottom:
            assert tree_dim == basic_dim
            if atom_attr_sizes is not None:
                for k in range(tree_level+1):
                    self.module_dict[f'atom_{k}'] = AttrEmbedLinearReLU(atom_attr_sizes, basic_dim)

            if atom_attr_dim is not None:
                for k in range(tree_level+1):
                    self.module_dict[f'atom_{k}'] = nn.Sequential(nn.Linear(atom_attr_dim, basic_dim),
                                                                  NormModule(basic_dim), NonLinearModule())

        else:
            assert isinstance(transfrom_lower_tree, bool) or transfrom_lower_tree == 'auto'
            if transfrom_lower_tree == 'auto':
                transfrom_lower_tree = tree_dim != lower_tree_dim
            assert transfrom_lower_tree or (tree_dim == lower_tree_dim)
            assert tree_dim >= lower_tree_dim
            assert where_assign_lower_tree in {'bottom', 'every_level'}

            if where_assign_lower_tree == 'bottom':
                # level: 0, 1, ..., tree_level-1
                # only `basic_dim` channels, saving parameters
                if atom_attr_sizes is not None:
                    for k in range(tree_level):
                        self.module_dict[f'atom_{k}'] = AttrEmbedLinearReLU(atom_attr_sizes, basic_dim)

                if atom_attr_dim is not None:
                    for k in range(tree_level):
                        self.module_dict[f'atom_{k}'] = nn.Sequential(nn.Linear(atom_attr_dim, basic_dim),
                                                                      NormModule(basic_dim), NonLinearModule())
                # level: tree_level
                if not transfrom_lower_tree:
                    self.module_dict[f'atom_{tree_level}'] = nn.Identity()
                else:
                    self.module_dict[f'atom_{tree_level}'] = nn.Sequential(nn.Linear(lower_tree_dim, tree_dim),
                                                                           NormModule(tree_dim), NonLinearModule())

            if where_assign_lower_tree == 'every_level':
                if not transfrom_lower_tree:
                    for k in range(tree_level+1):
                        self.module_dict[f'atom_{k}'] = nn.Identity()
                else:
                    for k in range(tree_level+1):
                        self.module_dict[f'atom_{k}'] = nn.Sequential(nn.Linear(lower_tree_dim, tree_dim),
                                                                      NormModule(tree_dim), NonLinearModule())

    def forward(self, TPF: DGLHeteroGraph, tree_x: Optional[Tensor]):
        atom_attr = TPF.nodes['atom'].data['attr']
        if (self.bond_attr_sizes is not None or self.bond_attr_dim is not None):
            bond_attr = TPF.nodes['bond'].data['attr']
            for k in range(1, self.tree_level + 1):
                bond_x: Tensor = self.module_dict[f'bond_{k}'](bond_attr)
                b_k_oidx = TPF.edges[f'bond_{k}'].data['oidx']
                b_k_x = bond_x.index_select(0, b_k_oidx)
                TPF.edges[f'bond_{k}'].data['bond_x'] = b_k_x

        if self.nest_bottom:
            atom_x: Tensor = self.module_dict[f'atom_{0}'](atom_attr)
            TPF.nodes[f'atom_{0}'].data['atom_x'] = atom_x
            for k in range(1, self.tree_level+1):
                atom_x: Tensor = self.module_dict[f'atom_{k}'](atom_attr)
                a_k_oidx = TPF.nodes[f'atom_{k}'].data['oidx']
                a_k_x = atom_x.index_select(0, a_k_oidx)
                TPF.nodes[f'atom_{k}'].data['atom_x' if k < self.tree_level else 'subtree_h'] = a_k_x
        else:
            assert tree_x is not None
            assert tree_x.shape[0] == TPF.num_nodes('atom_0')

            if self.where_assign_lower_tree == 'bottom':
                # assign embedding of attributes to layers ranging 0 to self.tree_level-1
                atom_x: Tensor = self.module_dict[f'atom_{0}'](atom_attr)
                TPF.nodes[f'atom_{0}'].data['atom_x'] = atom_x
                for k in range(1, self.tree_level):
                    atom_x: Tensor = self.module_dict[f'atom_{k}'](atom_attr)
                    a_k_oidx = TPF.nodes[f'atom_{k}'].data['oidx']
                    a_k_x = atom_x.index_select(0, a_k_oidx)
                    TPF.nodes[f'atom_{k}'].data['atom_x'] = a_k_x

                # assign transformation of lower layer tree features to bottom level nodes
                k = self.tree_level
                subtree_x: Tensor = self.module_dict[f'atom_{k}'](tree_x)
                a_k_oidx = TPF.nodes[f'atom_{k}'].data['oidx']
                a_k_x = subtree_x.index_select(0, a_k_oidx)
                TPF.nodes[f'atom_{k}'].data['subtree_h'] = a_k_x

            if self.where_assign_lower_tree == 'every_level':
                subtree_x: Tensor = self.module_dict[f'atom_{0}'](tree_x)
                TPF.nodes[f'atom_{0}'].data['atom_x'] = subtree_x
                for k in range(1, self.tree_level+1):
                    subtree_x: Tensor = self.module_dict[f'atom_{k}'](tree_x)
                    a_k_oidx = TPF.nodes[f'atom_{k}'].data['oidx']
                    a_k_x = subtree_x.index_select(0, a_k_oidx)
                    TPF.nodes[f'atom_{k}'].data['atom_x' if k < self.tree_level else 'subtree_h'] = a_k_x


class AttrEmbedLinearReLU(Module):
    def __init__(self, attr_sizes, emb_dim) -> None:
        super(AttrEmbedLinearReLU, self).__init__()
        self.attr_sizes = attr_sizes
        self.emb_dim = emb_dim
        self.nns = ModuleList()
        for attr_size in attr_sizes:
            emb = Embedding(attr_size, emb_dim)
            init.xavier_uniform_(emb.weight.data)
            self.nns.append(emb)

        self.trans = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU())

    def forward(self, attrs):
        embed: Tensor = 0.
        for idx, attr in enumerate(attrs.T):
            embed += self.nns[idx](attr)

        feat = self.trans(embed)
        feat = embed
        return feat

    def __repr__(self) -> str:
        repr = 'AttrEmbed(attr_sizes={attr_sizes}, emb_dim={emb_dim})\n'
        repr += f'trans: {self.trans}'
        return repr.format(**self.__dict__)
