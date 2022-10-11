from dgl import DGLHeteroGraph
from torch import nn
from typing import Optional, List, Union

from .tpf_init import TPF_INIT
from .tpf_encoder import TPF_Encoder

__all__ = ['Nested_TPF_Encoder']


class Nested_TPF_Encoder(nn.Module):
    def __init__(self,
                 atom_attr_sizes: Optional[List[int]] = None,
                 bond_attr_sizes: Optional[List[int]] = None,
                 atom_attr_dim: Optional[int] = None,
                 bond_attr_dim: Optional[int] = None,
                 embed_dim_list: List[int] = [64], tree_level: int = 3,
                 transfrom_lower_tree: Union[bool, str] = True,
                 init_norm: str = 'LayerNorm', init_nonlinear: str = 'LeakyReLU',
                 branch_norm: str = 'LayerNorm', bracnh_nonlinear: str = 'LeakyReLU',
                 tree_norm: str = 'LayerNorm', tree_nonlinear: str = 'LeakyReLU',
                 tree_update: str = 'Tree_Update_Sep',
                 branch_drop_p: float = 0.0, tree_drop_p: float = 0.0, nest_drop_p: float = 0.5,
                 where_assign_lower_tree: str = 'every_level'
                 ):
        super().__init__()

        self.num_nested_layers = len(embed_dim_list)
        self.module_dict = nn.ModuleDict()

        basic_dim = lower_tree_dim = embed_dim_list[0]
        has_bond_feat = (bond_attr_sizes is not None or bond_attr_dim is not None)

        for n in range(self.num_nested_layers):
            tree_dim = embed_dim_list[n]
            self.module_dict[f'tdfst_init_{n}'] = TPF_INIT(tree_level, atom_attr_sizes, bond_attr_sizes,
                                                           atom_attr_dim, bond_attr_dim, basic_dim, tree_dim,
                                                           lower_tree_dim, n == 0, transfrom_lower_tree,
                                                           init_norm, init_nonlinear, where_assign_lower_tree)
            self.module_dict[f'tPF_encoder_{n}'] = TPF_Encoder(tree_level, basic_dim, tree_dim, has_bond_feat,
                                                               branch_norm, bracnh_nonlinear, tree_norm, tree_nonlinear,
                                                               tree_update, branch_drop_p, tree_drop_p, nest_drop_p)
            lower_tree_dim = tree_dim

    def forward(self, TDFST: DGLHeteroGraph):
        tree_x = None
        for n in range(self.num_nested_layers):
            self.module_dict[f'tdfst_init_{n}'](TDFST, tree_x)
            tree_x = self.module_dict[f'tPF_encoder_{n}'](TDFST)

        tree_h = TDFST.nodes['atom_0'].data['subtree_h']
        return tree_h
