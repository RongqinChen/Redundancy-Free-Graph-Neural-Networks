from dgl import DGLHeteroGraph
from torch import nn
from dgl.ops.segment import segment_reduce
from typing import Optional, List, Union, Any
from .nested_tpf_enc import Nested_TPF_Encoder

__all__ = ['RFGNN_TPF_Predict']


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


class RFGNN_TPF_Predict(nn.Module):
    def __init__(self, atom_attr_sizes: Optional[List[int]] = None,
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
                 where_assign_lower_tree: str = 'every_level',
                 graph_readout='mean', g_drop_p=0.5, num_tasks=1):

        super(RFGNN_TPF_Predict, self).__init__()
        self.embed_dim_list = embed_dim_list
        self.TDFST_encoder = Nested_TPF_Encoder(atom_attr_sizes, bond_attr_sizes, atom_attr_dim, bond_attr_dim,
                                                embed_dim_list, tree_level, transfrom_lower_tree,
                                                init_norm, init_nonlinear, branch_norm, bracnh_nonlinear,
                                                tree_norm, tree_nonlinear, tree_update,
                                                branch_drop_p, tree_drop_p, nest_drop_p, where_assign_lower_tree)
        tree_embed_dim = embed_dim_list[-1]
        self.pred_layer = nn.Sequential(
            nn.Linear(tree_embed_dim, tree_embed_dim),
            nn.ReLU(),
            nn.Dropout(g_drop_p),
            nn.Linear(tree_embed_dim, num_tasks),
        )
        self.graph_readout = graph_readout
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.TDFST_encoder)
        reset(self.pred_layer)

    def forward(self, TDFST: DGLHeteroGraph):
        tree_h = self.TDFST_encoder(TDFST)
        tree_h_agg = segment_reduce(TDFST.batch_num_nodes('atom'), tree_h, self.graph_readout)
        graph_pred = self.pred_layer(tree_h_agg)
        return graph_pred
