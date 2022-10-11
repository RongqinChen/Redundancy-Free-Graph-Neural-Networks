from torch import nn, Tensor, hstack
from dgl import DGLHeteroGraph
from dgl.udf import EdgeBatch, NodeBatch
import dgl.function as fn


__all__ = ['TPF_Encoder']


class TPF_Encoder(nn.Module):
    def __init__(self, tree_level: int, basic_dim: int, tree_dim: int, has_bond_feat: bool,
                 branch_norm: str, bracnh_nonlinear: str, tree_norm: str, tree_nonlinear: str,
                 tree_update: str, branch_drop_p: float, tree_drop_p: float, nest_drop_p: float) -> None:
        """TDFST Encoder

        Args:
            tree_level (int): level of TDFST
            basic_dim (int): dimension of basic feature, feature dimension of any bond and the atoms on the bottom layer
            tree_dim (int): feature dimension of any tree
            has_bond_feat (bool): has bond feature or not
            branch_norm (str): which normlizaion method to apply to Branch_Update?
            tree_norm (str): which normlizaion method to apply to Tree_Update?
            bracnh_nonlinear (str): which non-linear method to apply to Branch_Update?
            tree_nonlinear (str): which non-linear method to apply to Tree_Update?
            tree_update (str): which tree_update module?
            branch_drop_p (float): drop propability of branch update
            tree_drop_p (float): drop propability of tree (subtree) update
            nest_drop_p (float): drop propability of Dropout applied between nested blocks

        Args of `forward`:
            TDFST (DGLHeteroGraph)

        Returns:
            tree_x
        """
        super(TPF_Encoder, self).__init__()
        self.tree_level = tree_level
        self.has_bond_feat = has_bond_feat
        self.model_dict = nn.ModuleDict()

        BrhNormModule = getattr(nn, branch_norm)
        BrhNonLinearModule = getattr(nn, bracnh_nonlinear)
        TreeNormModule = getattr(nn, tree_norm)
        TreeNonLinearModule = getattr(nn, tree_nonlinear)

        if tree_update == 'Tree_Update_Sep':
            Tree_Update = Tree_Update_Sep
        elif tree_update == 'Tree_Update_Com':
            Tree_Update = Tree_Update_Com
        else:
            assert tree_update in {'Tree_Update_Sep', 'Tree_Update_Com'}

        for k in range(self.tree_level, 0, -1):
            self.model_dict[f'branch_update_{k}'] = Branch_Update(basic_dim, tree_dim, branch_drop_p, has_bond_feat,
                                                                  BrhNormModule, BrhNonLinearModule)
            if k == 1 and nest_drop_p > tree_drop_p:
                drop_p = nest_drop_p
            else:
                drop_p = tree_drop_p
            self.model_dict[f'tree_update_{k}'] = Tree_Update(basic_dim, tree_dim, drop_p, TreeNormModule, TreeNonLinearModule)

    def forward(self, TDFST: DGLHeteroGraph):
        for k in range(self.tree_level, 0, -1):
            TDFST[f'bond_{k}'].update_all(self.model_dict[f'branch_update_{k}'],
                                          fn.sum('branch_h', 'branch_h_agg'),
                                          self.model_dict[f'tree_update_{k}'])

        tree_x = TDFST.nodes['atom_0'].data['subtree_h']
        return tree_x


class Branch_Update(nn.Module):
    def __init__(self, basic_dim, tree_dim, branch_drop_p, has_bond_feat, NormModule, NonLinearModule):
        # basic_dim: bond_dim
        super().__init__()
        self.has_bond_feat = has_bond_feat
        if has_bond_feat:
            self.mlp = nn.Sequential(
                nn.Linear(tree_dim + basic_dim, tree_dim),
                NormModule(tree_dim),
                NonLinearModule(),
                nn.Linear(tree_dim, tree_dim),
                NormModule(tree_dim),
                NonLinearModule(),
                nn.Dropout(branch_drop_p)
            )

    def forward(self, bondbatch: EdgeBatch):
        subtree_h: Tensor = bondbatch.src['subtree_h']
        if self.has_bond_feat:
            bond_x: Tensor = bondbatch.data['bond_x']
            branch_x = hstack([subtree_h, bond_x])
            branch_h = self.mlp(branch_x)
        else:
            branch_h = subtree_h
        return {'branch_h': branch_h}


class Tree_Update_Com(nn.Module):
    def __init__(self, basic_dim, tree_dim, tree_drop_p, NormModule, NonLinearModule) -> None:
        """Tree_Update_Com, apply 2 layers of MLP on head node """
        super().__init__()
        self.tree_mlp = nn.Sequential(
            nn.Linear(basic_dim + tree_dim, tree_dim),
            NormModule(tree_dim),
            NonLinearModule(),
            nn.Linear(tree_dim, tree_dim),
            NormModule(tree_dim),
            NonLinearModule(),
            nn.Dropout(tree_drop_p)
        )

    def forward(self, headnode: NodeBatch):
        head_x: Tensor = headnode.data['atom_x']
        branch_h_agg: Tensor = headnode.data['branch_h_agg']
        subtree_x = hstack([head_x, branch_h_agg])
        subtree_h = self.tree_mlp(subtree_x)
        return {'subtree_h': subtree_h}


class Tree_Update_Sep(nn.Module):
    def __init__(self, basic_dim, tree_dim, tree_drop_p, NormModule, NonLinearModule) -> None:
        """Tree_Update_Sep, apply 1 layer of MLP on head node """

        super().__init__()
        self.subtree_mlp = nn.Sequential(
            nn.Linear(basic_dim + tree_dim, tree_dim),
            NormModule(tree_dim),
            NonLinearModule(),
            nn.Linear(tree_dim, tree_dim),
            NormModule(tree_dim),
            NonLinearModule(),
        )
        self.head_mlp = nn.Sequential(
            nn.Linear(basic_dim + tree_dim, tree_dim),
            NormModule(tree_dim),
            NonLinearModule(),
            nn.Dropout(tree_drop_p),
        )

    def forward(self, headnode: NodeBatch):
        head_x: Tensor = headnode.data['atom_x']
        branch_h_agg: Tensor = headnode.data['branch_h_agg']
        subtree_x = hstack([head_x, branch_h_agg])
        subtree_f = self.subtree_mlp(subtree_x)
        tree_h = hstack([head_x, subtree_f])
        subtree_h = self.head_mlp(tree_h)
        return {'subtree_h': subtree_h}
