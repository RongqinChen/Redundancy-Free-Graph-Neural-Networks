# derived from https://github.com/muhanzhang/NestedGNN/blob/master/qm9.py

import sys
import errno
import pandas as pd
from six.moves import urllib
import ssl
import os
import os.path as osp

import numpy as np

from tqdm import tqdm
import zipfile
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from torch.utils.data import Dataset
from setting import Disable_Tqdm
rdBase.DisableLog('rdApp.error')


class QM9Dataset(Dataset):
    HAR2EV = 27.2113825435
    KCALMOL2EV = 0.04336414

    conversion = np.array([
        1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
        1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.], np.float32)[:12]

    raw_url = ('https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'http://www.roemisch-drei.de/qm9.zip'
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
    bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}
    raw_file_names = ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']
    raw_dir = 'datasets/QM9/'
    target_name_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']

    def __init__(self, transform, process='pre', one_hot_atom=True, norm_fn=False) -> None:
        self.one_hot_atom = one_hot_atom
        self.transform = transform
        self.norm_fn = norm_fn
        self.raw_paths = [f'{self.raw_dir}{filename}' for filename in self.raw_file_names]
        self.proccessed_file = f'{self.raw_dir}datalist.pkl'
        self.download()
        self.load_meta_datas()

        if process == 'pre':
            self.data_list = [self._proccess(index)
                              for index in tqdm(range(len(self)), disable=Disable_Tqdm)]
            self.data_list = [self.transform(data) for data in tqdm(self.data_list, 'pre transform', disable=Disable_Tqdm)]
        else:
            self.data_list = None

    def download(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        if all([osp.isfile(raw_path) for raw_path in self.raw_paths]):
            return
        if osp.isfile(f'{self.raw_dir}/qm9.zip'):
            extract_zip(f'{self.raw_dir}/qm9.zip', self.raw_dir)
        else:
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

        file_path = download_url(self.raw_url2, self.raw_dir)
        os.rename(osp.join(self.raw_dir, '3195404'), osp.join(self.raw_dir, 'uncharacterized.txt'))

    def load_meta_datas(self):
        targets = pd.read_csv(self.raw_paths[1])
        targets = targets[self.target_name_list].to_numpy()
        self._target = targets * self.conversion.reshape(1, -1)

        # with open(self.raw_paths[1], 'r') as f:
        #     target = f.read().split('\n')[1:-1]
        #     target = [[float(x) for x in line.split(',')[1:20]]
        #               for line in target]
        #     target = np.array(target, dtype=np.float)
        #     target = np.hstack([target[:, 3:], target[:, :3]])
        #     self._target = target * self.conversion.reshape(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            self._skip = [int(x.split()[0]) for x in f.read().split('\n')[9:-2]]
        assert len(self._skip) == 3054

        self._suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False)
        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self._factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        self._mol_list = []
        self._mol_idx_list = []
        self._pos_list = []
        data_size = self._target.shape[0]
        for mol_idx, mol in tqdm(enumerate(self._suppl), total=data_size, dynamic_ncols=True, disable=Disable_Tqdm):
            if mol is None:
                continue
            if mol_idx in self._skip:
                continue
            self._mol_idx_list.append(mol_idx)
            self._mol_list.append(mol)

            text = self._suppl.GetItemText(mol_idx)
            N_atoms = mol.GetNumAtoms()
            # atom position
            pos = text.split('\n')[4:4 + N_atoms]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = np.array(pos, dtype=np.float32)
            self._pos_list.append(pos)

    def __len__(self):
        # return 1024
        return len(self._mol_idx_list)

    def __getitem__(self, index):
        assert 0 <= index < len(self)
        if self.data_list is not None:
            sample_data = self.data_list[index]
        else:
            sample_data = self._proccess(index)
            sample_data = self.transform(sample_data)
        sample_data = self.norm_fn(sample_data)
        return sample_data

    def _proccess(self, index):
        mol_idx, mol = self._mol_idx_list[index], self._mol_list[index]
        pos = self._pos_list[index]

        type_idx = []
        atomic_number = []
        acceptor = []
        donor = []
        aromatic = []
        sp = []
        sp2 = []
        sp3 = []
        num_hs = []
        for atom in mol.GetAtoms():
            type_idx.append(self.types[atom.GetSymbol()])
            atomic_number.append(atom.GetAtomicNum())
            donor.append(0)
            acceptor.append(0)
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            hybridization = atom.GetHybridization()
            sp.append(1 if hybridization == HybridizationType.SP else 0)
            sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

        feats = self._factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == 'Donor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    donor[k] = 1
            elif feats[j].GetFamily() == 'Acceptor':
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    acceptor[k] = 1

        x2 = np.array([
            atomic_number, acceptor, donor, aromatic, sp, sp2, sp3, num_hs],
            dtype=np.float32).transpose()
        if self.one_hot_atom:
            x1 = np.eye(len(self.types), dtype=np.float32)[type_idx]
            x = np.hstack([x1, x2])
        else:
            node_type = np.array(type_idx, np.int64)
            x = x2

        edge_src, edge_dst, bond_idx = [], [], []
        dist_list = []
        edge_set = set()
        for bond in mol.GetBonds():
            src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_set.add((src, dst))
            edge_set.add((dst, src))
            edge_src += [src, dst]
            edge_dst += [dst, src]
            bond_idx += [self.bonds[bond.GetBondType()]]
            dist = np.linalg.norm(pos[src] - pos[dst], ord=2, axis=-1).reshape(-1, 1)
            dist_list.append(dist)

        assert len(edge_set) == len(edge_src)
        edge_feat = np.eye(len(self.bonds), dtype=np.float32)[bond_idx]
        edge_dist = np.array(dist_list).reshape((-1, 1))

        y = self._target[mol_idx].reshape((-1))
        name = mol.GetProp('_Name')

        edge_src, edge_dst = np.array(edge_src), np.array(edge_dst)
        sample_data = {'atom_attr': x, 'pos': pos, 'edge_src': edge_src, 'bond_dist': edge_dist,
                       'edge_dst': edge_dst, 'bond_attr': edge_feat, 'y': y, 'name': name}

        if not self.one_hot_atom:
            sample_data['node_type'] = node_type

        return sample_data

    def inputs_size(self,):
        if self.one_hot_atom:
            node_feat_size = len(self.types) + 8
            node_type_size = 0
        else:
            node_feat_size = 8
            node_type_size = 1

        edge_feat_size = len(self.bonds)
        return {'node_feat_size': node_feat_size, 'node_type_size': node_type_size, 'edge_feat_size': edge_feat_size}


# credit to torch_geometric.data.extract_zip
def extract_zip(path: str, folder: str):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
    """
    print(f'Extracting {path}', file=sys.stderr)
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(folder)


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(url: str, folder: str, log: bool = True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition('/')[2].split('?')[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print(f'Using existing file {filename}', file=sys.stderr)
        return path

    if log:
        print(f'Downloading {url}', file=sys.stderr)

    makedirs(folder)
    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, 'wb') as f:
        f.write(data.read())

    return path


r"""The QM9 dataset from the `"MoleculeNet: A Benchmark for Molecular
Machine Learning" <https://arxiv.org/abs/1703.00564>`_ paper, consisting of
about 130,000 molecules with 16 regression targets.
Each molecule includes complete spatial information for the single low
energy conformation of the atoms in the molecule.
In addition, we provide the atom features from the `"Neural Message
Passing for Quantum Chemistry" <https://arxiv.org/abs/1704.01212>`_ paper.

+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| Target | Property                         | Description                                                                       | Unit                                        |
+========+==================================+===================================================================================+=============================================+
| 0      | :math:`\mu`                      | Dipole moment                                                                     | :math:`\textrm{D}`                          |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 1      | :math:`\alpha`                   | Isotropic polarizability                                                          | :math:`{a_0}^3`                             |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 2      | :math:`\epsilon_{\textrm{HOMO}}` | Highest occupied molecular orbital energy                                         | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 3      | :math:`\epsilon_{\textrm{LUMO}}` | Lowest unoccupied molecular orbital energy                                        | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 4      | :math:`\Delta \epsilon`          | Gap between :math:`\epsilon_{\textrm{HOMO}}` and :math:`\epsilon_{\textrm{LUMO}}` | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 5      | :math:`\langle R^2 \rangle`      | Electronic spatial extent                                                         | :math:`{a_0}^2`                             |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 6      | :math:`\textrm{ZPVE}`            | Zero point vibrational energy                                                     | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 7      | :math:`U_0`                      | Internal energy at 0K                                                             | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 8      | :math:`U`                        | Internal energy at 298.15K                                                        | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 9      | :math:`H`                        | Enthalpy at 298.15K                                                               | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 10     | :math:`G`                        | Free energy at 298.15K                                                            | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 11     | :math:`c_{\textrm{v}}`           | Heat capavity at 298.15K                                                          | :math:`\frac{\textrm{cal}}{\textrm{mol K}}` |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 12     | :math:`U_0^{\textrm{ATOM}}`      | Atomization energy at 0K                                                          | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 13     | :math:`U^{\textrm{ATOM}}`        | Atomization energy at 298.15K                                                     | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 14     | :math:`H^{\textrm{ATOM}}`        | Atomization enthalpy at 298.15K                                                   | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 15     | :math:`G^{\textrm{ATOM}}`        | Atomization free energy at 298.15K                                                | :math:`\textrm{eV}`                         |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 16     | :math:`A`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 17     | :math:`B`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+
| 18     | :math:`C`                        | Rotational constant                                                               | :math:`\textrm{GHz}`                        |
+--------+----------------------------------+-----------------------------------------------------------------------------------+---------------------------------------------+

Args:
    root (string): Root directory where the dataset should be saved.
    transform (callable, optional): A function/transform that takes in an
        :obj:`torch_geometric.data.Data` object and returns a transformed
        version. The data object will be transformed before every access.
        (default: :obj:`None`)
    pre_transform (callable, optional): A function/transform that takes in
        an :obj:`torch_geometric.data.Data` object and returns a
        transformed version. The data object will be transformed before
        being saved to disk. (default: :obj:`None`)
    pre_filter (callable, optional): A function that takes in an
        :obj:`torch_geometric.data.Data` object and returns a boolean
        value, indicating whether the data object should be included in the
        final dataset. (default: :obj:`None`)
"""  # noqa: E501
