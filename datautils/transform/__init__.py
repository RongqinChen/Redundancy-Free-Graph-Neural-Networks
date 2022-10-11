from .tpf_transform_from_npdict import transform_into_TPF_DGLHG_from_npdict
from .tpf_collate_from_dglhg import collate_TPF_dglhg

transform_fn_dict = {
    'TPF': transform_into_TPF_DGLHG_from_npdict,
    # 'CWComplex': extract_CWComplex_dglhg_from_smiles,
}


collate_fn_dict = {
    'TPF': collate_TPF_dglhg,
    # 'CWComplex': collate_CWComplex_dglhg,
}


__all__ = ['collate_fn_dict', 'transform_fn_dict']
