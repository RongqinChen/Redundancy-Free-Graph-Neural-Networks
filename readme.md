# Redundancy-Free Graph Neural Networks

## Build Experiments Environment

```bash
conda create --name rfgnn_env python==3.8 pytorch cudatoolkit=11.3 \
rdkit graph-tool tensorboard dgl-cuda11.3 pip lmdb \
-c pytorch -c conda-forge -c rdkit -c dglteam -y

conda activate rfgnn_env

pip install tqdm ogb lmdb
```

## Compile `make_tpf` module 
```bash
bash datautils/transform/make_tpf/build.sh
```

## Usages

### GIN-bioinfo datasets (including 'MUTAG', 'NCI1', 'PROTEINS', 'PTC')

```bash 
python -m runners.rfgnn_tpf_gind_bioinfo $device $num_repeats
```

### TU datasets (including 'ENZYMES', 'BZR', 'COX2', 'DHFR')

```bash 
python -m runners.rfgnn_tpf_tud $device $num_repeats
```

### QM9 dataset

```bash 
python -m runners.rfgnn_tpf_qm9 $device $num_repeats $target_idx_begin $target_idx_end
```

Note that unit conversions should be performed for results to match the units used by [1].

[1]: Christopher Morris, Martin Ritzert, Matthias Fey, William L Hamilton, Jan Eric Lenssen, Gaurav Rattan, and Martin Grohe. Weisfeiler and leman go neural: Higher-order graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 4602–4609, 2019.
