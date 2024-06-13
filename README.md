# Catastrophic Forgetting under Distribution Shifts in Node Features, Graph Structure, Homophily, and Class Labels for Node Classification on Evolving Graphs

## Installation
1. Setup a python virtual environment (recommended)
2. Install [pytorch](https://pytorch.org/get-started/locally/)
3. Install [torch-geometric](https://github.com/rusty1s/pytorch_geometric)
4. Install other requirements via `pip install -r requirements.txt`

## File Descriptions

| File                                | Description                                                            |
|-------------------------------------|------------------------------------------------------------------------|
| src/fgn/                            | Adjusted source code for [FGN] (https://github.com/sair-lab/LGL)       |
| src/models/                         | Contains GCN, ER-GCN, and Reg-GCN                                      |
| src/util/                           | Contains code for early stopping                                       |
| src/csbm.py                         | Multi-class CSBM with its variations                                   |
| src/datasets.py                     | Preprocessing of real-world data                                       |
| src/generate_csbm.py                | Generates synthetic datasets                                           |
| src/measures.py                     | Important metrics for distributin shifts and performance               |
| src/node2vec_embedding.py           | node2vec                                                               |
| README.md                           | this file                                                              |
| requirements.txt                    | dependencies                                                           |
| src/train.py                        | main entry point for running experiments with GCN, ER-GCN, and Reg-GCN |
| src/<rw,csbm>_<shift_type>_shift.py | main entry point for calculating distribution shifts                   |

## Get the datasets
Download DBLP-hard [from zenodo](https://zenodo.org/record/3764770) and extraxt the files into `src/data` directory, such that it looks exactly like this:
- `src/data/dblp-hard/adjlist.txt`
- `src/data/dblp-hard/t.npy`
- `src/data/dblp-hard/X.npy`
- `src/data/dblp-hard/y.npy`

Run `src/generate_rw.py` to preprocess the three real-world datasets and `src/generate_csbm.py` to generate the synthetic datasets

## Run experiments
To calculate the average performance (AP) and average forgetting (AF) of baseline GCN, ER-GCN and Reg-GCN, specify the dataset in `src/train.py` in line <mark>34</mark>. You can choose both real-world:
```python
data_list = torch.load("data/real_world/<dataset>_tasks.pt")
```

or synthetic:
```python
data_list = torch.load("data/csbm/<dataset>.pt")
```

For FGN AP and AF on all datasets, copy the datasets into `src/fgn/data` and run `src/fgn/train.py`

## Plots
To reproduce the plots, you can run the Jupyter notebooks:
- `src/ignoring_the_problem.ipynb` for <b>Figure 2</b>
- `src/Untitled.ipynb` for <b>Figure 3</b>
- `src/feature_shift.ipynb` for<b>Figure 4</b>, <b>Figure 6</b>, <b>Figure 7</b>, and <b>Figure 8</b>
- `src/class_label_shift.ipynb` for <b>Figure 5</b>
- `src/real_world_feature.ipynb` for <b>Figure 9</b>
