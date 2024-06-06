# Catastrophic Forgetting under Distribution Shifts in Node Features, Graph Structure, Homophily, and Class Labels for Node Classification on Evolving Graphs

## Installation
1. Setup a python virtual environment (recommended)
2. Install [pytorch](https://pytorch.org/get-started/locally/)
3. Install [torch-geometric](https://github.com/rusty1s/pytorch_geometric)
4. Install other requirements via `pip install -r requirements.txt`

## File Descriptions

| File                            | Description                                                            |
|---------------------------------|------------------------------------------------------------------------|
| ./fgn/                          | Adjusted source code for [FGN] (https://github.com/sair-lab/LGL)       |
| ./models/                       | Contains GCN, ER-GCN, and Reg-GCN                                      |
| ./util/                         | Contains code for early stopping                                       |
| csbm.py                         | Multi-class CSBM with its variations                                   |
| datasets.py                     | Preprocessing of real-world data                                       |
| generate_csbm.py                | Generates synthetic datasets                                           |
| measures.py                     | Important metrics for distributin shifts and performance               |
| node2vec_embedding.py           | node2vec                                                               |
| README.md                       | this file                                                              |
| requirements.txt                | dependencies                                                           |
| train.py                        | main entry point for running experiments with GCN, ER-GCN, and Reg-GCN |
| <rw,csbm>_<shift_type>_shift.py | main entry point for calculating distribution shifts                   |
