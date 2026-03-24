Project Name
A hypergraph-based training framework for subhypergraph classification.

1. Project Overview
This project implements a hypergraph representation learning framework for classification over target hyperedges.
Given a global hypergraph in JSON format, the code samples a local k-hop subhypergraph around each target hyperedge, encodes node and hyperedge features, and performs classification with prototype-based or classifier-head-based inference. The training script supports three modes: head_only, head_kd, and proto_only .

2. Project Structure
.
├── run_new_best.py      # main training / validation / test script
├── model.py             # HyperEdgeEncoder model
├── layers1.py           # hypergraph message passing layer
├── dataset_new.py       # dataset and subhypergraph sampling
├── utils.py             # graph loading and utility functions
├── eval_logger.py       # evaluation and confusion matrix plotting
├── environment.py       # environment checking script
├── requirements.txt     # pip dependencies
└── environment.yml      # conda environment file

3. Environment Requirements
Recommended environment:
Python 3.9 or 3.10
PyTorch 2.0+
NumPy 1.23+
scikit-learn 1.2+
matplotlib 3.6+
tqdm 4.64+

Install with pip:
pip install -r requirements.txt
Or create conda environment:
conda env create -f environment.yml
conda activate hypergraph-env

You can also check the runtime environment using:
python environment.py
The environment checking script reports OS, Python version, PyTorch version, CUDA availability, cuDNN version, and GPU information .

4. Input Data Format
The training / validation / test data are expected to be split-level hypergraph JSON files, for example:
data/train.json
data/val.json
data/test.json

Each JSON file should contain at least:
nodes: list of nodes
hyperedges: list of hyperedges

For each node, the loader uses fields such as:
node_id
token
type

For each hyperedge, the loader uses fields such as:
hyperedge_id
members
label
semantic_score
features.event_types
features.techniques
The loader builds node features, hyperedge features, and the incidence structure from these fields .

5. Training
Basic usage:
python run_new_best.py \
  --train_hg data/train.json \
  --val_hg data/val.json \
  --test_hg data/test.json

Common arguments include:
--mode head_kd
--emb_dim 256
--num_layers 2
--dropout 0.2
--epochs 50
--batch_size 128
--lr 3e-4
--weight_decay 1e-4
--k_hop 2
--max_edges 48
--max_nodes 192

The script supports the following modes:
head_only: classifier head only
head_kd: classifier head with prototype-based KD
proto_only: prototype-only inference
These modes are explicitly defined in the main argument parser .

6. Model and Sampling
Subhypergraph Sampling
For each target hyperedge, the dataset constructs a local k-hop subhypergraph and returns:
incidence tensor H
global node IDs
global hyperedge IDs
masks
target label
This logic is implemented in HyperedgeSubgraphDataset and collate_subgraph_ids .

Model
The core model is HyperEdgeEncoder, which:
encodes node features and hyperedge features,
applies multi-layer hypergraph message passing,
outputs the embedding of the center hyperedge for downstream classification .

7. Output Files
For each run, the script creates a timestamped output directory and saves:
args.json: all runtime arguments
exp_sig_full.txt: experiment signature
metrics_best.csv: training / validation metrics
prediction results and related logs
This saving logic is implemented in the main script and CSV logger .

8. Evaluation
The project supports common classification metrics, including:
Accuracy
Macro-Precision
Macro-Recall
Macro-F1
ROC-AUC
PR-AUC
Top-k accuracy
The evaluation helper in eval_logger.py also supports saving confusion matrices and JSON metric summaries .

9. Notes
Make sure that labels in validation and test splits are consistent with the train label space; otherwise the dataset loader will raise an error .
Hyperedge IDs in JSON files do not need to be continuous, because the code maintains a mapping between hyperedge_id and row index during loading .
GPU is recommended for faster training, but CPU execution is also supported.

10. Citation
If you use this code in your research, please cite the corresponding paper or project report.

## License
This project is released under the MIT License. You are free to use, copy, modify, and distribute this code, provided that the original copyright notice and license notice are retained. For more details, please refer to the `LICENSE` file in the root directory.