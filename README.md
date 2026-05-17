# EZTrainer

A flexible PyTorch training framework for classification, regression, and multi-task learning across tabular, image, and time-series datasets.

## Features

- **14+ built-in datasets** — MNIST, CIFAR-10, FairFace, credit risk, heart failure, insurance, UCI Adult, synthetic time series, and more
- **Automatic task detection** — classification vs regression, image vs tabular vs time-series, multi-task
- **Custom architecture** — AdaptiveBatchNorm, learnable gated skip connections, linear attention, recurrent blocks
- **Uncertainty-weighted loss** — per-sample heteroscedastic uncertainty estimation (Kendall & Gal 2017)
- **Proper BPTT** — gradient accumulation across full sequences with gradient clipping
- **LR scheduling** — `ReduceLROnPlateau` with patience
- **Regularization** — weight decay, optional dropout, focal loss for imbalanced classification
- **Visualization** — confusion matrices, ROC curves, t-SNE, weight distributions, misclassified examples
- **Vector database** — FAISS (short-term) + Weaviate (long-term) with RAG engine (experimental)

## Installation

```bash
pip install torch torchvision lightning pytorch-lightning
pip install scikit-learn imbalanced-learn matplotlib pandas numpy tqdm tensorboard
pip install faiss-cpu  # optional, for vector database
pip install weaviate-client  # optional, for long-term vector storage
```

## Usage

```bash
python main.py --dataset <name> [options]
```

### Examples

```bash
# Train on MNIST with default settings
python main.py --dataset mnist

# Binary classification with focal loss (imbalanced dataset)
python main.py --dataset credit_card_fraud --lr 5e-5

# Time-series regression with recurrent block
python main.py --dataset time_series --lr 1e-3 --epochs 20

# With regularization
python main.py --dataset heart_failure --dropout 0.2 --lr 1e-4 --seed 42

# Multi-task (FairFace: race + gender)
python main.py --dataset fairface --epochs 30 --dropout 0.1
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | Dataset name (`mnist`, `fairface`, `credit_card_fraud`, `heart_failure`, `insurance`, `time_series`, `air_quality`, `uci_adult`, `obesity`, `boston_house_price`, etc.) |
| `--epochs` | 50 | Number of training epochs |
| `--hidden_dim` | 512 | Hidden dimension (adjusted for images/time-series) |
| `--n_hidden` | 16 | Number of hidden layers |
| `--lr` | 1e-4 | Learning rate (10x for time-series) |
| `--reg_factor` | 1e-2 | Weight decay |
| `--batch_size` | 32 | Batch size |
| `--dropout` | 0.0 | Dropout probability |
| `--n_recurse` | 1 | Recursions per sample (forces recurrent behavior) |
| `--seed` | None | Random seed for reproducibility |

## Architecture

```
Input → [UnitBlock × N] → OutputHeads + LogVarHead
       ↕ (RecurrentBlock for time-series)

UnitBlock:
  AffineBlock (AdaptiveBatchNorm → Linear/Conv2d)
  → Dropout (optional)
  → SkipBlock (Activation → QuickSkip gated residual)

Network produces:
  - Output: per-task predictions (logits for classification, raw for regression)
  - Log variance: per-sample uncertainty estimate (used in loss weighting)
```

## Key Design Decisions

- **Everything is sequential** — all data is treated as a sequence of time steps. Non-temporal data has 1 step. This enables uniform handling of tabular, image, and time-series data.
- **Uncertainty weighting** — the loss function automatically down-weights high-uncertainty samples using the predicted log variance, making training robust to noisy/ambiguous examples.
- **Adaptive batch normalization** — tracks EMA statistics and compensates learned affine parameters when statistics shift, enabling stable training with small batches.
- **Gated skip connections** — `QuickSkip` uses a learnable scalar to balance between transformed and carried information, providing a lightweight residual mechanism.
