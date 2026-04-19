# 🧠 The Self-Pruning Neural Network

> **Case Study — Tredence Analytics AI Engineering Internship 2025**

A neural network that **learns to prune itself** during training. Instead of manually removing weights after training, each weight has a learnable "gate" parameter that the optimizer uses to decide what stays and what goes.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Results](#results)
- [Report](#report)

---

## Overview

Traditional neural network pruning is a **two-step process**: train the full model, then remove unimportant weights. This project implements **self-pruning** — the network identifies and removes its own weakest connections *during* training by learning gate parameters alongside the weights.

**Key Idea:** Each weight `w_ij` is multiplied by a gate `g_ij = sigmoid(s_ij)`. An L1 penalty on the gates encourages them to become exactly zero, effectively pruning the corresponding weights.

---

## How It Works

### 1. PrunableLinear Layer (Custom `nn.Module`)
```
weight:      [out_features × in_features]  — standard learnable weights
gate_scores: [out_features × in_features]  — learnable pruning gates

Forward Pass:
  gates = sigmoid(gate_scores)        # values in (0, 1)
  pruned_weights = weight × gates     # element-wise masking
  output = x @ pruned_weights.T + bias
```

### 2. Loss Function
```
Total Loss = CrossEntropy(predictions, labels) + λ × Σ(all gate values)
```
- **λ = 0** → No pruning (standard training)
- **λ → ∞** → Maximum pruning (all gates forced to 0)

### 3. Self-Pruning in Action
During backpropagation, the optimizer updates both weights and gate scores. The L1 penalty pushes unimportant gates toward zero while important gates resist (because removing them would increase classification loss).

---

## Project Structure

```
tendrance/
├── self_pruning_network.py   # Main script (all implementations)
├── report.md                 # Analysis report (theory + results)
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── data/                     # CIFAR-10 (auto-downloaded)
└── results/                  # Generated outputs
    ├── gate_distribution_lambda_*.png
    ├── training_curves_lambda_*.png
    ├── sparsity_vs_accuracy.png
    ├── experiment_results.json
    └── results_table.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/self-pruning-network.git
cd self-pruning-network

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows

# Install requirements
pip install -r requirements.txt
```

---

## Usage

### Run All Experiments (Default: λ = 0.0001, 0.001, 0.01)
```bash
python self_pruning_network.py
```

### Run a Single Experiment
```bash
python self_pruning_network.py --lambdas 0.001
```

### Custom Configuration
```bash
python self_pruning_network.py \
    --lambdas 0.0001 0.001 0.01 \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.0005
```

### Command-Line Arguments

| Argument       | Default              | Description                          |
|:---------------|:---------------------|:-------------------------------------|
| `--lambdas`    | `0.0001 0.001 0.01`  | Sparsity regularization strengths    |
| `--epochs`     | `30`                 | Training epochs per experiment       |
| `--batch-size` | `128`                | Mini-batch size                      |
| `--lr`         | `0.001`              | Adam optimizer learning rate         |

---

## Results

After running the script, check the `results/` directory for:

1. **`experiment_results.json`** — Numeric results for all experiments
2. **`results_table.md`** — Markdown table ready for the report
3. **`gate_distribution_*.png`** — Histograms showing the bimodal gate distribution
4. **`training_curves_*.png`** — Loss, accuracy, and sparsity over epochs
5. **`sparsity_vs_accuracy.png`** — Comparison chart across all λ values

---

## Report

See [`report.md`](report.md) for the detailed analysis including:
- Why L1 penalty on sigmoid gates encourages sparsity (theoretical explanation)
- Results table for different λ values
- Gate distribution analysis
- Observations and potential extensions

---

## Technical Highlights

- **Custom autograd-compatible layer** — `PrunableLinear` with correct gradient flow through sigmoid gates
- **Device-agnostic** — Automatically detects CUDA, MPS (Apple Silicon), or CPU
- **Production-quality code** — Type hints, logging, argparse, comprehensive documentation
- **Automatic visualization** — All plots generated and saved during training
- **Windows-compatible** — DataLoader `num_workers` auto-configured for the platform

---

## License

This project is submitted as a case study for the Tredence Analytics AI Engineering Internship.

---

*Built with PyTorch 🔥*
