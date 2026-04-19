"""
The Self-Pruning Neural Network
================================
Case Study for Tredence Analytics — AI Engineering Internship 2025

This script implements a neural network that learns to prune itself during training.
Instead of post-training pruning, each weight is paired with a learnable "gate"
parameter that controls whether the weight remains active or gets pruned.

Architecture : CNN backbone + PrunableLinear classifier head
Dataset      : CIFAR-10 (10 classes, 32x32 RGB images)
Pruning Mech.: Learnable sigmoid gates with L1 sparsity regularization

Usage:
    python self_pruning_network.py                        # Run all 3 lambda experiments
    python self_pruning_network.py --lambdas 0.001        # Single lambda
    python self_pruning_network.py --epochs 50 --lr 0.001 # Custom config

Author : Mohit
Date   : April 2026
"""

import os
import sys
import json
import math
import argparse
import logging
import platform
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# ============================================================
# Configuration & Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# On Windows, DataLoader with num_workers > 0 can cause issues
NUM_WORKERS = 0 if platform.system() == "Windows" else 2


def get_device() -> torch.device:
    """Auto-detect the best available compute device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU (training will be slower)")
    return device


# ====================================================================
# PART 1: The PrunableLinear Layer (Custom nn.Module)
# ====================================================================
class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate parameters for self-pruning.

    Each weight w_ij has a corresponding gate score s_ij. During the forward pass:
      1. Gate values are computed:   g_ij = sigmoid(s_ij)  ∈ (0, 1)
      2. Pruned weights:             w'_ij = w_ij * g_ij
      3. Output:                     y = x @ w'.T + bias

    When a gate g_ij → 0, the corresponding weight is effectively pruned.
    The gate_scores are registered as nn.Parameter so they are updated
    by the optimizer alongside the regular weights.

    Why gradients flow correctly:
      - sigmoid() is differentiable everywhere
      - Element-wise multiplication is differentiable
      - F.linear is differentiable
      → Autograd can compute ∂Loss/∂weight AND ∂Loss/∂gate_scores

    Args:
        in_features  (int): Size of each input sample.
        out_features (int): Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # --- Standard weight and bias (same as nn.Linear) ---
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        # --- Gate scores: same shape as weight, also learnable ---
        # These will be passed through sigmoid to produce gate values ∈ (0, 1)
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # Initialize all parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize weights using Kaiming Uniform (same as nn.Linear default).
        Initialize gate_scores to 5.0 so that sigmoid(5.0) ≈ 0.993,
        meaning all gates start nearly "open" and the network begins fully connected.
        The optimizer will then learn which gates to close (prune).
        """
        # Kaiming initialization for weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias initialization (same as PyTorch's nn.Linear)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Gate scores initialized high → sigmoid(5) ≈ 0.993 → gates start "open"
        nn.init.constant_(self.gate_scores, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Steps:
          1. Compute gate values via sigmoid:  gates = σ(gate_scores)
          2. Element-wise multiplication:      pruned_weights = weight * gates
          3. Standard linear transformation:   output = x @ pruned_weights.T + bias
        """
        # Step 1: Transform gate_scores into (0, 1) range using sigmoid
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Apply gates to weights (element-wise masking)
        pruned_weights = self.weight * gates

        # Step 3: Standard linear operation using pruned weights
        return F.linear(x, pruned_weights, self.bias)

    def get_gate_values(self) -> torch.Tensor:
        """Return the current gate values (sigmoid of gate_scores) as a detached tensor."""
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores).detach().cpu()

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Calculate the percentage of gates below the threshold (effectively pruned).
        A gate value < 0.01 means the weight contributes < 1% of its value → pruned.
        """
        gate_values = self.get_gate_values()
        num_pruned = (gate_values < threshold).sum().item()
        total = gate_values.numel()
        return (num_pruned / total) * 100.0

    def extra_repr(self) -> str:
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, '
            f'gate_params={self.in_features * self.out_features}'
        )


# ====================================================================
# PART 2: Self-Pruning Network Architecture
# ====================================================================
class SelfPruningNetwork(nn.Module):
    """
    A CNN-based network for CIFAR-10 that prunes its own weights during training.

    Architecture:
      Feature Extractor (standard, not pruned):
        - Block 1: Conv2d(3→64)  + BatchNorm + ReLU + MaxPool  → 64×16×16
        - Block 2: Conv2d(64→128) + BatchNorm + ReLU + MaxPool  → 128×8×8
        - Block 3: Conv2d(128→256) + BatchNorm + ReLU + MaxPool → 256×4×4

      Classifier (self-pruning):
        - PrunableLinear(4096 → 512) + ReLU + Dropout(0.3)
        - PrunableLinear(512  → 256) + ReLU + Dropout(0.3)
        - PrunableLinear(256  → 10)

    Total prunable weights: 4096×512 + 512×256 + 256×10 = 2,230,784
    """

    def __init__(self):
        super(SelfPruningNetwork, self).__init__()

        # Feature extractor: standard CNN backbone (not pruned)
        self.features = nn.Sequential(
            # Block 1: 3×32×32 → 64×16×16
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64×16×16 → 128×8×8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 128×8×8 → 256×4×4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Classifier: PrunableLinear layers (self-pruning)
        self.classifier = nn.Sequential(
            PrunableLinear(256 * 4 * 4, 512),   # Layer 1: 4096 → 512
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),            # Layer 2: 512 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            PrunableLinear(256, 10),             # Layer 3: 256 → 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten: (batch, 256, 4, 4) → (batch, 4096)
        x = self.classifier(x)
        return x

    def get_prunable_layers(self) -> List[PrunableLinear]:
        """Return all PrunableLinear layers in the network."""
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        Compute the L1 sparsity regularization loss.

        L1_sparsity = Σ sigmoid(gate_scores)  across ALL PrunableLinear layers

        Since sigmoid outputs are always positive, the L1 norm simplifies to the
        straight sum of all gate values. Minimizing this pushes gates toward 0,
        effectively pruning the corresponding weights.
        """
        sparsity_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.get_prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            sparsity_loss = sparsity_loss + gates.sum()
        return sparsity_loss

    def get_overall_sparsity(self, threshold: float = 1e-2) -> float:
        """Calculate overall sparsity percentage across all prunable layers."""
        total_pruned = 0
        total_params = 0
        for layer in self.get_prunable_layers():
            gate_values = layer.get_gate_values()
            total_pruned += (gate_values < threshold).sum().item()
            total_params += gate_values.numel()
        return (total_pruned / total_params) * 100.0 if total_params > 0 else 0.0

    def get_all_gate_values(self) -> np.ndarray:
        """Collect all gate values from every PrunableLinear layer into a flat array."""
        all_gates = []
        for layer in self.get_prunable_layers():
            all_gates.append(layer.get_gate_values().numpy().flatten())
        return np.concatenate(all_gates)

    def permanently_prune(self, threshold: float = 1e-2):
        """
        Permanently zero-out weights whose gates are below the threshold.
        This can be used after training to create a truly sparse model for inference.
        """
        with torch.no_grad():
            for layer in self.get_prunable_layers():
                gates = torch.sigmoid(layer.gate_scores)
                mask = (gates >= threshold).float()
                layer.weight.data *= mask
                # Set pruned gate scores to a very negative value (sigmoid → ~0)
                layer.gate_scores.data[gates < threshold] = -20.0
        logger.info(f"Model permanently pruned at threshold={threshold}")


# ====================================================================
# PART 3: Data Loading (CIFAR-10)
# ====================================================================
def get_data_loaders(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare CIFAR-10 train and test data loaders with data augmentation.

    Training augmentation:
      - RandomHorizontalFlip (p=0.5): doubles effective dataset size
      - RandomCrop with 4px padding: translation invariance
      - Normalize per channel to zero mean and unit variance

    Test set: Only normalization applied (no augmentation).
    """
    # CIFAR-10 per-channel statistics (precomputed from training set)
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    logger.info(f"CIFAR-10 loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_loader, test_loader


# ====================================================================
# PART 4: Training & Evaluation Functions
# ====================================================================
def train_one_epoch(
    model: SelfPruningNetwork,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    lambda_val: float,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    """
    Train the model for one epoch.

    The total loss is computed as:
        Total Loss = CrossEntropyLoss + λ × SparsityLoss

    where SparsityLoss = Σ(all gate values) — the L1 norm of the sigmoid gates.

    Returns:
        Tuple of (avg_total_loss, avg_cls_loss, avg_sparsity_loss)
    """
    model.train()
    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:02d}", leave=False, file=sys.stdout)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Classification loss (Cross-Entropy)
        cls_loss = criterion(outputs, labels)

        # Sparsity regularization loss (L1 norm of all gate values)
        sparsity_loss = model.compute_sparsity_loss()

        # Combined loss: accuracy objective + pruning pressure
        total_loss = cls_loss + lambda_val * sparsity_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track metrics
        total_loss_sum += total_loss.item()
        cls_loss_sum += cls_loss.item()
        sparsity_loss_sum += sparsity_loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar with live stats
        pbar.set_postfix({
            'loss': f'{total_loss.item():.4f}',
            'acc': f'{100. * correct / total:.1f}%',
            'sparse': f'{model.get_overall_sparsity():.1f}%'
        })

    n_batches = len(loader)
    return (
        total_loss_sum / n_batches,
        cls_loss_sum / n_batches,
        sparsity_loss_sum / n_batches,
    )


@torch.no_grad()
def evaluate(
    model: SelfPruningNetwork,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model accuracy on a dataset. Returns accuracy as a percentage."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return 100.0 * correct / total


# ====================================================================
# PART 5: Visualization
# ====================================================================
def plot_gate_distribution(
    gate_values: np.ndarray,
    lambda_val: float,
    sparsity: float,
    accuracy: float,
    save_path: str,
):
    """
    Plot the distribution of gate values as a histogram.

    A successful self-pruning result will show:
      - A large spike near 0 (pruned weights → dead connections)
      - A smaller cluster of values away from 0 (important surviving connections)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Histogram with 100 bins across the [0, 1] range
    ax.hist(
        gate_values, bins=100, range=(0, 1),
        color='#2196F3', edgecolor='#1565C0', alpha=0.85
    )

    ax.set_xlabel('Gate Value', fontsize=13)
    ax.set_ylabel('Count', fontsize=13)
    ax.set_title(
        f'Gate Value Distribution  (λ = {lambda_val})\n'
        f'Sparsity: {sparsity:.1f}%   |   Test Accuracy: {accuracy:.2f}%',
        fontsize=14, fontweight='bold'
    )

    # Vertical line at the pruning threshold
    ax.axvline(
        x=0.01, color='#F44336', linestyle='--', linewidth=2,
        label='Pruning Threshold (0.01)'
    )
    ax.legend(fontsize=11, loc='upper center')

    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Plot saved: {save_path}")


def plot_comparison(results: List[Dict], save_path: str):
    """
    Bar chart comparing Test Accuracy vs Sparsity across different λ values.
    Visualizes the fundamental trade-off: more pruning → lower accuracy.
    """
    lambdas = [r['lambda'] for r in results]
    accuracies = [r['test_accuracy'] for r in results]
    sparsities = [r['sparsity'] for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(lambdas))
    width = 0.35

    # Accuracy bars (left axis, blue)
    ax1.bar(
        x - width / 2, accuracies, width,
        label='Test Accuracy (%)', color='#2196F3', alpha=0.85, edgecolor='white'
    )
    ax1.set_ylabel('Test Accuracy (%)', color='#2196F3', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='#2196F3')
    ax1.set_ylim(0, 100)

    # Sparsity bars (right axis, orange)
    ax2 = ax1.twinx()
    ax2.bar(
        x + width / 2, sparsities, width,
        label='Sparsity (%)', color='#FF5722', alpha=0.85, edgecolor='white'
    )
    ax2.set_ylabel('Sparsity (%)', color='#FF5722', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='#FF5722')
    ax2.set_ylim(0, 100)

    # Common formatting
    ax1.set_xlabel('Lambda (λ)', fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'λ = {l}' for l in lambdas])
    ax1.set_title(
        'Sparsity vs. Accuracy Trade-off Across Lambda Values',
        fontsize=14, fontweight='bold'
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Comparison plot saved: {save_path}")


def plot_training_curves(history: Dict, lambda_val: float, save_path: str):
    """Plot training loss and accuracy curves over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curve
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Total Loss')
    ax1.plot(epochs, history['cls_loss'], 'g--', linewidth=1.5, label='Classification Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss (λ={lambda_val})', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history['test_acc'], 'r-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'Test Accuracy (λ={lambda_val})', fontweight='bold')
    ax2.grid(alpha=0.3)

    # Sparsity curve
    ax3.plot(epochs, history['sparsity'], 'm-', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Sparsity (%)')
    ax3.set_title(f'Sparsity Over Training (λ={lambda_val})', fontweight='bold')
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Training curves saved: {save_path}")


# ====================================================================
# PART 6: Experiment Runner
# ====================================================================
def run_experiment(
    lambda_val: float,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> Dict:
    """
    Run a complete training experiment for a given lambda value.

    Pipeline:
      1. Load CIFAR-10 data with augmentation
      2. Initialize SelfPruningNetwork
      3. Train with combined loss (CrossEntropy + λ × L1_gates) for N epochs
      4. Evaluate final accuracy and sparsity
      5. Generate visualizations

    Returns:
        Dictionary with test accuracy, sparsity, gate values, and training history.
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  EXPERIMENT:  λ = {lambda_val}")
    logger.info(f"{'=' * 60}")

    # --- Data ---
    train_loader, test_loader = get_data_loaders(batch_size)

    # --- Model ---
    model = SelfPruningNetwork().to(device)

    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    prunable_weights = sum(l.weight.numel() for l in model.get_prunable_layers())
    gate_params = sum(l.gate_scores.numel() for l in model.get_prunable_layers())
    logger.info(f"  Total parameters:  {total_params:,}")
    logger.info(f"  Prunable weights:  {prunable_weights:,}")
    logger.info(f"  Gate parameters:   {gate_params:,}")

    # --- Optimizer, Scheduler, Loss ---
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # --- Training History ---
    history = {
        'train_loss': [], 'cls_loss': [], 'sparsity_loss': [],
        'test_acc': [], 'sparsity': []
    }

    # --- Training Loop ---
    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        total_loss, cls_loss, sp_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, lambda_val, device, epoch
        )

        test_acc = evaluate(model, test_loader, device)
        sparsity = model.get_overall_sparsity()

        scheduler.step()

        # Record history
        history['train_loss'].append(total_loss)
        history['cls_loss'].append(cls_loss)
        history['sparsity_loss'].append(sp_loss)
        history['test_acc'].append(test_acc)
        history['sparsity'].append(sparsity)

        if test_acc > best_accuracy:
            best_accuracy = test_acc

        # Log every 5 epochs and the final epoch
        if epoch % 5 == 0 or epoch == epochs:
            current_lr = scheduler.get_last_lr()[0]
            logger.info(
                f"  Epoch {epoch:02d}/{epochs} | "
                f"Loss: {total_loss:.4f} (cls: {cls_loss:.4f}, sp: {sp_loss:.1f}) | "
                f"Test Acc: {test_acc:.2f}% | "
                f"Sparsity: {sparsity:.1f}% | "
                f"LR: {current_lr:.6f}"
            )

    # --- Final Evaluation ---
    final_accuracy = evaluate(model, test_loader, device)
    final_sparsity = model.get_overall_sparsity()
    gate_values = model.get_all_gate_values()

    logger.info(f"\n  ┌──────────────────────────────────────┐")
    logger.info(f"  │  FINAL RESULTS for λ = {lambda_val:<14}│")
    logger.info(f"  ├──────────────────────────────────────┤")
    logger.info(f"  │  Test Accuracy:  {final_accuracy:>6.2f}%             │")
    logger.info(f"  │  Sparsity Level: {final_sparsity:>6.1f}%             │")
    logger.info(f"  │  Best Accuracy:  {best_accuracy:>6.2f}%             │")
    logger.info(f"  └──────────────────────────────────────┘")

    # Per-layer sparsity breakdown
    for i, layer in enumerate(model.get_prunable_layers()):
        logger.info(
            f"    Layer {i + 1} ({layer.in_features:>4d} → {layer.out_features:<4d}): "
            f"Sparsity = {layer.get_sparsity():.1f}%"
        )

    # --- Generate Plots ---
    # Gate distribution histogram
    plot_path = os.path.join(RESULTS_DIR, f"gate_distribution_lambda_{lambda_val}.png")
    plot_gate_distribution(gate_values, lambda_val, final_sparsity, final_accuracy, plot_path)

    # Training curves (loss, accuracy, sparsity over epochs)
    curves_path = os.path.join(RESULTS_DIR, f"training_curves_lambda_{lambda_val}.png")
    plot_training_curves(history, lambda_val, curves_path)

    return {
        'lambda': lambda_val,
        'test_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
        'sparsity': final_sparsity,
        'gate_values': gate_values,
        'total_params': total_params,
        'prunable_weights': prunable_weights,
        'history': history,
    }


# ====================================================================
# PART 7: Main Entry Point
# ====================================================================
def generate_results_table(results: List[Dict]) -> str:
    """Generate a Markdown results table for the report."""
    lines = [
        "| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) |",
        "|:----------:|:-----------------:|:------------------:|",
    ]
    for r in results:
        lines.append(
            f"| {r['lambda']:<10} | {r['test_accuracy']:>17.2f} | {r['sparsity']:>18.1f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='The Self-Pruning Neural Network — CIFAR-10 Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python self_pruning_network.py                          # Run all 3 experiments
  python self_pruning_network.py --lambdas 0.001          # Single experiment
  python self_pruning_network.py --epochs 50 --lr 0.0005  # Custom config
        """
    )
    parser.add_argument(
        '--lambdas', nargs='+', type=float,
        default=[1e-4, 1e-3, 1e-2],
        help='Lambda values for sparsity regularization (default: 0.0001 0.001 0.01)'
    )
    parser.add_argument(
        '--epochs', type=int, default=30,
        help='Number of training epochs per experiment (default: 30)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Training batch size (default: 128)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Adam learning rate (default: 0.001)'
    )

    args = parser.parse_args()
    device = get_device()

    # Print experiment configuration
    logger.info("=" * 60)
    logger.info("  THE SELF-PRUNING NEURAL NETWORK")
    logger.info("  CIFAR-10 Classification with Learned Sparsity")
    logger.info("=" * 60)
    logger.info(f"  Lambda values : {args.lambdas}")
    logger.info(f"  Epochs        : {args.epochs}")
    logger.info(f"  Batch size    : {args.batch_size}")
    logger.info(f"  Learning rate : {args.lr}")
    logger.info(f"  Device        : {device}")
    logger.info(f"  Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # === Run experiments for each lambda value ===
    all_results = []
    for lambda_val in args.lambdas:
        result = run_experiment(
            lambda_val=lambda_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        all_results.append(result)

    # ============================================================
    # Summary
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("  EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  {'Lambda':<12} {'Test Acc (%)':<15} {'Sparsity (%)':<15}")
    logger.info(f"  {'-' * 42}")
    for r in all_results:
        logger.info(
            f"  {r['lambda']:<12} {r['test_accuracy']:<15.2f} {r['sparsity']:<15.1f}"
        )

    # Comparison plot
    comparison_path = os.path.join(RESULTS_DIR, "sparsity_vs_accuracy.png")
    plot_comparison(all_results, comparison_path)

    # Save results to JSON (without numpy arrays)
    results_json = [{
        'lambda': r['lambda'],
        'test_accuracy': round(r['test_accuracy'], 2),
        'best_accuracy': round(r['best_accuracy'], 2),
        'sparsity': round(r['sparsity'], 1),
        'total_params': r['total_params'],
        'prunable_weights': r['prunable_weights'],
    } for r in all_results]

    json_path = os.path.join(RESULTS_DIR, "experiment_results.json")
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    # Generate markdown results table
    results_table = generate_results_table(all_results)
    table_path = os.path.join(RESULTS_DIR, "results_table.md")
    with open(table_path, 'w') as f:
        f.write("# Experiment Results\n\n")
        f.write(results_table)
        f.write("\n")

    logger.info(f"\n  Results JSON  : {json_path}")
    logger.info(f"  Results Table : {table_path}")
    logger.info(f"\n  All experiments completed successfully!")
    logger.info(f"  Check the '{RESULTS_DIR}/' directory for plots and results.\n")


if __name__ == "__main__":
    main()
