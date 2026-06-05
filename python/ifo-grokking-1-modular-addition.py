"""
IFO Grokking Experiment: Topological Phase Transition
Task: Modular Addition (classic grokking task)
Hypothesis: Betti numbers change dramatically at grokking moment

CK Hung, 2026/6/2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from ripser import ripser
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# Task: Modular Addition
# ============================================================

def generate_modular_addition_data(p=97, train_fraction=0.5):
    """
    Generate (a + b) mod p dataset
    Classic grokking task from Power et al. 2022
    """
    all_pairs = [(i, j) for i in range(p) for j in range(p)]
    np.random.shuffle(all_pairs)
    
    train_size = int(len(all_pairs) * train_fraction)
    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size:]
    
    def to_tensors(pairs):
        X = torch.tensor([[a, b] for a, b in pairs], dtype=torch.long)
        y = torch.tensor([(a + b) % p for a, b in pairs], dtype=torch.long)
        return X, y
    
    return to_tensors(train_pairs), to_tensors(val_pairs), p

# ============================================================
# Model
# ============================================================

class ModularAdditionNet(nn.Module):
    def __init__(self, p, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(p, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, p)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, 2) - two numbers to add
        emb_a = self.embed(x[:, 0])  # (batch, hidden)
        emb_b = self.embed(x[:, 1])  # (batch, hidden)
        h = torch.cat([emb_a, emb_b], dim=1)  # (batch, hidden*2)
        h = self.relu(self.fc1(h))  # (batch, hidden) - THIS IS THE ACTIVATION WE TRACK
        out = self.fc2(h)  # (batch, p)
        return out, h  # return both output and hidden activation

# ============================================================
# Topological Data Analysis
# ============================================================

def compute_betti_numbers(activations, max_dim=1, max_edge_length=10.0):
    """
    Compute Betti-0 and Betti-1 using persistent homology
    
    activations: (n_samples, dim) numpy array
    Returns: (betti_0, betti_1) - counts of connected components and loops
    """
    # Use PCA to reduce to 3D for computational efficiency
    if activations.shape[1] > 3:
        pca = PCA(n_components=3)
        activations = pca.fit_transform(activations)
    
    # Run persistent homology
    result = ripser(activations, maxdim=max_dim, thresh=max_edge_length)
    diagrams = result['dgms']
    
    # Count features that persist significantly
    # Betti-0: connected components (dim 0)
    # Betti-1: loops (dim 1)
    persistence_threshold = 0.1  # Only count features with persistence > threshold
    
    betti_0 = np.sum(diagrams[0][:, 1] - diagrams[0][:, 0] > persistence_threshold)
    betti_1 = 0
    if len(diagrams) > 1:
        # Filter out infinite bars
        finite_bars = diagrams[1][diagrams[1][:, 1] != np.inf]
        if len(finite_bars) > 0:
            betti_1 = np.sum(finite_bars[:, 1] - finite_bars[:, 0] > persistence_threshold)
    
    return betti_0, betti_1

# ============================================================
# Training Loop with Topology Tracking
# ============================================================

def train_and_track(model, train_data, val_data, p, 
                    steps=10000, lr=1e-3, weight_decay=1.0,
                    track_interval=50):
    """
    Train model and track topology at intervals
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # History
    history = {
        'step': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'betti_0': [],
        'betti_1': []
    }
    
    print("Starting training...")
    for step in tqdm(range(steps)):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        if step % track_interval == 0:
            model.eval()
            with torch.no_grad():
                # Train accuracy
                train_logits, train_h = model(X_train)
                train_pred = train_logits.argmax(dim=1)
                train_acc = (train_pred == y_train).float().mean().item()
                
                # Validation accuracy
                val_logits, _ = model(X_val)
                val_pred = val_logits.argmax(dim=1)
                val_acc = (val_pred == y_val).float().mean().item()
                
                # Topology: use ALL data (train + val) to see full activation space structure
                X_all = torch.cat([X_train, X_val], dim=0)
                _, h_all = model(X_all)
                activations = h_all.cpu().numpy()
                
                # Compute Betti numbers
                betti_0, betti_1 = compute_betti_numbers(activations)
                
                # Record
                history['step'].append(step)
                history['train_loss'].append(loss.item())
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['betti_0'].append(betti_0)
                history['betti_1'].append(betti_1)
                
                print(f"Step {step}: train_acc={train_acc:.3f}, val_acc={val_acc:.3f}, "
                      f"β0={betti_0}, β1={betti_1}")
    
    return history

# ============================================================
# Visualization
# ============================================================

def plot_results(history):
    """Plot training dynamics and topology evolution"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    steps = history['step']
    
    # Top-left: Accuracy
    ax = axes[0, 0]
    ax.plot(steps, history['train_acc'], label='Train Acc', alpha=0.7)
    ax.plot(steps, history['val_acc'], label='Val Acc', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Grokking: Accuracy over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top-right: Loss
    ax = axes[0, 1]
    ax.plot(steps, history['train_loss'], alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Train Loss')
    ax.set_title('Training Loss')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Betti-0 (connected components)
    ax = axes[1, 0]
    ax.plot(steps, history['betti_0'], color='blue', marker='o', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Betti-0 (Connected Components)')
    ax.set_title('Topology: β0 Evolution')
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Betti-1 (loops)
    ax = axes[1, 1]
    ax.plot(steps, history['betti_1'], color='red', marker='o', markersize=3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Betti-1 (Loops)')
    ax.set_title('Topology: β1 Evolution (IFO Prediction)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ifo-grokking-results.png', dpi=150)
    print("\nPlot saved to: ifo-grokking-results.png")
    plt.show()

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IFO Grokking Experiment")
    print("Hypothesis: Betti-1 should spike at grokking moment")
    print("=" * 60)
    
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    
    p = 97  # prime for modular addition
    train_data, val_data, _ = generate_modular_addition_data(p, train_fraction=0.5)
    
    print(f"\nTask: ({p} + {p}) mod {p}")
    print(f"Train samples: {len(train_data[0])}")
    print(f"Val samples: {len(val_data[0])}")
    
    # Model
    model = ModularAdditionNet(p, hidden_dim=128)
    
    # Train and track
    history = train_and_track(
        model, train_data, val_data, p,
        steps=10000,
        lr=1e-3,
        weight_decay=1.0,  # Strong regularization to induce grokking
        track_interval=100
    )
    
    # Visualize
    plot_results(history)
    
    print("\n" + "=" * 60)
    print("Experiment complete.")
    print("Check the plot for correlation between val_acc jump and β1 spike.")
    print("=" * 60)
