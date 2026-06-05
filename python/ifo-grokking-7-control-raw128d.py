
"""
IFO Grokking Experiment V7: Raw 128D Control Group (WD=0.0)
Focus: Running 5 seeds with NO weight decay to establish the baseline.

CK Hung & Echo, 2026/6/2
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ripser")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from ripser import ripser
import matplotlib.pyplot as plt

# ============================================================
# Task: Modular Addition
# ============================================================

def generate_modular_addition_data(p=97, train_fraction=0.5, seed=42):
    np.random.seed(seed)
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
        emb_a = self.embed(x[:, 0])
        emb_b = self.embed(x[:, 1])
        h = torch.cat([emb_a, emb_b], dim=1)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out, h

# ============================================================
# Raw High-Dimensional Topological Probe (No PCA)
# ============================================================

def get_max_h1_persistence_raw(points):
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    result = ripser(points_scaled, maxdim=1)
    dgms = result['dgms']
    
    max_persistence = 0.0
    if len(dgms) > 1 and len(dgms[1]) > 0:
        finite_h1 = dgms[1][dgms[1][:, 1] != np.inf]
        if len(finite_h1) > 0:
            persistences = finite_h1[:, 1] - finite_h1[:, 0]
            max_persistence = np.max(persistences)
            
    return max_persistence

# ============================================================
# Single Seed Run
# ============================================================

def run_single_seed(seed, p=97, steps=4000, track_interval=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data, val_data, _ = generate_modular_addition_data(p, train_fraction=0.5, seed=seed)
    model = ModularAdditionNet(p, hidden_dim=128)
    
    # CONTROL SETTING: weight_decay = 0.0
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    history = {
        'step': [],
        'val_acc': [],
        'embed_h1': []
    }
    
    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        if step % track_interval == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_val)
                val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
                
                embed_weights = model.embed.weight.cpu().numpy()
                embed_h1 = get_max_h1_persistence_raw(embed_weights)
                
                history['step'].append(step)
                history['val_acc'].append(val_acc)
                history['embed_h1'].append(embed_h1)
                
    return history

# ============================================================
# Main Multi-Seed Loop
# ============================================================

if __name__ == "__main__":
    seeds = [42, 100, 2026, 7, 999]
    all_results = {}
    
    print("Starting Raw 128D Control Experiment (WD=0.0)...")
    
    for seed in seeds:
        print(f"\n>>> Running Control Seed {seed}...")
        history = run_single_seed(seed, steps=4000, track_interval=100)
        all_results[seed] = history
        print(f"Seed {seed} Finished. Final Val Acc: {history['val_acc'][-1]:.3f} | Final Raw H1: {history['embed_h1'][-1]:.4f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    steps = all_results[seeds[0]]['step']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    ax = axes[0]
    for i, seed in enumerate(seeds):
        ax.plot(steps, all_results[seed]['val_acc'], color=colors[i], alpha=0.8, linewidth=2, label=f'Seed {seed}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Control (WD=0): Val Acc (No Grokking)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for i, seed in enumerate(seeds):
        ax.plot(steps, all_results[seed]['embed_h1'], color=colors[i], alpha=0.8, linewidth=2, label=f'Seed {seed}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Max H1 Persistence')
    ax.set_title('Control (WD=0): Raw 128D H1 (No Topology)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ifo-grokking-v7-control-raw128d.png', dpi=150)
    print("\nControl plot saved to: ifo-grokking-v7-control-raw128d.png")
    plt.show()
