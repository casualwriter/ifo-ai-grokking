"""
IFO Grokking Experiment V4: Multi-Seed Robustness Test
Focus: Running 5 different seeds to check the statistical consistency of the H1 jump.

CK Hung & Echo, 2026/6/2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from ripser import ripser
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# Task: Modular Addition
# ============================================================

def generate_modular_addition_data(p=97, train_fraction=0.5, seed=42):
    # Set seed for data split to ensure reproducibility per run
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
# Topological Probe
# ============================================================

def get_max_h1_persistence(points, max_pca_components=3):
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    if points_scaled.shape[1] > max_pca_components:
        pca = PCA(n_components=max_pca_components)
        points_scaled = pca.fit_transform(points_scaled)
        
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
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data, val_data, _ = generate_modular_addition_data(p, train_fraction=0.5, seed=seed)
    model = ModularAdditionNet(p, hidden_dim=128)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
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
                embed_h1 = get_max_h1_persistence(embed_weights, max_pca_components=3)
                
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
    
    print(f"Starting Multi-Seed Experiment for {len(seeds)} seeds...")
    
    for seed in seeds:
        print(f"\n>>> Running Seed {seed}...")
        history = run_single_seed(seed, steps=4000, track_interval=100)
        all_results[seed] = history
        print(f"Seed {seed} Finished. Final Val Acc: {history['val_acc'][-1]:.3f} | Final H1: {history['embed_h1'][-1]:.4f}")

    # ============================================================
    # Plotting Combined Results
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    steps = all_results[seeds[0]]['step']
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Val Accuracies
    ax = axes[0]
    for i, seed in enumerate(seeds):
        ax.plot(steps, all_results[seed]['val_acc'], 
                color=colors[i], alpha=0.8, linewidth=2, label=f'Seed {seed}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Grokking Dynamics (Val Acc) across 5 Seeds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Embed H1 Persistence
    ax = axes[1]
    for i, seed in enumerate(seeds):
        ax.plot(steps, all_results[seed]['embed_h1'], 
                color=colors[i], alpha=0.8, linewidth=2, label=f'Seed {seed}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Max H1 Persistence (Embedding)')
    ax.set_title('Embedding Space: Circularity (H1) across 5 Seeds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ifo-grokking-v4-multiseed.png', dpi=150)
    print("\nMulti-seed plot saved to: ifo-grokking-v4-multiseed.png")
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nPlot window closed by user.")
