"""
IFO Grokking Experiment V3: Control Group (No Weight Decay)
Focus: Testing if H1 remains flat when Grokking is suppressed (WD = 0.0).

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

def generate_modular_addition_data(p=97, train_fraction=0.5):
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
# Refined Topological Probe
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
# Training & Tracking Loop
# ============================================================

def train_and_track(model, train_data, val_data, p, 
                    steps=8000, lr=1e-3, weight_decay=0.0, # CONTROL: NO WEIGHT DECAY
                    track_interval=100):
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    history = {
        'step': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'embed_h1_persistence': [],
        'act_h1_persistence': []
    }
    
    print(f"Starting Control V3 Training (Weight Decay = {weight_decay})...")
    for step in tqdm(range(steps)):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        if step % track_interval == 0:
            model.eval()
            with torch.no_grad():
                train_logits, train_h = model(X_train)
                train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()
                
                val_logits, val_h = model(X_val)
                val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
                
                # Probe 1: Embedding Space
                embed_weights = model.embed.weight.cpu().numpy()
                embed_h1 = get_max_h1_persistence(embed_weights, max_pca_components=3)
                
                # Probe 2: Activation Space
                all_h = torch.cat([train_h, val_h], dim=0).cpu().numpy()
                subsample_idx = np.random.choice(len(all_h), size=400, replace=False)
                act_subsampled = all_h[subsample_idx]
                act_h1 = get_max_h1_persistence(act_subsampled, max_pca_components=3)
                
                history['step'].append(step)
                history['train_loss'].append(loss.item())
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['embed_h1_persistence'].append(embed_h1)
                history['act_h1_persistence'].append(act_h1)
                
                print(f"Step {step:4d} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f} | "
                      f"Embed H1: {embed_h1:.4f} | Act H1: {act_h1:.4f}")
                
                # No early stop here to see if it ever groks or if H1 stays flat
                    
    return history

# ============================================================
# Plotting
# ============================================================

def plot_results(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    steps = history['step']
    
    # 1. Accuracy
    ax = axes[0, 0]
    ax.plot(steps, history['train_acc'], label='Train Acc', color='blue', alpha=0.8)
    ax.plot(steps, history['val_acc'], label='Val Acc', color='orange', alpha=0.8)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Accuracy')
    ax.set_title('Grokking Dynamics (Accuracy) - CONTROL (WD=0)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Loss
    ax = axes[0, 1]
    ax.plot(steps, history['train_loss'], color='purple')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss - CONTROL (WD=0)')
    ax.grid(True, alpha=0.3)
    
    # 3. Embedding H1 Persistence
    ax = axes[1, 0]
    ax.plot(steps, history['embed_h1_persistence'], color='red', marker='o', markersize=3)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Max H1 Persistence (Embedding)')
    ax.set_title('Embedding Space: Circularity (CONTROL)')
    ax.grid(True, alpha=0.3)
    
    # 4. Activation H1 Persistence
    ax = axes[1, 1]
    ax.plot(steps, history['act_h1_persistence'], color='green', marker='o', markersize=3)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Max H1 Persistence (Activation)')
    ax.set_title('Activation Space: Circularity (CONTROL)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ifo-grokking-v3-control-results.png', dpi=150)
    print("\nControl plot saved to: ifo-grokking-v3-control-results.png")
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nPlot window closed by user.")

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    p = 97
    train_data, val_data, _ = generate_modular_addition_data(p, train_fraction=0.5)
    
    model = ModularAdditionNet(p, hidden_dim=128)
    
    # Run same 8000 steps but with WD=0.0
    history = train_and_track(
        model, train_data, val_data, p,
        steps=8000,
        lr=1e-3,
        weight_decay=0.0, # CONTROL
        track_interval=100
    )
    
    plot_results(history)
