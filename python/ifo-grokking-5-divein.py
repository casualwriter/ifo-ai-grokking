
"""
IFO Grokking Experiment V5: Deep Dive into Seed 2026
Focus: Visualizing the 3D Embedding manifold at Step 1800 vs Step 3000.

CK Hung & Echo, 2026/6/2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Task & Model Setup (Same as before)
# ============================================================

def generate_modular_addition_data(p=97, train_fraction=0.5, seed=2026):
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
# Helper: Project and Plot 3D Manifold
# ============================================================

def get_3d_projection(weights):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(weights)
    pca = PCA(n_components=3)
    projected = pca.fit_transform(scaled)
    # Return projected coordinates and explained variance ratio
    return projected, pca.explained_variance_ratio_

def plot_manifold_3d(ax, coords, var_ratio, title):
    p = len(coords)
    # Use HSV colormap to show the periodic nature of modular arithmetic
    colors = plt.cm.hsv(np.linspace(0, 1, p))
    
    # Scatter plot of tokens
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
               c=colors, s=50, edgecolor='k', alpha=0.8)
    
    # Draw lines connecting consecutive tokens (0 -> 1 -> ... -> 96 -> 0)
    # This reveals the actual topology of the learned sequence
    for i in range(p):
        next_i = (i + 1) % p
        ax.plot([coords[i, 0], coords[next_i, 0]],
                [coords[i, 1], coords[next_i, 1]],
                [coords[i, 2], coords[next_i, 2]],
                color='gray', alpha=0.5, linewidth=1.5)
        
    # Label a few points for reference
    for label in [0, 25, 50, 75]:
        ax.text(coords[label, 0], coords[label, 1], coords[label, 2], 
                str(label), color='black', fontsize=12, weight='bold')
        
    ax.set_title(f"{title}\nPCA Var: {var_ratio[0]:.2f}, {var_ratio[1]:.2f}, {var_ratio[2]:.2f}")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

# ============================================================
# Execution
# ============================================================

if __name__ == "__main__":
    seed = 2026
    p = 97
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data, val_data, _ = generate_modular_addition_data(p, train_fraction=0.5, seed=seed)
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    model = ModularAdditionNet(p, hidden_dim=128)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
    criterion = nn.CrossEntropyLoss()
    
    saved_weights = {}
    target_steps = [1800, 3000]
    
    print("Training Seed 2026 to capture manifold states...")
    for step in range(3001):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
        if step in target_steps:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_val)
                val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
                weights = model.embed.weight.cpu().numpy().copy()
                saved_weights[step] = (weights, val_acc)
                print(f"Captured Step {step:4d} | Val Acc: {val_acc:.3f}")

    # ============================================================
    # Plotting 3D Comparison
    # ============================================================
    fig = plt.figure(figsize=(15, 7))
    
    # Step 1800 Plot
    ax1 = fig.add_subplot(121, projection='3d')
    w_1800, acc_1800 = saved_weights[1800]
    coords_1800, var_1800 = get_3d_projection(w_1800)
    plot_manifold_3d(ax1, coords_1800, var_1800, f"Step 1800 (Val Acc: {acc_1800:.3f})")
    
    # Step 3000 Plot
    ax2 = fig.add_subplot(122, projection='3d')
    w_3000, acc_3000 = saved_weights[3000]
    coords_3000, var_3000 = get_3d_projection(w_3000)
    plot_manifold_3d(ax2, coords_3000, var_3000, f"Step 3000 (Val Acc: {acc_3000:.3f})")
    
    plt.suptitle("Seed 2026: Manifold Evolution (High-Dim Twist vs Clean Crystallization)", fontsize=16)
    plt.tight_layout()
    plt.savefig('seed_2026_manifold_dive.png', dpi=150)
    print("\nDiagnostic plot saved to: seed_2026_manifold_dive.png")
    plt.show()
