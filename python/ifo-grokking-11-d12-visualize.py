
"""
IFO Grokking Experiment V11: Embedding Geometry Visualization (D12)
Focus: Extracting 128D embeddings from Seed 7 (Crystallized) and Seed 100 (Uncrystallized),
       projecting to 2D via PCA, and drawing the underlying Cayley rings.

CK Hung & Echo, 2026/6/2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ============================================================
# D12 Setup
# ============================================================

def get_d12_multiplication_table():
    table = np.zeros((24, 24), dtype=np.int64)
    for i in range(24):
        s1 = i // 12
        r1 = i % 12
        for j in range(24):
            s2 = j // 12
            r2 = j % 12
            if s1 == 0 and s2 == 0:
                out_s = 0; out_r = (r1 + r2) % 12
            elif s1 == 0 and s2 == 1:
                out_s = 1; out_r = (-r1 + r2) % 12
            elif s1 == 1 and s2 == 0:
                out_s = 1; out_r = (r1 + r2) % 12
            else:
                out_s = 0; out_r = (-r1 + r2) % 12
            table[i, j] = out_s * 12 + out_r
    return table

def generate_d12_data(train_fraction=0.7, seed=42):
    np.random.seed(seed)
    table = get_d12_multiplication_table()
    all_pairs = [(i, j) for i in range(24) for j in range(24)]
    np.random.shuffle(all_pairs)
    train_size = int(len(all_pairs) * train_fraction)
    train_pairs = all_pairs[:train_size]
    val_pairs = all_pairs[train_size:]
    
    def to_tensors(pairs):
        X = torch.tensor([[a, b] for a, b in pairs], dtype=torch.long)
        y = torch.tensor([table[a, b] for a, b in pairs], dtype=torch.long)
        return X, y
    return to_tensors(train_pairs), to_tensors(val_pairs)

class DihedralNet(nn.Module):
    def __init__(self, num_elements=24, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(num_elements, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_elements)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        emb_a = self.embed(x[:, 0])
        emb_b = self.embed(x[:, 1])
        h = torch.cat([emb_a, emb_b], dim=1)
        h = self.relu(self.fc1(h))
        out = self.fc2(h)
        return out, h

# ============================================================
# Training Function (Fast, no intermediate TDA)
# ============================================================

def train_and_get_embeddings(seed, steps=40000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data, val_data = generate_d12_data(train_fraction=0.7, seed=seed)
    model = DihedralNet(num_elements=24, hidden_dim=128)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    print(f"Training Seed {seed} for {steps} steps...")
    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(X_val)
        val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
    
    
    embed_weights = model.embed.weight.detach().cpu().numpy()
    return embed_weights, val_acc

# ============================================================
# Plotting Helper
# ============================================================

def plot_embedding_pca(ax, embed, title, val_acc):
    # Standardize and PCA to 2D
    scaler = StandardScaler()
    embed_scaled = scaler.fit_transform(embed)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embed_scaled)
    
    # Split into Rotations (0-11) and Reflections (12-23)
    rot_coords = coords[0:12]
    ref_coords = coords[12:24]
    
    # Plot Rotations Ring (Blue)
    ax.plot(np.append(rot_coords[:, 0], rot_coords[0, 0]), 
            np.append(rot_coords[:, 1], rot_coords[0, 1]), 
            color='#1f77b4', linestyle='-', linewidth=1.5, alpha=0.6, label='Rotations (0-11) Ring')
    ax.scatter(rot_coords[:, 0], rot_coords[:, 1], color='#1f77b4', marker='o', s=80, edgecolors='black', zorder=3)
    
    # Plot Reflections Ring (Orange)
    ax.plot(np.append(ref_coords[:, 0], ref_coords[0, 0]), 
            np.append(ref_coords[:, 1], ref_coords[0, 1]), 
            color='#ff7f0e', linestyle='-', linewidth=1.5, alpha=0.6, label='Reflections (12-23) Ring')
    ax.scatter(ref_coords[:, 0], ref_coords[:, 1], color='#ff7f0e', marker='X', s=100, edgecolors='black', zorder=3)
    
    # Annotate indices
    for i in range(24):
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]), 
                    textcoords="offset points", xytext=(0,6), ha='center', fontweight='bold', fontsize=9)
        
    ax.set_title(f"{title}\n(Val Acc: {val_acc:.1%})", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', 'datalim')
    ax.legend(loc='upper right', fontsize=8)

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Starting D12 Embedding Geometry Analysis...")
    
    # 1. Run Seed 7 (Crystallized)
    embed_7, acc_7 = train_and_get_embeddings(seed=7, steps=40000)
    
    # 2. Run Seed 100 (Uncrystallized)
    embed_100, acc_100 = train_and_get_embeddings(seed=100, steps=40000)
    
    # Plot Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    
    plot_embedding_pca(axes[0], embed_7, "Seed 7: Crystallized Representation", acc_7)
    plot_embedding_pca(axes[1], embed_100, "Seed 100: Alternative Representation", acc_100)
    
    plt.suptitle("D12 Group Embedding Space (PCA 2D Projection at Step 40,000)\nComparing Crystallized vs. Alternative Representations", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_img = 'ifo-grokking-v11-d12-geometry.png'
    plt.savefig(output_img, dpi=150)
    print(f"\nGeometry plot saved to: {output_img}")
    plt.show()
