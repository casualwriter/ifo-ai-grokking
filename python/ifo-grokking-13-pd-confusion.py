
"""
IFO Grokking Experiment V13: Full PD & Error Pattern Analysis
1. Plots complete Persistence Diagrams (H0 & H1) for Seed 7 & 100.
2. Plots 24x24 Error Grids to pinpoint where generalization fails.

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
import matplotlib.colors as mcolors

# ============================================================
# D12 Setup
# ============================================================

def get_d12_multiplication_table():
    table = np.zeros((24, 24), dtype=np.int64)
    for i in range(24):
        s1 = i // 12; r1 = i % 12
        for j in range(24):
            s2 = j // 12; r2 = j % 12
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
    return to_tensors(train_pairs), to_tensors(val_pairs), train_pairs, val_pairs

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
# Analysis Helpers
# ============================================================

def train_and_analyze(seed, steps=40000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_tensors, val_tensors, train_pairs, val_pairs = generate_d12_data(train_fraction=0.7, seed=seed)
    model = DihedralNet(num_elements=24, hidden_dim=128)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_tensors
    X_val, y_val = val_tensors
    
    print(f"Training Seed {seed}...")
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
        preds = val_logits.argmax(dim=1)
        corrects = (preds == y_val).cpu().numpy()
        
    embed_weights = model.embed.weight.detach().cpu().numpy()
    
    # Run full TDA
    scaler = StandardScaler()
    embed_scaled = scaler.fit_transform(embed_weights)
    tda_res = ripser(embed_scaled, maxdim=1)
    
    # Build Error Grid
    # 0: Train (Gray), 1: Val Correct (White), 2: Val Incorrect (Red)
    grid = np.zeros((24, 24))
    for (a, b) in train_pairs:
        grid[a, b] = 0
    for idx, (a, b) in enumerate(val_pairs):
        grid[a, b] = 1 if corrects[idx] else 2
        
    return tda_res['dgms'], grid

def plot_pd(ax, dgms, title):
    h0 = dgms[0]
    h1 = dgms[1] if len(dgms) > 1 else np.array([])
    
    # Plot H0 (excluding infinity for scale)
    h0_finite = h0[h0[:, 1] != np.inf]
    ax.scatter(h0_finite[:, 0], h0_finite[:, 1], color='red', marker='o', label='H0 (0D)', alpha=0.6)
    
    # Plot H1
    if len(h1) > 0:
        ax.scatter(h1[:, 0], h1[:, 1], color='blue', marker='^', s=50, label='H1 (1D)', alpha=0.8)
        
    # Draw diagonal line
    max_val = max(np.max(h0_finite), np.max(h1) if len(h1) > 0 else 1.0) * 1.1
    ax.plot([0, max_val], [0, max_val], color='gray', linestyle='--')
    
    ax.set_xlim(-0.1, max_val)
    ax.set_ylim(-0.1, max_val)
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(f"{title}: Persistence Diagram")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.2)

def plot_error_grid(ax, grid, title):
    cmap = mcolors.ListedColormap(['#d3d3d3', '#ffffff', '#ff4d4d']) # Gray, White, Red
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    im = ax.imshow(grid, cmap=cmap, norm=norm, origin='upper')
    
    # Draw quadrant lines to separate Rotations (0-11) and Reflections (12-23)
    ax.axvline(11.5, color='black', linewidth=1.5, linestyle=':')
    ax.axhline(11.5, color='black', linewidth=1.5, linestyle=':')
    
    ax.set_xticks(range(24))
    ax.set_yticks(range(24))
    ax.set_xticklabels(range(24), fontsize=7)
    ax.set_yticklabels(range(24), fontsize=7)
    
    ax.set_xlabel('Element B')
    ax.set_ylabel('Element A')
    ax.set_title(f"{title}: Error Grid (A * B)\nRed = Error, Gray = Train, White = Correct")
    
    # Add labels for quadrants
    ax.text(5, -1.5, "Rot", ha='center', fontweight='bold', color='blue')
    ax.text(17, -1.5, "Ref", ha='center', fontweight='bold', color='orange')
    ax.text(-1.5, 5, "Rot", va='center', fontweight='bold', color='blue', rotation=90)
    ax.text(-1.5, 17, "Ref", va='center', fontweight='bold', color='orange', rotation=90)

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Starting D12 Deep Analysis (PD & Error Pattern)...")
    
    # Run both seeds
    dgms_7, grid_7 = train_and_analyze(seed=7)
    dgms_100, grid_100 = train_and_analyze(seed=100)
    
    # Plotting 2x2 Layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    
    # Row 1: Seed 7
    plot_pd(axes[0, 0], dgms_7, "Seed 7 (Crystallized)")
    plot_error_grid(axes[0, 1], grid_7, "Seed 7 (Crystallized)")
    
    # Row 2: Seed 100
    plot_pd(axes[1, 0], dgms_100, "Seed 100 (Alternative)")
    plot_error_grid(axes[1, 1], grid_100, "Seed 100 (Alternative)")
    
    plt.tight_layout()
    output_img = 'ifo-grokking-v13-pd-confusion.png'
    plt.savefig(output_img, dpi=150)
    print(f"\nAnalysis plot saved to: {output_img}")
    plt.show()
