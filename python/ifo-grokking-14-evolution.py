
"""
IFO Grokking Experiment V14: Dynamic Evolution Analysis
Tracks Error Grids and H1 persistence across 5 key training steps:
[5000, 15000, 22500, 30000, 40000] for Seed 7 and Seed 100.

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
# Evolution Tracker
# ============================================================

def track_evolution(seed, target_steps=[5000, 15000, 22500, 30000, 40000]):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_tensors, val_tensors, train_pairs, val_pairs = generate_d12_data(train_fraction=0.7, seed=seed)
    model = DihedralNet(num_elements=24, hidden_dim=128)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_tensors
    X_val, y_val = val_tensors
    
    history = {}
    
    print(f"\nTraining Seed {seed} and capturing snapshots...")
    current_step = 0
    for target in target_steps:
        steps_to_run = target - current_step
        for _ in range(steps_to_run):
            model.train()
            optimizer.zero_grad()
            logits, _ = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
        current_step = target
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val)
            preds = val_logits.argmax(dim=1)
            corrects = (preds == y_val).cpu().numpy()
            val_acc = corrects.mean()
            
        # TDA
        embed_weights = model.embed.weight.detach().cpu().numpy()
        scaler = StandardScaler()
        embed_scaled = scaler.fit_transform(embed_weights)
        tda_res = ripser(embed_scaled, maxdim=1)
        dgms = tda_res['dgms']
        h1_max = 0.0
        if len(dgms) > 1 and len(dgms[1]) > 0:
            finite_h1 = dgms[1][dgms[1][:, 1] != np.inf]
            if len(finite_h1) > 0:
                h1_max = np.max(finite_h1[:, 1] - finite_h1[:, 0])
                
        # Error Grid
        grid = np.zeros((24, 24))
        for (a, b) in train_pairs:
            grid[a, b] = 0 # Train (Gray)
        
        quad_errors = {"Rot-Rot": 0, "Rot-Ref": 0, "Ref-Rot": 0, "Ref-Ref": 0}
        for idx, (a, b) in enumerate(val_pairs):
            is_correct = corrects[idx]
            grid[a, b] = 1 if is_correct else 2 # 1: Correct (White), 2: Error (Red)
            
            if not is_correct:
                if a < 12 and b < 12: quad_errors["Rot-Rot"] += 1
                elif a < 12 and b >= 12: quad_errors["Rot-Ref"] += 1
                elif a >= 12 and b < 12: quad_errors["Ref-Rot"] += 1
                else: quad_errors["Ref-Ref"] += 1
                
        history[target] = {
            "acc": val_acc,
            "h1": h1_max,
            "grid": grid,
            "errors": quad_errors
        }
        print(f"  Step {target:5d} | Val Acc: {val_acc:.4f} | H1: {h1_max:.4f} | Errors: {sum(quad_errors.values())} {list(quad_errors.values())}")
        
    return history

def plot_evolution_grid(axes_row, history, seed_name, target_steps):
    cmap = mcolors.ListedColormap(['#d3d3d3', '#ffffff', '#ff4d4d']) # Gray, White, Red
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    for idx, step in enumerate(target_steps):
        ax = axes_row[idx]
        data = history[step]
        grid = data["grid"]
        
        ax.imshow(grid, cmap=cmap, norm=norm, origin='upper')
        ax.axvline(11.5, color='black', linewidth=1.0, linestyle=':')
        ax.axhline(11.5, color='black', linewidth=1.0, linestyle=':')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Title with key metrics
        ax.set_title(f"Step {step}\nAcc: {data['acc']:.3f} | H1: {data['h1']:.2f}", fontsize=10)
        
        # Print error breakdown at the bottom of each subplot
        err = data["errors"]
        ax.set_xlabel(f"Err: {err['Rot-Rot']}/{err['Rot-Ref']}/{err['Ref-Rot']}/{err['Ref-Ref']}", fontsize=8)

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Starting D12 Dynamic Evolution Analysis...")
    target_steps = [5000, 15000, 22500, 30000, 40000]
    
    hist_7 = track_evolution(seed=7, target_steps=target_steps)
    hist_100 = track_evolution(seed=100, target_steps=target_steps)
    
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    
    # Row 0: Seed 7
    plot_evolution_grid(axes[0], hist_7, "Seed 7", target_steps)
    axes[0, 0].set_ylabel("Seed 7 (Crystallized)\nRow=A, Col=B", fontsize=12, fontweight='bold')
    
    # Row 1: Seed 100
    plot_evolution_grid(axes[1], hist_100, "Seed 100", target_steps)
    axes[1, 0].set_ylabel("Seed 100 (Alternative)\nRow=A, Col=B", fontsize=12, fontweight='bold')
    
    plt.suptitle("D12 Grokking Evolution: Seed 7 vs Seed 100\nError Grid Quadrants: [Rot-Rot / Rot-Ref / Ref-Rot / Ref-Ref]", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_img = 'ifo-grokking-v14-evolution.png'
    plt.savefig(output_img, dpi=150)
    print(f"\nEvolution plot saved to: {output_img}")
    plt.show()
