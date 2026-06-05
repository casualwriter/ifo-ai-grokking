
"""
IFO Grokking Experiment V9: Non-Abelian Dihedral Group D12 (Order 24)
Focus: Testing if a simpler non-abelian group with clear ring-like symmetries 
       exhibits topological phase transitions during grokking.

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
# D12 Dihedral Group Setup (Order 24)
# ============================================================

def get_d12_multiplication_table():
    # 24 elements:
    # 0..11:  r^0 .. r^11  (Rotations)
    # 12..23: s*r^0 .. s*r^11 (Reflections)
    table = np.zeros((24, 24), dtype=np.int64)
    for i in range(24):
        s1 = i // 12  # 0 or 1
        r1 = i % 12   # 0..11
        for j in range(24):
            s2 = j // 12
            r2 = j % 12
            
            # D12 Relations:
            # r^a * r^b = r^(a+b)
            # r^a * (s r^b) = s r^(-a+b)
            # (s r^a) * r^b = s r^(a+b)
            # (s r^a) * (s r^b) = r^(-a+b)
            if s1 == 0 and s2 == 0:
                out_s = 0
                out_r = (r1 + r2) % 12
            elif s1 == 0 and s2 == 1:
                out_s = 1
                out_r = (-r1 + r2) % 12
            elif s1 == 1 and s2 == 0:
                out_s = 1
                out_r = (r1 + r2) % 12
            else: # s1 == 1 and s2 == 1
                out_s = 0
                out_r = (-r1 + r2) % 12
                
            table[i, j] = out_s * 12 + out_r
    return table

def generate_d12_data(train_fraction=0.6, seed=42):
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

# ============================================================
# Model
# ============================================================

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
# Topological Probe
# ============================================================

def get_topology_persistence_raw(points):
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    # Compute up to H1
    result = ripser(points_scaled, maxdim=1)
    dgms = result['dgms']
    
    max_persistence = 0.0
    if len(dgms) > 1 and len(dgms[1]) > 0:
        finite_h1 = dgms[1][dgms[1][:, 1] != np.inf]
        if len(finite_h1) > 0:
            max_persistence = np.max(finite_h1[:, 1] - finite_h1[:, 0])
            
    return max_persistence

# ============================================================
# Single Seed Run
# ============================================================

def run_single_seed_d12(seed, steps=15000, track_interval=300):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data, val_data = generate_d12_data(train_fraction=0.6, seed=seed)
    model = DihedralNet(num_elements=24, hidden_dim=128)
    
    # Moderate Weight Decay for non-abelian structure
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
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
                h1 = get_topology_persistence_raw(embed_weights)
                
                history['step'].append(step)
                history['val_acc'].append(val_acc)
                history['embed_h1'].append(h1)
                
    return history

# ============================================================
# Main Loop
# ============================================================

if __name__ == "__main__":
    seeds = [42, 100, 2026, 7, 999]
    all_results = {}
    
    print("Starting D12 Dihedral Group Experiment...")
    
    for seed in seeds:
        print(f"\n>>> Running D12 Seed {seed}...")
        history = run_single_seed_d12(seed, steps=15000, track_interval=300)
        all_results[seed] = history
        print(f"Seed {seed} Finished. Final Val Acc: {history['val_acc'][-1]:.3f} | Final Raw H1: {history['embed_h1'][-1]:.4f}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    steps = all_results[seeds[0]]['step']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Val Acc
    ax = axes[0]
    for i, seed in enumerate(seeds):
        ax.plot(steps, all_results[seed]['val_acc'], color=colors[i], alpha=0.8, linewidth=2, label=f'Seed {seed}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('D12 Group: Val Acc')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: H1
    ax = axes[1]
    for i, seed in enumerate(seeds):
        ax.plot(steps, all_results[seed]['embed_h1'], color=colors[i], alpha=0.8, linewidth=2, label=f'Seed {seed}')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Max H1 Persistence')
    ax.set_title('D12 Embedding: Raw H1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ifo-grokking-v9-d12dihedral.png', dpi=150)
    print("\nD12 plot saved to: ifo-grokking-v9-d12dihedral.png")
    plt.show()
