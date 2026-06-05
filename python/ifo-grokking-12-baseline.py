
"""
IFO Grokking Experiment V12: Baseline & Verification
1. Compute baseline H1 persistence of random 128D embeddings (size 24).
2. Print exact Val Acc and H1 for Seed 7 and Seed 100 at Step 40,000.

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

# ============================================================
# 1. Baseline Calculator
# ============================================================

def get_topology_persistence_raw(points):
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    result = ripser(points_scaled, maxdim=1)
    dgms = result['dgms']
    max_persistence = 0.0
    if len(dgms) > 1 and len(dgms[1]) > 0:
        finite_h1 = dgms[1][dgms[1][:, 1] != np.inf]
        if len(finite_h1) > 0:
            max_persistence = np.max(finite_h1[:, 1] - finite_h1[:, 0])
    return max_persistence

def run_random_baseline(num_samples=100, num_elements=24, dim=128):
    h1_list = []
    for _ in range(num_samples):
        # PyTorch Embedding 預設是用 N(0, 1) 初始化
        weights = torch.randn(num_elements, dim).numpy()
        h1 = get_topology_persistence_raw(weights)
        h1_list.append(h1)
    return np.mean(h1_list), np.std(h1_list)

# ============================================================
# 2. D12 Training Setup
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

def train_and_evaluate(seed, steps=40000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_data, val_data = generate_d12_data(train_fraction=0.7, seed=seed)
    model = DihedralNet(num_elements=24, hidden_dim=128)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
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
    h1 = get_topology_persistence_raw(embed_weights)
    
    return val_acc, h1

# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("=== STEP 1: Running Random Baseline (100 trials) ===")
    mean_h1, std_h1 = run_random_baseline(num_samples=100)
    print(f"Random Embedding (24x128) H1 Baseline: {mean_h1:.4f} (std: {std_h1:.4f})")
    print(f"Significance threshold (Mean + 2*Std): {mean_h1 + 2*std_h1:.4f}")
    
    print("\n=== STEP 2: Verifying Seed 7 and Seed 100 ===")
    
    # Seed 7
    acc_7, h1_7 = train_and_evaluate(seed=7)
    print(f"\n[Seed 7]  Final Val Acc: {acc_7:.4f} | Final H1: {h1_7:.4f}")
    
    # Seed 100
    acc_100, h1_100 = train_and_evaluate(seed=100)
    print(f"[Seed 100] Final Val Acc: {acc_100:.4f} | Final H1: {h1_100:.4f}")
