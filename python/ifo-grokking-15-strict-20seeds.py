
"""
IFO Grokking Experiment V15: Strict 20-Seed Survey with Fixed Split
Focus: 
1. Fixes the Train/Val split to ensure identical evaluation across seeds.
2. Runs 20 initialization seeds.
3. Tracks Val Acc, Raw H1, and Weight Norms at Step 15k and Step 40k.

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
# D12 Setup (Fixed Split)
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

def generate_fixed_d12_data(train_fraction=0.7, split_seed=42):
    # Fixed seed for data generation ensures identical split for all runs
    np.random.seed(split_seed)
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
# Core Metrics
# ============================================================

def get_h1_persistence(embed_weights):
    scaler = StandardScaler()
    embed_scaled = scaler.fit_transform(embed_weights)
    tda_res = ripser(embed_scaled, maxdim=1)
    dgms = tda_res['dgms']
    h1_max = 0.0
    if len(dgms) > 1 and len(dgms[1]) > 0:
        finite_h1 = dgms[1][dgms[1][:, 1] != np.inf]
        if len(finite_h1) > 0:
            h1_max = np.max(finite_h1[:, 1] - finite_h1[:, 0])
    return h1_max

def get_weight_norms(model):
    embed_norm = model.embed.weight.norm(2).item()
    total_norm = sum(p.norm(2).item() for p in model.parameters())
    return embed_norm, total_norm

# ============================================================
# Single Seed Runner
# ============================================================

def run_seed(seed, train_data, val_data):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = DihedralNet(num_elements=24, hidden_dim=128)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Target 1: Step 15,000
    for step in range(15000):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(X_val)
        acc_15k = (val_logits.argmax(dim=1) == y_val).float().mean().item()
    embed_15k = model.embed.weight.detach().cpu().numpy()
    h1_15k = get_h1_persistence(embed_15k)
    emb_norm_15k, tot_norm_15k = get_weight_norms(model)
    
    # Target 2: Step 40,000 (run remaining 25,000 steps)
    for step in range(25000):
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(X_val)
        acc_40k = (val_logits.argmax(dim=1) == y_val).float().mean().item()
    embed_40k = model.embed.weight.detach().cpu().numpy()
    h1_40k = get_h1_persistence(embed_40k)
    emb_norm_40k, tot_norm_40k = get_weight_norms(model)
    
    return {
        "15k": {"acc": acc_15k, "h1": h1_15k, "emb_norm": emb_norm_15k, "tot_norm": tot_norm_15k},
        "40k": {"acc": acc_40k, "h1": h1_40k, "emb_norm": emb_norm_40k, "tot_norm": tot_norm_40k}
    }

# ============================================================
# Main Loop
# ============================================================

if __name__ == "__main__":
    print("Preparing fixed D12 Train/Val split (seed=42)...")
    train_data, val_data = generate_fixed_d12_data(train_fraction=0.7, split_seed=42)
    
    seeds = list(range(1, 21)) # 20 seeds
    results = {}
    
    print("\nStarting 20-Seed Strict Survey...")
    print(f"{'Seed':<5} | {'Acc@15k':<7} | {'H1@15k':<6} | {'EmbN@15k':<8} | {'Acc@40k':<7} | {'H1@40k':<6} | {'EmbN@40k':<8}")
    print("-" * 72)
    
    for seed in seeds:
        res = run_seed(seed, train_data, val_data)
        results[seed] = res
        
        print(f"{seed:<5d} | "
              f"{res['15k']['acc']:<7.4f} | {res['15k']['h1']:<6.4f} | {res['15k']['emb_norm']:<8.2f} | "
              f"{res['40k']['acc']:<7.4f} | {res['40k']['h1']:<6.4f} | {res['40k']['emb_norm']:<8.2f}")

    # Summary Statistics
    accs_15k = [results[s]['15k']['acc'] for s in seeds]
    h1s_15k = [results[s]['15k']['h1'] for s in seeds]
    accs_40k = [results[s]['40k']['acc'] for s in seeds]
    h1s_40k = [results[s]['40k']['h1'] for s in seeds]
    
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Step 15k | Mean Acc: {np.mean(accs_15k):.4f} | Mean H1: {np.mean(h1s_15k):.4f}")
    print(f"Step 40k | Mean Acc: {np.mean(accs_40k):.4f} | Mean H1: {np.mean(h1s_40k):.4f}")
    
    # Count anomalies
    perf_gen_low_h1 = sum(1 for s in seeds if results[s]['15k']['acc'] > 0.98 and results[s]['15k']['h1'] < 0.8)
    acc_drops = sum(1 for s in seeds if results[s]['40k']['acc'] < results[s]['15k']['acc'] - 0.02)
    
    print(f"\nAnomalies detected:")
    print(f"- Seeds with Perfect Gen (>98%) but Low H1 (<0.8) at 15k: {perf_gen_low_h1} / 20")
    print(f"- Seeds with significant Acc drop (>2%) from 15k to 40k: {acc_drops} / 20")
