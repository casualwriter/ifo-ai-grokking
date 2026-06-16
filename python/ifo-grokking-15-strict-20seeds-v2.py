"""
IFO Grokking Experiment V15: Strict 20-Seed Survey with Fixed Split
Focus: 
1. Fixes the Train/Val split to ensure identical evaluation across seeds.
2. Runs 20 initialization seeds.
3. Tracks Val Acc, Unified Eq.1 CCC, Raw H1, and Weight Norms at Step 15k and Step 40k.
4. Computes Pearson Correlation for CCC vs ValAcc and H1 vs ValAcc.

Emmy Team (Vector), Amoy Studio. 2026/06/11 (Revised to Unified CCC)
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ripser")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from ripser import ripser

# Device selection for speed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# D12 Setup (Fixed Split & Inverses)
# ============================================================

def get_d12_table_and_inverses():
    table = np.zeros((24, 24), dtype=np.int64)
    inverses = np.zeros(24, dtype=np.int64)
    for i in range(24):
        s1 = i // 12; r1 = i % 12
        if s1 == 0:
            inverses[i] = (12 - r1) % 12
        else:
            inverses[i] = i
            
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
    return table, inverses

def generate_fixed_d12_data(train_fraction=0.7, split_seed=42):
    np.random.seed(split_seed)
    table, _ = get_d12_table_and_inverses()
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
# Core Metrics (Unified Eq. 1 CCC)
# ============================================================

def calculate_ccc(model, inverses):
    """
    Unified 3-parameter Causal Closure Count (CCC) for D12 (Eq. 1).
    Tests if Net(Net(a, b), b^-1) == a for all 576 combinations.
    """
    model.eval()
    all_pairs = torch.tensor([[a, b] for a in range(24) for b in range(24)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits1, _ = model(all_pairs)
        preds1 = logits1.argmax(dim=1).cpu().numpy()
        
        step2_inputs = []
        targets_a = []
        
        for idx, (a, b) in enumerate(zip(all_pairs[:, 0].cpu().numpy(), all_pairs[:, 1].cpu().numpy())):
            inv_b = inverses[b] 
            step2_inputs.append([preds1[idx], inv_b])
            targets_a.append(a)
            
        step2_inputs = torch.tensor(step2_inputs, dtype=torch.long, device=device)
        logits2, _ = model(step2_inputs)
        preds2 = logits2.argmax(dim=1).cpu().numpy()
        
        targets_a = np.array(targets_a)
        ccc = np.sum(preds2 == targets_a) / 576.0
        
    return ccc

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

def pearson_corr(x, y):
    if len(x) < 2:
        return 0.0
    x = np.array(x)
    y = np.array(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return np.corrcoef(x, y)[0, 1]

# ============================================================
# Single Seed Runner
# ============================================================

def run_seed(seed, train_data, val_data, inverses):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = DihedralNet(num_elements=24, hidden_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data[0].to(device), train_data[1].to(device)
    X_val, y_val = val_data[0].to(device), val_data[1].to(device)
    
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
    ccc_15k = calculate_ccc(model, inverses)
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
    ccc_40k = calculate_ccc(model, inverses)
    embed_40k = model.embed.weight.detach().cpu().numpy()
    h1_40k = get_h1_persistence(embed_40k)
    emb_norm_40k, tot_norm_40k = get_weight_norms(model)
    
    return {
        "15k": {"acc": acc_15k, "ccc": ccc_15k, "h1": h1_15k, "emb_norm": emb_norm_15k, "tot_norm": tot_norm_15k},
        "40k": {"acc": acc_40k, "ccc": ccc_40k, "h1": h1_40k, "emb_norm": emb_norm_40k, "tot_norm": tot_norm_40k}
    }

# ============================================================
# Main Loop
# ============================================================

if __name__ == "__main__":
    print(f"Using device: {device}")
    print("Preparing fixed D12 Train/Val split (seed=42)...")
    train_data, val_data = generate_fixed_d12_data(train_fraction=0.7, split_seed=42)
    _, inverses = get_d12_table_and_inverses()
    
    seeds = list(range(1, 21)) # 20 seeds
    results = {}
    
    print("\nStarting 20-Seed Strict Survey (Unified CCC)...")
    print(f"{'Seed':<5} | {'Acc@15k':<7} | {'CCC@15k':<7} | {'H1@15k':<6} | {'EmbN@15k':<8} | {'Acc@40k':<7} | {'CCC@40k':<7} | {'H1@40k':<6} | {'EmbN@40k':<8}")
    print("-" * 90)
    
    for seed in seeds:
        res = run_seed(seed, train_data, val_data, inverses)
        results[seed] = res
        
        print(f"{seed:<5d} | "
              f"{res['15k']['acc']:<7.4f} | {res['15k']['ccc']:<7.4f} | {res['15k']['h1']:<6.4f} | {res['15k']['emb_norm']:<8.2f} | "
              f"{res['40k']['acc']:<7.4f} | {res['40k']['ccc']:<7.4f} | {res['40k']['h1']:<6.4f} | {res['40k']['emb_norm']:<8.2f}")

    # Summary Statistics
    accs_15k = [results[s]['15k']['acc'] for s in seeds]
    cccs_15k = [results[s]['15k']['ccc'] for s in seeds]
    h1s_15k = [results[s]['15k']['h1'] for s in seeds]
    
    accs_40k = [results[s]['40k']['acc'] for s in seeds]
    cccs_40k = [results[s]['40k']['ccc'] for s in seeds]
    h1s_40k = [results[s]['40k']['h1'] for s in seeds]
    
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Step 15k | Mean Acc: {np.mean(accs_15k):.4f} | Mean CCC: {np.mean(cccs_15k):.4f} | Mean H1: {np.mean(h1s_15k):.4f}")
    print(f"Step 40k | Mean Acc: {np.mean(accs_40k):.4f} | Mean CCC: {np.mean(cccs_40k):.4f} | Mean H1: {np.mean(h1s_40k):.4f}")
    
    # Pearson Correlations
    corr_ccc_15k = pearson_corr(accs_15k, cccs_15k)
    corr_h1_15k = pearson_corr(accs_15k, h1s_15k)
    corr_ccc_40k = pearson_corr(accs_40k, cccs_40k)
    corr_h1_40k = pearson_corr(accs_40k, h1s_40k)
    
    print("\n=== CAUSAL VS GEOMETRIC CORRELATIONS ===")
    print(f"Step 15k | Pearson Corr(ValAcc, CCC): {corr_ccc_15k:+.4f}  <-- Causal Probe")
    print(f"Step 15k | Pearson Corr(ValAcc, H1) : {corr_h1_15k:+.4f}  <-- Geometric Probe")
    print(f"Step 40k | Pearson Corr(ValAcc, CCC): {corr_ccc_40k:+.4f}")
    print(f"Step 40k | Pearson Corr(ValAcc, H1) : {corr_h1_40k:+.4f}")