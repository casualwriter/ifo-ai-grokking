
"""
IFO Grokking V16+: Multi-Seed Diagnostic (Priority 2)
Tracks 20 seeds to check if CCC degradation synchronizes with H1 and Val Acc.
Includes early-stage degeneracy diagnostics (Uniques & Entropy).

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
from scipy.stats import entropy

# ============================================================
# D12 Group Setup
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
        return out

# ============================================================
# Metrics
# ============================================================
def calculate_ccc_and_diagnostics(model, table, inverses, device):
    model.eval()
    all_pairs = torch.tensor([[a, b] for a in range(24) for b in range(24)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits1 = model(all_pairs)
        preds1 = logits1.argmax(dim=1).cpu().numpy()
        
        step2_inputs = []
        for idx, (a, b) in enumerate(zip(all_pairs[:, 0].cpu().numpy(), all_pairs[:, 1].cpu().numpy())):
            true_ab = table[a, b]
            inv_ab = inverses[true_ab]
            step2_inputs.append([preds1[idx], inv_ab])
            
        step2_inputs = torch.tensor(step2_inputs, dtype=torch.long, device=device)
        logits2 = model(step2_inputs)
        preds2 = logits2.argmax(dim=1).cpu().numpy()
        
        ccc = np.sum(preds2 == 0) / 576.0
        
        # Diagnostics
        s1_uniques = len(np.unique(preds1))
        s2_uniques = len(np.unique(preds2))
        _, counts = np.unique(preds2, return_counts=True)
        pk = counts / len(preds2)
        s2_entropy = entropy(pk)
        
    return ccc, s1_uniques, s2_uniques, s2_entropy

def calculate_pr_volume(embed_weights):
    centered = embed_weights - np.mean(embed_weights, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues**2)
    return (sum_lambda**2) / sum_lambda_sq

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

# ============================================================
# Core Runner
# ============================================================
def run_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, inverses = get_d12_table_and_inverses()
    train_data, val_data = generate_fixed_d12_data(train_fraction=0.7, split_seed=42)
    
    seeds = list(range(1, 21))  # Run 20 seeds
    max_steps = 30000
    eval_every = 1000
    
    summary_results = []
    
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = DihedralNet().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
        criterion = nn.CrossEntropyLoss()
        
        X_train, y_train = train_data[0].to(device), train_data[1].to(device)
        X_val, y_val = val_data[0].to(device), val_data[1].to(device)
        
        print(f"\n=== Running Seed {seed} ===")
        print(f"{'Step':<6} | {'ValAcc':<6} | {'CCC':<6} | {'PR':<6} | {'H1':<6} | {'S1_U':<4} | {'S2_U':<4} | {'S2_Ent':<6}")
        print("-" * 65)
        
        # Track specific check points for summary
        grok_step = -1
        degrade_step = -1
        last_val_acc = 0.0
        
        for step in range(max_steps + 1):
            if step % eval_every == 0 or step == 500: # Force eval at 500 for diagnostic
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val)
                    val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
                
                embed_w = model.embed.weight.detach().cpu().numpy()
                ccc, s1_u, s2_u, s2_ent = calculate_ccc_and_diagnostics(model, table, inverses, device)
                pr = calculate_pr_volume(embed_w)
                h1 = get_h1_persistence(embed_w)
                
                print(f"{step:<6d} | {val_acc:<6.3f} | {ccc:<6.3f} | {pr:<6.2f} | {h1:<6.3f} | {s1_u:<4d} | {s2_u:<4d} | {s2_ent:<6.3f}")
                
                # Simple heuristic tracking
                if val_acc >= 0.95 and grok_step == -1:
                    grok_step = step
                if grok_step != -1 and val_acc < 0.90 and degrade_step == -1 and step > grok_step:
                    degrade_step = step
                last_val_acc = val_acc
                
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
            
        summary_results.append({
            'seed': seed,
            'final_acc': last_val_acc,
            'grok_step': grok_step,
            'degrade_step': degrade_step
        })
        
    print("\n" + "="*40 + "\n=== 20 SEEDS SUMMARY ===")
    print(f"{'Seed':<5} | {'Final Acc':<10} | {'Grok Step':<10} | {'Degrade Step':<12}")
    print("-" * 45)
    for r in summary_results:
        g_s = str(r['grok_step']) if r['grok_step'] != -1 else "N/A"
        d_s = str(r['degrade_step']) if r['degrade_step'] != -1 else "N/A"
        print(f"{r['seed']:<5d} | {r['final_acc']:<10.4f} | {g_s:<10} | {d_s:<12}")

if __name__ == "__main__":
    run_experiment()
