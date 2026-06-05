
"""
IFO Grokking V16.3: High-Frequency Collapse Diagnostic
Tracks the micro-dynamics of Seed 1 & 2 between Step 20000 and 24000 
to find the physical trigger of the Step 22000 collective collapse.

CK Hung & Echo, 2026/6/2
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ripser")

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def calculate_ccc(model, table, inverses, device):
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
    return ccc

def calculate_pr_volume(embed_weights):
    centered = embed_weights - np.mean(embed_weights, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues**2)
    return (sum_lambda**2) / sum_lambda_sq

# ============================================================
# Diagnostic Runner
# ============================================================
def run_diagnostic(seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, inverses = get_d12_table_and_inverses()
    train_data, val_data = generate_fixed_d12_data(train_fraction=0.7, split_seed=42)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = DihedralNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data[0].to(device), train_data[1].to(device)
    X_val, y_val = val_data[0].to(device), val_data[1].to(device)
    
    print(f"\n==============================================================")
    print(f"=== DIAGNOSING SEED {seed} ===")
    print(f"==============================================================")
    print(f"{'Step':<6} | {'Loss':<8} | {'ValAcc':<6} | {'CCC':<6} | {'PR':<6} | {'EmbNorm':<8} | {'FC1Norm':<8}")
    print("-" * 70)
    
    for step in range(25001):
        # Determine evaluation frequency
        is_eval_step = False
        if step <= 20000 and step % 2000 == 0:
            is_eval_step = True
        elif 20000 < step <= 24000 and step % 100 == 0:  # High-frequency zoom-in
            is_eval_step = True
        elif step > 24000 and step % 1000 == 0:
            is_eval_step = True
            
        if is_eval_step:
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
                train_logits = model(X_train)
                train_loss = criterion(train_logits, y_train).item()
                
            embed_w = model.embed.weight.detach().cpu().numpy()
            fc1_w = model.fc1.weight.detach().cpu().numpy()
            
            ccc = calculate_ccc(model, table, inverses, device)
            pr = calculate_pr_volume(embed_w)
            
            # Calculate L2 Norms of weights
            emb_norm = np.linalg.norm(embed_w)
            fc1_norm = np.linalg.norm(fc1_w)
            
            print(f"{step:<6d} | {train_loss:<8.5f} | {val_acc:<6.3f} | {ccc:<6.3f} | {pr:<6.2f} | {emb_norm:<8.3f} | {fc1_norm:<8.3f}")
            
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    run_diagnostic(1)
    run_diagnostic(2)
