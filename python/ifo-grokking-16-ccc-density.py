
"""
IFO Grokking Experiment V16+: Cycle Closure Density & Phase Transition
Tracks and compares Seed 4 (H1-crystallized) vs Seed 100 (H1-flat).
Measures:
1. CCC (Cycle Closure Count): Functional 3-cycle consistency.
2. PR (Participation Ratio): Effective volume of embedding space.
3. Closure Density: CCC / PR.
4. Raw H1 (TDA).

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
# D12 Group Setup
# ============================================================
def get_d12_table_and_inverses():
    table = np.zeros((24, 24), dtype=np.int64)
    inverses = np.zeros(24, dtype=np.int64)
    for i in range(24):
        s1 = i // 12; r1 = i % 12
        # Compute inverse
        if s1 == 0:
            inverses[i] = (12 - r1) % 12
        else:
            inverses[i] = i # Reflection is its own inverse
            
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
# IFO Metrics
# ============================================================
def calculate_ccc(model, table, inverses, device):
    """
    Calculates the fraction of closed 3-cycles: Net(Net(a, b), (a*b)^-1) == e (0)
    """
    model.eval()
    closed_count = 0
    total_cycles = 24 * 24
    
    with torch.no_grad():
        # Step 1: Predict all a * b
        all_pairs = torch.tensor([[a, b] for a in range(24) for b in range(24)], dtype=torch.long, device=device)
        logits1, _ = model(all_pairs)
        preds1 = logits1.argmax(dim=1).cpu().numpy() # shape (576,)
        
        # Step 2: Prepare second step inputs: (preds1, (a*b)^-1)
        step2_inputs = []
        for idx, (a, b) in enumerate(zip(all_pairs[:, 0].cpu().numpy(), all_pairs[:, 1].cpu().numpy())):
            true_ab = table[a, b]
            inv_ab = inverses[true_ab]
            step2_inputs.append([preds1[idx], inv_ab])
            
        step2_inputs = torch.tensor(step2_inputs, dtype=torch.long, device=device)
        logits2, _ = model(step2_inputs)
        preds2 = logits2.argmax(dim=1).cpu().numpy()
        
        # Identity element in D12 is 0
        closed_count = np.sum(preds2 == 0)
        
    return closed_count / total_cycles

def calculate_pr_volume(embed_weights):
    """
    Calculates Participation Ratio (PR) of the embedding weights as effective volume.
    """
    # Center the embeddings
    centered = embed_weights - np.mean(embed_weights, axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.clip(eigenvalues, a_min=1e-10, a_max=None)
    
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues**2)
    pr = (sum_lambda**2) / sum_lambda_sq
    return pr

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
# Training & Tracking Loop
# ============================================================
def monitor_seed(seed, train_data, val_data, table, inverses, device, max_steps=30000, eval_every=200):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = DihedralNet(num_elements=24, hidden_dim=128).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.2)
    criterion = nn.CrossEntropyLoss()
    
    X_train, y_train = train_data[0].to(device), train_data[1].to(device)
    X_val, y_val = val_data[0].to(device), val_data[1].to(device)
    
    history = []
    
    print(f"\n--- Monitoring Seed {seed} ---")
    print(f"{'Step':<6} | {'Val Acc':<7} | {'CCC (Loop)':<10} | {'PR (Vol)':<8} | {'Density':<8} | {'H1':<6}")
    print("-" * 60)
    
    for step in range(max_steps + 1):
        if step % eval_every == 0:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(X_val)
                val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
                
            embed_w = model.embed.weight.detach().cpu().numpy()
            ccc = calculate_ccc(model, table, inverses, device)
            pr = calculate_pr_volume(embed_w)
            density = ccc / pr
            h1 = get_h1_persistence(embed_w)
            
            print(f"{step:<6d} | {val_acc:<7.4f} | {ccc:<10.4f} | {pr:<8.2f} | {density:<8.4f} | {h1:<6.4f}")
            history.append((step, val_acc, ccc, pr, density, h1))
            
        model.train()
        optimizer.zero_grad()
        logits, _ = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    return history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table, inverses = get_d12_table_and_inverses()
    train_data, val_data = generate_fixed_d12_data(train_fraction=0.7, split_seed=42)
    
    # Run Seed 4 (H1-crystallizer)
    history_s4 = monitor_seed(4, train_data, val_data, table, inverses, device, max_steps=25000, eval_every=500)
    
    # Run Seed 100 (H1-flat but generalizes)
    history_s100 = monitor_seed(100, train_data, val_data, table, inverses, device, max_steps=25000, eval_every=500)
