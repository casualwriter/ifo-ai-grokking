
"""
IFO Grokking V16.2: Step 500 Parity (Z2 Quotient Group) Verification
Tests if the network learns coarse-grained homomorphism before fine-grained isomorphism.

CK Hung & Echo, 2026/6/2
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# D12 Setup
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

def run_parity_diagnostic(seed):
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
    
    # Train to Step 500
    for step in range(501):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        
    # Evaluate at Step 500
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val)
        val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()
        
        # Test on ALL 576 pairs
        all_pairs = torch.tensor([[a, b] for a in range(24) for b in range(24)], dtype=torch.long, device=device)
        logits1 = model(all_pairs)
        preds1 = logits1.argmax(dim=1).cpu().numpy()
        
        # Step 2 inputs
        step2_inputs = []
        true_abs = []
        for idx, (a, b) in enumerate(zip(all_pairs[:, 0].cpu().numpy(), all_pairs[:, 1].cpu().numpy())):
            true_ab = table[a, b]
            true_abs.append(true_ab)
            inv_ab = inverses[true_ab]
            step2_inputs.append([preds1[idx], inv_ab])
            
        step2_inputs = torch.tensor(step2_inputs, dtype=torch.long, device=device)
        logits2 = model(step2_inputs)
        preds2 = logits2.argmax(dim=1).cpu().numpy()
        true_abs = np.array(true_abs)
        
        # Parity Analysis (s = element // 12. 0: Rotation, 1: Reflection)
        parity_preds1 = preds1 // 12
        parity_true_ab = true_abs // 12
        
        # 1. Step 1 Parity Accuracy
        step1_parity_acc = np.mean(parity_preds1 == parity_true_ab)
        
        # 2. Split CCC by parity match
        match_mask = (parity_preds1 == parity_true_ab)
        num_match = np.sum(match_mask)
        num_mismatch = np.sum(~match_mask)
        
        ccc_on_match = np.mean(preds2[match_mask] == 0) if num_match > 0 else 0.0
        ccc_on_mismatch = np.mean(preds2[~match_mask] == 0) if num_mismatch > 0 else 0.0
        overall_ccc = np.mean(preds2 == 0)
        
        print(f"Seed {seed:2d} | ValAcc: {val_acc:.4f} | Overall CCC: {overall_ccc:.4f}")
        print(f"        | S1 Parity Acc: {step1_parity_acc:.4f} (Random is 0.5)")
        print(f"        | CCC on Parity Match ({num_match:3d} pairs): {ccc_on_match:.4f}")
        print(f"        | CCC on Parity Mismatch ({num_mismatch:3d} pairs): {ccc_on_mismatch:.4f}")
        print("-" * 65)

if __name__ == "__main__":
    print("=== Step 500 Parity Verification ===")
    for s in [1, 2, 3, 4]:
        run_parity_diagnostic(s)
